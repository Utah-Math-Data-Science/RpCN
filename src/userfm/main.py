from functools import partial
import logging
import pprint
import sys

import hydra
from omegaconf import OmegaConf
import torch
import torch.utils.data.dataloader
import jax
import jax.numpy as jnp
import numpy as np
import orbax
import optax
import diffrax
import lineax
import lightning.pytorch as pl
import spt_guidance.jax.spt as spt
import spt_guidance.jax.fk_steering as fk_steering
import spt_guidance.jax.rpcn as rpcn

import conf.conf
import conf.dataset
import conf.model
import userfm.unet
import userfm.callbacks
import userfm.datasets
import userfm.event_constraints
import userfm.loggers
import userfm.sde_diffusion
import userfm.utils


# turn off orbax info logging
logging.getLogger('absl').setLevel(logging.WARNING)

log = userfm.utils.getLoggerByFilename(__file__)


class JaxLightning(pl.LightningModule):
    def __init__(self, cfg, key, dataloaders, train_data_std, model):
        super().__init__()
        self.automatic_optimization = False

        self.cfg = cfg
        (
            self.key_model_init,
            self.key_train,
            self.key_predict
        ) = jax.random.split(key, 3)
        self.dataloaders = dataloaders
        self.train_data_std = train_data_std
        self.model = model

        if isinstance(self.cfg.get_model().probability_path, conf.model.VarianceExploding):
            self.diffusion = userfm.sde_diffusion.get_sde_diffusion(self.cfg.get_model().probability_path)

        self.ema_ts = self.cfg.get_model().epoch_count / self.cfg.get_model().ema_folding_count

        self.loss_and_grad = jax.value_and_grad(self.loss, argnums=2, has_aux=True)

    def __hash__(self):
        return hash(id(self))

    @property
    def noise_std(self):
        if isinstance(self.cfg.get_model().probability_path, conf.model.VarianceExploding):
            return self.diffusion.sigma(1)
        elif isinstance(self.cfg.get_model().probability_path, conf.model.ConditionalOT):
            return 1
        else:
            raise ValueError(f'Noise std. dev. unknown for probability path: {self.cfg.get_model().probability_path}')

    def setup(self, stage):
        if stage == 'fit':
            key_model_init, self.key_model_init = jax.random.split(self.key_model_init)
            self.params = self.model_init(
                key_model_init,
                (self.cfg.dataset.batch_size, self.cfg.dataset.len_trajectory, self.cfg.dataset.dim),
                self.model
            )
            self.params_ema = self.params
        elif stage == 'validate':
            if self.cfg.predict:
                self.event_constraint = userfm.event_constraints.get_event_constraint(self.cfg.dataset)
                self.times = self.get_sample_times(self.cfg.steering.sampling_step_count)
                dt = self.times[1] - self.times[0]
                # prevent repeated compilation
                if isinstance(self.cfg.steering, conf.model.SourceParallelTempering):
                    def push_forward(y0):
                        return self.event_constraint.potential_fn(self.sample(self.times[0], self.times[-1], dt, y0).ys[-1], power=self.cfg.steering.penalization_power)
                    self.ptmcmc = spt.PTMCMC(
                        spt.KernelPreconditionedCrankNicolson(
                            (self.cfg.dataset.len_trajectory, self.cfg.dataset.dim),
                            std=self.noise_std
                        ),
                        jax.jit(push_forward),
                        jnp.linspace(0, self.cfg.steering.tilt, self.cfg.steering.chain_count),
                        jnp.sin(jnp.linspace(jnp.pi / 2 - 1e-3, .05, self.cfg.steering.chain_count)),
                    )
                elif isinstance(self.cfg.steering, conf.model.FeynmanKac):
                    def propose(key, t0, t1, sample):
                        return self.sample(t0, t1, dt, sample, as_score=True, key=key).ys[-1]
                    if isinstance(self.cfg.steering.intermediate_reward, conf.model.FeynmanKacIntermediateRewardExpectedSample):
                        intermediate_reward = fk_steering.IntermediateReward.EXPECTED_SAMPLE
                        def expected_sample_fn(key, t0, batch):
                            t0 = t0 * jnp.ones((self.cfg.dataset.batch_size, 1, 1))
                            g2_score = 2 * self.velocity(batch, t0, self.params_ema)
                            score_multiplier = self.diffusion.sigma(1 - t0)**2 / self.diffusion.g2(1 - t0)
                            return (batch + score_multiplier * g2_score) / self.diffusion.scale(1 - t0)
                        kwargs_intermediate_reward = dict(
                            expected_sample_fn=expected_sample_fn,
                        )
                    elif isinstance(self.cfg.steering.intermediate_reward, conf.model.FeynmanKacIntermediateRewardSubEnsemblePushforward):
                        intermediate_reward = fk_steering.IntermediateReward.SUB_ENSEMBLE_PUSHFORWARD
                        kwargs_intermediate_reward = dict(
                            sub_ensemble_size=self.cfg.steering.intermediate_reward.sub_ensemble_size,
                            pushforward_fn=lambda key, t0, batch: self.sample(
                                t0, self.times[-1], dt,
                                batch,
                                as_score=True,
                                key=key,
                                virtual_brownian_tree=diffrax.VirtualBrownianTree(t0, 1., tol=1e-3, shape=batch.shape, key=key),
                            ).ys[-1],
                        )
                    else:
                        raise ValueError(f'Unknown Feynman-Kac steering intermediate reward: {self.cfg.steering.intermediate_reward}')
                    inverse_temperature = self.cfg.steering.tilt
                    if self.cfg.steering.chain_count > 1:
                        inverse_temperature = jnp.linspace(
                            0, self.cfg.steering.tilt, self.cfg.steering.chain_count,
                        )
                    self.feynman_kac = fk_steering.FeynmanKac(
                        init_param_fn=lambda key: self.noise_std * jax.random.normal(key, (self.cfg.dataset.len_trajectory, self.cfg.dataset.dim)),
                        reward_fn=partial(self.event_constraint.reward_fn, power=self.cfg.steering.penalization_power),
                        propose_fn=propose,
                        resample_fn=fk_steering.ResampleFn.adaptive_resample,
                        potential=self.cfg.steering.potential,
                        intermediate_reward=intermediate_reward,
                        inverse_temperature=inverse_temperature,
                        kwargs_intermediate_reward=kwargs_intermediate_reward,
                    )
                elif isinstance(self.cfg.steering, conf.model.RpCN):
                    def expected_next_fn(t0, t1, batch):
                        t0 = t0 * jnp.ones((self.cfg.dataset.batch_size, 1, 1))
                        t1 = t1 * jnp.ones((self.cfg.dataset.batch_size, 1, 1))
                        g2_score = 2 * self.velocity(batch, t0, self.params_ema)
                        score_multiplier = self.diffusion.sigma2_from_one_step_forward(1 - t1, 1 - t0) / self.diffusion.g2(1 - t0)
                        return (batch + score_multiplier * g2_score) / self.diffusion.scale(1 - t0)
                    if isinstance(self.cfg.steering.intermediate_reward, conf.model.FeynmanKacIntermediateRewardExpectedSample):
                        intermediate_reward = fk_steering.IntermediateReward.EXPECTED_SAMPLE
                        def expected_sample_fn(key, t0, batch):
                            t0 = t0 * jnp.ones((self.cfg.dataset.batch_size, 1, 1))
                            g2_score = 2 * self.velocity(batch, t0, self.params_ema)
                            score_multiplier = self.diffusion.sigma(1 - t0)**2 / self.diffusion.g2(1 - t0)
                            return (batch + score_multiplier * g2_score) / self.diffusion.scale(1 - t0)
                        kwargs_intermediate_reward = dict(
                            expected_sample_fn=expected_sample_fn,
                        )
                    elif isinstance(self.cfg.steering.intermediate_reward, conf.model.FeynmanKacIntermediateRewardSubEnsemblePushforward):
                        intermediate_reward = fk_steering.IntermediateReward.SUB_ENSEMBLE_PUSHFORWARD
                        kwargs_intermediate_reward = dict(
                            sub_ensemble_size=self.cfg.steering.intermediate_reward.sub_ensemble_size,
                            pushforward_fn=lambda key, t0, batch: self.sample(
                                t0, self.times[-1], dt,
                                batch,
                                as_score=True,
                                key=key,
                                virtual_brownian_tree=diffrax.VirtualBrownianTree(t0, 1., tol=1e-3, shape=batch.shape, key=key),
                            ).ys[-1],
                        )
                    else:
                        raise ValueError(f'Unknown intermediate reward: {self.cfg.steering.intermediate_reward}')
                    self.rpcn = rpcn.ResidualMCMCSteering(
                        rpcn.KernelPreconditionedCrankNicolson(
                            (self.cfg.dataset.len_trajectory, self.cfg.dataset.dim),
                        ),
                        jnp.sin(jnp.linspace(jnp.pi / 2 - 1e-3, .05, self.cfg.steering.chain_count)),
                        inverse_temperature=jnp.linspace(0, self.cfg.steering.tilt, self.cfg.steering.chain_count),
                        reward_fn=partial(self.event_constraint.reward_fn, power=self.cfg.steering.penalization_power),
                        expected_next_fn=expected_next_fn,
                        std_fn=lambda t0, t1=None: jnp.sqrt(self.diffusion.sigma(1 - t0)**2 - self.diffusion.sigma(1 - t1)**2) if t1 is not None else self.diffusion.sigma(1 - t0),
                        intermediate_reward=intermediate_reward,
                        kwargs_intermediate_reward=kwargs_intermediate_reward,
                    )
        else:
            raise ValueError(f'Unknown stage: {stage}')

    def model_init(self, key, x_shape, model):
        x = jnp.ones(x_shape)
        t = jnp.ones(x_shape[0])
        params = model.init(key, x=x, t=t, train=False)
        return params

    def configure_optimizers(self):
        learning_rate_scheduler = optax.exponential_decay(
            init_value=self.cfg.get_model().lr,
            transition_steps=512,  # Number of steps before decay
            decay_rate=self.cfg.get_model().lr_decay,  # Decay factor
            staircase=True  # Whether to use staircase decay
        )
        self.optimizer = optax.adam(learning_rate=learning_rate_scheduler)
        self.opt_state = self.optimizer.init(self.params)

    def train_dataloader(self):
        return self.dataloaders['train']

    def training_step(self, batch, batch_idx):
        key_training_step, self.key_train = jax.random.split(self.key_train)
        loss, monitors, self.params, self.params_ema, self.opt_state = self.step(
            key_training_step, batch,
            self.params, self.params_ema,
            self.opt_state,
        )
        # use same key_training_step to ensure identical sampling
        loss_ema, monitors_ema = self.loss(key_training_step, batch, self.params_ema)
        self.optimizers().step()  # increment global step for PyTorch Lightning logging and checkpointing
        outputs = dict(
            loss=loss,
            loss_ema=loss_ema,
            monitors=monitors,
            monitors_ema=monitors_ema,
        )
        return jax.tree.map(lambda x: torch.tensor(x.item()), outputs)

    def val_dataloader(self):
        if self.cfg.fit:
            return self.dataloaders['val']
        elif self.cfg.predict:
            return self.predict_dataloader()

    def validation_step(self, batch, batch_idx):
        if self.cfg.fit:
            if self.trainer.sanity_checking:
                return dict(loss_val=torch.tensor(-1.))
            else:
                return dict(loss_val=self.trainer.callback_metrics['train_loss_ema'])
        elif self.cfg.predict:
            return self.predict_step(batch, batch_idx)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            jax.random.split(self.key_predict, self.cfg.steering.batch_count),
            collate_fn=lambda x: x[0],
        )

    def predict_step(self, key, batch_idx):
        if isinstance(self.cfg.steering, conf.model.NoSteering):
            (
                key_initial,
                key_solver,
            ) = jax.random.split(key)
            t0 = self.times[0]
            t1 = self.times[-1]
            dt = self.times[1] - self.times[0]
            batch = self.sample(
                t0, t1, dt,
                self.noise_std * jax.random.normal(key, (self.cfg.dataset.batch_size, self.cfg.dataset.len_trajectory, self.cfg.dataset.dim)),
                as_score=self.cfg.steering.solver is conf.model.Solver.EULER_MARUYAMA,
                key=key_solver,
            ).ys[-1]
        elif isinstance(self.cfg.steering, conf.model.SourceParallelTempering):
            tempered_noise = self.ptmcmc.run(
                key,
                self.cfg.steering.update_count,
                batch_size=self.cfg.dataset.batch_size,
            )[-1]
            t0 = self.times[0]
            t1 = self.times[-1]
            dt = self.times[1] - self.times[0]
            batch = self.sample(t0, t1, dt, tempered_noise).ys[-1]
        elif isinstance(self.cfg.steering, conf.model.FeynmanKac):
            final_particles, _ = self.feynman_kac.run(
                key, self.times, self.cfg.steering.ensemble_size,
                batch_size=self.cfg.dataset.batch_size
            )
            if self.cfg.steering.chain_count > 1:
                final_particles = final_particles[-1]
            batch = fk_steering.SelectFn.highest_reward(
                final_particles,
                partial(self.event_constraint.reward_fn, power=self.cfg.steering.penalization_power),
            )
        elif isinstance(self.cfg.steering, conf.model.RpCN):
            batch = self.rpcn.run(
                key,
                self.times,
                pcn_should_update_fn=lambda time_step, *_: time_step >= len(self.times) // 3.5 ,
                pcn_update_count=self.cfg.steering.pcn_update_count,
                batch_size=self.cfg.dataset.batch_size,
            )[-1]
        else:
            raise ValueError(f'Unknown steering: {self.cfg.steering}')
        self._predict_state = dict(batch=batch, event_constraint=self.event_constraint)

    @partial(jax.jit, static_argnames=['self', 'train'])
    def velocity(self, x, t, params, train=False):
        if (
            isinstance(self.cfg.get_model().probability_path, conf.model.VarianceExploding)
            and
            self.cfg.get_model().probability_path.finzi_karras_weighting
        ):
            # scaling is equivalent to that in Karras et al. https://arxiv.org/abs/2206.00364
            sigma, scale = self.diffusion.sigma(1 - t), self.diffusion.scale(1 - t)
            # Karras et al. $c_in$ and $s(t)$ of EDM.
            input_scale = 1 / jnp.sqrt(sigma**2 + (scale * self.train_data_std)**2)
            out = self.model.apply(params, x=x * input_scale, t=t.squeeze((1, 2)), train=train)
            # Karras et al. the demonimator of $c_out$ of EDM; where is the numerator?
            return out / jnp.sqrt(sigma**2 + (scale * self.train_data_std)**2)
        else:
            return self.model.apply(params, x=x, t=t.squeeze((1, 2)), train=train)

    @partial(jax.jit, static_argnames=['self'])
    def conditional_ot(self, t, x_noise, x_data):
        mean_scale, std = t, 1 - t
        xt = std * x_noise + mean_scale * x_data
        velocity_target = x_data - x_noise
        eps = 1e-6
        return dict(
            xt=xt,
            mean_scale=mean_scale, std=std,
            velocity_target=velocity_target,
            dx_velocity_target=-1 / (std + eps),
            dx_log_pt=-(xt - mean_scale * x_data) / (std + eps)**2,
        )

    @partial(jax.jit, static_argnames=['self'])
    def variance_exploding_conditional(self, t, x_noise, x_data):
        mean_scale, std = jnp.ones_like(t), self.diffusion.sigma(1 - t)
        eps = 1e-6
        # add eps here to make equal to divisor in velocity_target
        xt = x_data + (std + eps) * x_noise
        dt_std = self.diffusion.dsigma(1 - t)
        dx_velocity_target = -dt_std / (std + eps)
        velocity_target = dx_velocity_target * (xt - x_data)
        return dict(
            xt=xt,
            mean_scale=1., std=std,
            velocity_target=velocity_target,
            dx_velocity_target=dx_velocity_target,
            dx_log_pt=-(xt - mean_scale * x_data) / (std + eps)**2,
            dt_std=dt_std,
        )

    @partial(jax.jit, static_argnames=['self'])
    def loss(self, key, x_data, params):
        (
            key_time,
            key_noise,
        ) = jax.random.split(key, 2)
        if isinstance(self.cfg.get_model().probability_path, conf.model.VarianceExploding):
            u0 = jax.random.uniform(key_time)
            u = jnp.remainder(u0 + jnp.linspace(0, 1, x_data.shape[0]), 1)
            t = u * (1 - self.diffusion.tmin) + self.diffusion.tmin
            t = t[:, None, None]
        else:
            t = jax.random.uniform(key_time, shape=(x_data.shape[0], 1, 1))

        x_noise = jax.random.normal(key_noise, x_data.shape)

        if isinstance(self.cfg.get_model().probability_path, conf.model.ConditionalOT):
            context = self.conditional_ot(t, x_noise, x_data)
            weighting = 1.
        elif isinstance(self.cfg.get_model().probability_path, conf.model.VarianceExploding):
            context = self.variance_exploding_conditional(t, x_noise, x_data)
            weighting = 1 / context['dt_std']**2
        else:
            raise ValueError(f'Unknown conditional flow: {self.cfg.get_model().conditional_flow}')

        velocity_pred = self.velocity(context['xt'], t, params, train=True)
        flow_loss = ((velocity_pred - context['velocity_target'])**2 * weighting).mean()

        monitors = {'flow_loss': flow_loss}

        return flow_loss, monitors

    @partial(jax.jit, static_argnames=['self'])
    def step(self, key, batch, params, params_ema, opt_state):
        (loss, monitors), grads = self.loss_and_grad(key, batch, params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        ema_update = lambda p, ema: ema + (p - ema) / self.ema_ts
        params_ema = jax.tree.map(ema_update, params, params_ema)
        return loss, monitors, params, params_ema, opt_state

    @partial(jax.jit, static_argnames=['self', 'as_score'])
    def sample(self, t0, t1, dt, y0, as_score=False, key=None, virtual_brownian_tree=None, params=None, saveat=diffrax.SaveAt(t1=True)):
        if params is None:
            params = self.params_ema
        if as_score:
            if not isinstance(self.cfg.get_model().probability_path, conf.model.VarianceExploding):
                raise ValueError(
                    f'Sampling using the score function is only implemented for a VarianceExploding probability path, not {self.cfg.get_model().probability_path.__class__.__name__}. '
                    'Please choose a different probability path or set as_score=False.'
                )
            if key is None:
                raise ValueError('Sampling using the score function requires a PRNG key. Please pass a key as a keyword argument.')
            def g2_score(t, y, args):
                if not hasattr(t, 'shape') or not t.shape:
                    t = jnp.ones((y0.shape[0], 1, 1)) * t
                return 2 * self.velocity(y, t, params)
            def diffusion(t, y, args):
                return lineax.DiagonalLinearOperator(self.diffusion.diffusion(None, None, 1 - t))
            if virtual_brownian_tree is None:
               virtual_brownian_tree = diffrax.VirtualBrownianTree(t0, t1, tol=1e-3, shape=y0.shape, key=key)
            return diffrax.diffeqsolve(
                diffrax.MultiTerm(
                    diffrax.ODETerm(g2_score),
                    diffrax.ControlTerm(
                        diffusion,
                        virtual_brownian_tree,
                    ),
                ),
                solver=diffrax.Euler(),
                t0=t0, dt0=dt, t1=t1,
                y0=y0,
                saveat=saveat,
            )
        else:
            def velocity(t, y, args):
                if not hasattr(t, 'shape') or not t.shape:
                    t = jnp.ones((y0.shape[0], 1, 1)) * t
                return self.velocity(y, t, params)
            return diffrax.diffeqsolve(
                diffrax.ODETerm(vector_field=velocity),
                diffrax.Heun(),
                t0=t0, dt0=dt, t1=t1,
                y0=y0,
                saveat=saveat,
            )

    def get_sample_times(self, sample_step_count):
        time_min = 0.
        if isinstance(self.cfg.get_model().probability_path, conf.model.VarianceExploding):
            time_min = self.cfg.get_model().probability_path.time_min
        return (time_min + jnp.arange(sample_step_count)) / sample_step_count


@hydra.main(**userfm.utils.HYDRA_INIT)
def main(cfg: conf.conf.TrainingAndEvaluation):
    with conf.conf.Session() as db:
        cfg = conf.conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        log.info('Command: python %s', ' '.join(sys.argv[:-1]))
        log.info(pprint.pformat(cfg))
        log.info('Output directory: %s', cfg.run_dir)

        log.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
        log.info('JAX devices: %r', jax.devices())

        splits = ('train', 'val', 'test')
        (
            *key_dataset_splits,
            key_jax_lightning,
        ) = jax.random.split(
            jax.random.key(userfm.utils.RNG_RANDBITS[cfg.rng_seed]),
            len(splits) + 1
        )

        splits = {
            s: userfm.datasets.get_dataset(cfg.dataset, k, batch_count)
            for s, k in zip(splits, key_dataset_splits)
            if (batch_count := getattr(cfg.dataset, f'batch_count_{s}')) > 0
        }
        if cfg.fit:
            train_data_std = splits['train'][:].std()
            np.save(cfg.run_dir/'train_data_std.npy', np.array(train_data_std))
        else:
            train_data_std = np.load(cfg.model.conf.run_dir/'train_data_std.npy')
        log.info('Training set standard deviation: %(data_std).7f', dict(data_std=train_data_std))
        dataloaders = {
            split: torch.utils.data.DataLoader(
                v,
                batch_sampler=torch.utils.data.BatchSampler(
                    torch.utils.data.SequentialSampler(v),
                    cfg.dataset.batch_size, drop_last=True,
                ),
                collate_fn=lambda x: x,
            )
            for split, v in splits.items()
        }
        dataloaders['predict'] = dataloaders.get(cfg.predict_split)

        cfg_unet = userfm.unet.unet_64_config(
            cfg.dataset.dim,
            base_channels=cfg.get_model().base_channel_count,
            attention=cfg.get_model().use_attention,
        )
        model = userfm.unet.UNet(cfg_unet)
        jax_lightning = JaxLightning(cfg, key_jax_lightning, dataloaders, train_data_std, model)
        if isinstance(cfg.model, conf.conf.Trained):
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            ckpt_path = cfg.model.conf.run_dir/cfg.model.ckpt_filename
            jax_lightning.params = orbax_checkpointer.restore(ckpt_path)
            jax_lightning.params_ema = orbax_checkpointer.restore(ckpt_path.parent/f'{ckpt_path.stem}_ema{ckpt_path.suffix}')

        logger = userfm.loggers.CSVLogger(cfg.run_dir, name=None)

        pl_trainer = pl.Trainer(
            max_epochs=cfg.get_model().epoch_count,
            logger=logger,
            precision=32,
            callbacks=[
                userfm.callbacks.ModelCheckpoint(
                    dirpath=cfg.run_dir,
                    filename='{epoch}',
                    # save_top_k=1,
                    # monitor='train_loss_ema',
                    save_last='link',
                    every_n_epochs=100,
                    save_on_train_epoch_end=False,
                    enable_version_counter=False,
                ),
                userfm.callbacks.LogStats() if cfg.fit else userfm.callbacks.LogPredictStats(cfg.log_constraint_values),
            ],
            log_every_n_steps=1,
            check_val_every_n_epoch=1 if cfg.predict else 50,
            deterministic=True,
        )

        if cfg.fit:
            pl_trainer.fit(jax_lightning)
        elif cfg.predict:
            # use validation instead because it enables logging
            pl_trainer.validate(jax_lightning)


if __name__ == '__main__':
    last_override, run_dir = conf.conf.get_run_dir()
    conf.conf.set_run_dir(last_override, run_dir)
    main()
