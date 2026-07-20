from dataclasses import field
import enum
from pathlib import Path
import sys
from typing import Any, List

import hydra
import hydra_orm.utils
import omegaconf
from omegaconf import OmegaConf
import sqlalchemy as sa
from hydra_orm import orm

import conf.dataset
import conf.model
import userfm.utils


def get_engine(dir=str(userfm.utils.DIR_ROOT), name='runs'):
    return sa.create_engine(f'sqlite+pysqlite:///{dir}/{name}.sqlite')


engine = get_engine()
orm.create_all(engine)
Session = sa.orm.sessionmaker(engine)


def get_run_dir(hydra_init=userfm.utils.HYDRA_INIT, commit=True, engine_name='runs'):
    if '-m' in sys.argv or '--multirun' in sys.argv:
        raise ValueError("The flags '-m' and '--multirun' are not supported. Use GNU parallel instead.")
    with hydra.initialize(version_base=hydra_init['version_base'], config_path=hydra_init['config_path']):
        last_override = None
        overrides = []
        for i, a in enumerate(sys.argv):
            if '=' in a:
                overrides.append(a)
                last_override = i
        cfg = hydra.compose(hydra_init['config_name'], overrides=overrides)
        engine = get_engine(name=engine_name)
        orm.create_all(engine)
        with sa.orm.Session(engine, expire_on_commit=False) as db:
            cfg = orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
            # if commit and '-c' not in sys.argv:
            if commit:
                db.commit()
                cfg.run_dir.mkdir(exist_ok=True)
            return last_override, str(cfg.run_dir)


def set_run_dir(last_override, run_dir):
    run_dir_override = f'hydra.run.dir={run_dir}'
    if last_override is None:
        sys.argv.append(run_dir_override)
    else:
        sys.argv.insert(last_override + 1, run_dir_override)


class Conf(orm.InheritableTable):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(dataset=omegaconf.MISSING),
        dict(model=omegaconf.MISSING),
        '_self_',
    ])
    root_dir: str = field(default=str(userfm.utils.DIR_ROOT.resolve()))
    out_dir: str = field(default=str((userfm.utils.DIR_ROOT/'..'/'..'/'out'/'rpcn-user-defined-events').resolve()))
    run_subdir: str = field(default='runs')
    prediction_filename: str = field(default='output')
    device: str = field(default='cuda')

    alt_id: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True, unique=True), init=False, omegaconf_ignore=True)
    rng_seed: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2376999025)

    @property
    def run_dir(self):
        return Path(self.out_dir)/self.run_subdir/self.alt_id


sa.event.listens_for(Conf, 'before_insert', propagate=True)(
    hydra_orm.utils.set_attr_to_func_value(Conf, Conf.alt_id.key, hydra_orm.utils.generate_random_string)
)


class PredictSplit(enum.StrEnum):
    VAL = 'val'
    TEST = 'test'


class TrainingAndEvaluation(Conf):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(dataset=omegaconf.MISSING),
        dict(model=omegaconf.MISSING),
        dict(steering=omegaconf.MISSING),
        '_self_',
    ])
    fit: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)
    predict: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    predict_split: PredictSplit = orm.make_field(orm.ColumnRequired(sa.Enum(PredictSplit)), default=PredictSplit.VAL)
    log_constraint_values: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)

    dataset = orm.OneToManyField(conf.dataset.Dataset, default=omegaconf.MISSING)
    model = orm.OneToManyField(conf.model.Model, default=omegaconf.MISSING)
    steering = orm.OneToManyField(conf.model.Steering, default=omegaconf.MISSING)

    def get_model(self):
        if isinstance(self.model, Trained):
            return self.model.conf.model
        else:
            return self.model

    def __post_init__(self):
        if not (self.fit ^ self.predict):
            raise ValueError(f'Please set either fit=true (currently {self.fit=}) or predict=true (currently {self.predict=}).')


class Trained(conf.model.Model):
    conf = orm.OneToManyField(Conf, default=omegaconf.MISSING, enforce_element_type=False)
    ckpt_filename: str = orm.make_field(orm.ColumnRequired(sa.String(len('epoch_####'))), default='last')

    @staticmethod
    def transform_conf(session, conf_alt_id):
        if conf_alt_id == omegaconf.MISSING:
            raise ValueError('Please set a conf alt_id with model.conf=<conf_alt_id>.')
        conf = session.query(Conf).filter_by(alt_id=conf_alt_id).first()
        assert conf is not None
        return conf


orm.store_config(TrainingAndEvaluation)
for dataset in (conf.dataset.GaussianMixture, conf.dataset.Lorenz63, conf.dataset.FitzHughNagumo):
    orm.store_config(dataset, group=TrainingAndEvaluation.dataset.key, name=f'_{dataset.__name__}')
for model in (Trained, conf.model.FlowMatching,):
    orm.store_config(model, group=TrainingAndEvaluation.model.key)
for probability_path in (conf.model.ConditionalOT, conf.model.VarianceExploding):
    orm.store_config(probability_path, group=f'{TrainingAndEvaluation.model.key}/{conf.model.FlowMatching.probability_path.key}')
for steering in (conf.model.NoSteering, conf.model.SourceParallelTempering, conf.model.FeynmanKac, conf.model.RpCN):
    orm.store_config(steering, group=TrainingAndEvaluation.steering.key)
for intermediate_reward in (conf.model.FeynmanKacIntermediateRewardExpectedSample, conf.model.FeynmanKacIntermediateRewardSubEnsemblePushforward):
    orm.store_config(intermediate_reward, group=f'{TrainingAndEvaluation.steering.key}/{conf.model.FeynmanKac.intermediate_reward.key}')
