import sys
from dataclasses import field
from pathlib import Path
from typing import Any

import hydra
import hydra_orm.utils
import omegaconf
from omegaconf import OmegaConf
import sqlalchemy as sa
from hydra_orm import orm

import conf.setting
import conf.steering
import toy.utils


def get_engine(dir: str = str(toy.utils.DIR_ROOT), name: str = 'runs'):
    return sa.create_engine(f'sqlite+pysqlite:///{dir}/{name}.sqlite')


engine = get_engine()
orm.create_all(engine)
Session = sa.orm.sessionmaker(engine)


def get_run_dir(hydra_init=toy.utils.HYDRA_INIT, commit: bool = True, engine_name: str = 'runs'):
    if '-m' in sys.argv or '--multirun' in sys.argv:
        raise ValueError("The flags '-m' and '--multirun' are not supported. Use GNU parallel instead.")
    with hydra.initialize(version_base=hydra_init['version_base'], config_path=hydra_init['config_path']):
        last_override = None
        overrides = []
        for i, arg in enumerate(sys.argv):
            if '=' in arg:
                overrides.append(arg)
                last_override = i
        cfg = hydra.compose(hydra_init['config_name'], overrides=overrides)
        engine_local = get_engine(name=engine_name)
        orm.create_all(engine_local)
        with sa.orm.Session(engine_local, expire_on_commit=False) as db:
            cfg = orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
            if commit:
                db.commit()
                cfg.run_dir.mkdir(exist_ok=True, parents=True)
            return last_override, str(cfg.run_dir)


def set_run_dir(last_override, run_dir):
    run_dir_override = f'hydra.run.dir={run_dir}'
    if last_override is None:
        sys.argv.append(run_dir_override)
    else:
        sys.argv.insert(last_override + 1, run_dir_override)


class EnsembleScoreModel(orm.Table):
    n_steps: int = field(default=50)
    vmap_chunk_size: int = field(default=8192)
    epsilon_alpha: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-3)
    epsilon_beta: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-5)
    sampling_max_score_norm: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1000.0)
    n_reference: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1024)


class Conf(orm.Table):
    defaults: list[Any] = hydra_orm.utils.make_defaults_list([
        {'steering': omegaconf.MISSING},
        '_self_',
    ])

    root_dir: str = field(default=str(toy.utils.DIR_ROOT.resolve()))
    out_dir: str = field(default=str((toy.utils.DIR_ROOT/'../..').resolve()))
    run_subdir: str = field(default='runs')
    device: str = field(default='cuda')
    seed: int = field(default=31)
    batch_size: int = field(default=1024)

    alt_id: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True, unique=True), init=False, omegaconf_ignore=True)
    setting: conf.setting.GMMExperiment = orm.OneToManyField(conf.setting.GMMExperiment, default_factory=conf.setting.GMMExperiment)
    steering: conf.steering.SteeringBase = orm.OneToManyField(conf.steering.SteeringBase, required=True, default=omegaconf.MISSING)
    score_model: EnsembleScoreModel = orm.OneToManyField(EnsembleScoreModel, default_factory=EnsembleScoreModel)

    @property
    def run_dir(self):
        return Path(self.out_dir) / self.run_subdir / self.alt_id


sa.event.listens_for(Conf, 'before_insert')(
    hydra_orm.utils.set_attr_to_func_value(Conf, Conf.alt_id.key, hydra_orm.utils.generate_random_string)
)


orm.store_config(Conf)
orm.store_config(conf.setting.GMMExperiment, group=Conf.setting.key, name='_GMMExperiment')
orm.store_config(conf.steering.Unsteered, group=Conf.steering.key, name='_Unsteered')
orm.store_config(conf.steering.FK, group=Conf.steering.key, name='_FK')
orm.store_config(conf.steering.RpCN, group=Conf.steering.key, name='_RpCN')
