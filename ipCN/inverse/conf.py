import enum
from dataclasses import field
from pathlib import Path

import sqlalchemy as sa
import omegaconf
from omegaconf import OmegaConf
import hydra
import hydra_orm.utils
from hydra_orm import orm

DIR_ROOT = (Path(__file__).parent).resolve()


def get_engine(dir=str(DIR_ROOT), name='runs'):
    return sa.create_engine(f'sqlite+pysqlite:///{dir}/{name}.sqlite')




class Conf(orm.Table):
    root_dir: str = field(default=str(DIR_ROOT.resolve()))
    out_dir: str = field(default=('/mntb/jkellerData/inverse-ipcn'))
    # run_subdir: str = field(default='runs')
    device: str = field(default='cuda')

    alt_id: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True, unique=True), init=False, omegaconf_ignore=True)
    rng_seed: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    img_id: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    operator: str = orm.make_field(orm.ColumnRequired(sa.String(8)), default=omegaconf.MISSING)

    @property
    def run_dir(self):
        path = Path(self.out_dir)/self.alt_id
        path.mkdir(exist_ok=True)
        return path


engine = get_engine()
orm.create_all(engine)
Session = sa.orm.sessionmaker(engine)

sa.event.listens_for(Conf, 'before_insert')(
    hydra_orm.utils.set_attr_to_func_value(Conf, Conf.alt_id.key, hydra_orm.utils.generate_random_string)
)
