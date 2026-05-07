from dataclasses import field

import omegaconf
from hydra_orm import orm
import sqlalchemy as sa

import userfm.utils


log = userfm.utils.getLoggerByFilename(__file__)


class Dataset(orm.InheritableTable):
    device_batch_size: int = field(default=500)

    batch_size: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    batch_count_train: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    batch_count_val: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    batch_count_test: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)


class GaussianMixture(Dataset):
    time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)


class DynamicalSystem(Dataset):
    time_step: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)
    time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    time_step_count_drop_first: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)

    odeint_rtol: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)

    @property
    def len_trajectory(self):
        return self.time_step_count - self.time_step_count_drop_first

    @property
    def dim(self):
        raise NotImplementedError()

    def __post_init__(self):
        if self.len_trajectory != 60:
            log.warning(
                'Finzi et al., 2023, trim the trajectories to include only first 60 time steps after the "burn-in" time steps, but these trajectories have %(len_trajectory)d time steps.'
                ' Consider setting dataset.time_step_count equal to dataset.time_step_count_drop_first + 60.',
                dict(len_trajectory=self.len_trajectory)
            )


class Lorenz63(DynamicalSystem):
    # Lorenz system parameters
    rho: float = orm.make_field(orm.ColumnRequired(sa.Double), default=28.)
    sigma: float = orm.make_field(orm.ColumnRequired(sa.Double), default=10.)
    beta: float = orm.make_field(orm.ColumnRequired(sa.Double), default=8/3)
    # Scale strange attractor
    rescaling: float = orm.make_field(orm.ColumnRequired(sa.Double), default=20.)

    @property
    def dim(self):
        return 3


class FitzHughNagumo(DynamicalSystem):
    a1: float = orm.make_field(orm.ColumnRequired(sa.Double), default=-.025794)
    a2: float = orm.make_field(orm.ColumnRequired(sa.Double), default=-.025794)
    b1: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.0065)
    b2: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.0135)
    c1: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.02)
    c2: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.02)
    k: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.128)
    coupling12: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.)
    coupling21: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.)

    @property
    def dim(self):
        return 4
