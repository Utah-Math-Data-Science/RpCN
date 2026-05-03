import sqlalchemy as sa
from hydra_orm import orm


class SteeringBase(orm.InheritableTable):
    pass


class Unsteered(SteeringBase):
    pass


class FK(SteeringBase):
    n_chains: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=20)
    inverse_temperature: float = orm.make_field(orm.ColumnRequired(sa.Double), default=40.0)
    resample_threshold: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.5)
    check_every: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1)

    def __post_init__(self):
        if self.n_chains < 1:
            raise ValueError('FK n_chains must be >= 1.')
        if not (0.0 <= self.resample_threshold <= 1.0):
            raise ValueError('FK resample_threshold must be in [0, 1].')
        if self.check_every < 1:
            raise ValueError('FK check_every must be >= 1.')


class RpCN(SteeringBase):
    n_chains: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=20)
    pcn_update_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=5)
    inverse_temperature: float = orm.make_field(orm.ColumnRequired(sa.Double), default=40.0)
    check_every: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1)

    def __post_init__(self):
        if self.n_chains < 1:
            raise ValueError('RpCN n_chains must be >= 1.')
        if self.pcn_update_count < 1:
            raise ValueError('RpCN pcn_update_count must be >= 1.')
        if self.check_every < 1:
            raise ValueError('RpCN check_every must be >= 1.')
