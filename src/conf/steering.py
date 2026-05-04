import sqlalchemy as sa
from hydra_orm import orm


class SteeringBase(orm.InheritableTable):
    pass


class Unsteered(SteeringBase):
    pass


class FK(SteeringBase):
    n_chains: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=4)
    inverse_temperature: float = orm.make_field(orm.ColumnRequired(sa.Double), default=25.0)
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
    n_chains: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=4)
    pcn_update_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=5)
    inverse_temperature: float = orm.make_field(orm.ColumnRequired(sa.Double), default=25.0)
    check_every: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1)

    def __post_init__(self):
        if self.n_chains < 1:
            raise ValueError('RpCN n_chains must be >= 1.')
        if self.pcn_update_count < 1:
            raise ValueError('RpCN pcn_update_count must be >= 1.')
        if self.check_every < 1:
            raise ValueError('RpCN check_every must be >= 1.')


class MH(SteeringBase):
    n_chains: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=64)
    num_samples: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2000)
    burn_in: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2000)
    thinning: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2)
    proposal_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.4)
    inverse_temperature: float = orm.make_field(orm.ColumnRequired(sa.Double), default=25.0)

    def __post_init__(self):
        if self.n_chains < 1:
            raise ValueError('MH n_chains must be >= 1.')
        if self.num_samples < 1:
            raise ValueError('MH num_samples must be >= 1.')
        if self.burn_in < 0:
            raise ValueError('MH burn_in must be >= 0.')
        if self.thinning < 1:
            raise ValueError('MH thinning must be >= 1.')
        if self.proposal_std <= 0:
            raise ValueError('MH proposal_std must be > 0.')
        if self.inverse_temperature < 0:
            raise ValueError('MH inverse_temperature must be >= 0.')
