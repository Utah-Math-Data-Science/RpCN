import enum

import omegaconf
import sqlalchemy as sa
from hydra_orm import orm


class RewardKind(str, enum.Enum):
    SIGNED_GAUSSIAN_PAIR = 'signed_gaussian_pair'


class GMMExperiment(orm.Table):
    means: str = orm.make_field(orm.ColumnRequired(sa.String), default='0.0')
    stds: str = orm.make_field(orm.ColumnRequired(sa.String), default='3.0')
    weights: str = orm.make_field(orm.ColumnRequired(sa.String), default='1.0')

    reward_kind: RewardKind = orm.make_field(orm.ColumnRequired(sa.Enum(RewardKind)), default=RewardKind.SIGNED_GAUSSIAN_PAIR)
    reward_mu: float = orm.make_field(orm.ColumnRequired(sa.Double), default=3.0)
    reward_sigma: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.5)

    @staticmethod
    def parse_csv_floats(value: str) -> tuple[float, ...]:
        return tuple(float(v.strip()) for v in value.split(',') if v.strip() != '')

    def __post_init__(self):
        means = self.parse_csv_floats(self.means)
        stds = self.parse_csv_floats(self.stds)
        weights = self.parse_csv_floats(self.weights)
        if len(means) == 0:
            raise ValueError('means must contain at least one mode.')
        if len(means) != len(stds):
            raise ValueError('means and stds must have matching lengths.')
        if len(means) != len(weights):
            raise ValueError('means and weights must have matching lengths.')
        if any(s <= 0 for s in stds):
            raise ValueError('All stds must be > 0.')
        if any(w <= 0 for w in weights):
            raise ValueError('All weights must be > 0.')
        if self.reward_kind != RewardKind.SIGNED_GAUSSIAN_PAIR:
            raise ValueError(f'Unsupported reward kind: {self.reward_kind}')
        if self.reward_sigma <= 0:
            raise ValueError('reward_sigma must be > 0.')
