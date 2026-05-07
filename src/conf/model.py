import enum
from typing import Any, List

import omegaconf
import hydra_orm
from hydra_orm import orm
import sqlalchemy as sa


class ProbabilityPath(orm.InheritableTable):
    pass


class ConditionalOT(ProbabilityPath):
    pass


class VarianceExploding(ProbabilityPath):
    time_min: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-3)
    sigma_min: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-3)
    sigma_max: float = orm.make_field(orm.ColumnRequired(sa.Double), default=300.)
    finzi_karras_weighting: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)


class Model(orm.InheritableTable):
    pass


class FlowMatching(Model):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(probability_path=omegaconf.MISSING),
        '_self_',
    ])
    epoch_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    lr: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)
    lr_decay: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)
    ema_folding_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    base_channel_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    use_attention: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=omegaconf.MISSING)
    probability_path = orm.OneToManyField(ProbabilityPath, default=omegaconf.MISSING)


class Steering(orm.InheritableTable):
    batch_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    sampling_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    penalization_power: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)
    symmetric_binary_reward: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)


class Solver(enum.StrEnum):
    EULER_MARUYAMA = 'euler_maruyama'
    HEUN = 'heun'


class NoSteering(Steering):
    solver: Solver = orm.make_field(orm.ColumnRequired(sa.Enum(Solver)), default=Solver.HEUN)


class SourceParallelTempering(Steering):
    chain_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    update_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    tilt: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)


class FeynmannKacPotential(enum.StrEnum):
    DIFFERENCE = 'difference'
    MAX = 'max'
    SUM = 'sum'


class FeynmannKacIntermediateReward(orm.InheritableTable):
    pass


class FeynmannKacIntermediateRewardExpectedSample(FeynmannKacIntermediateReward):
    pass


class FeynmannKacIntermediateRewardSubEnsemblePushforward(FeynmannKacIntermediateReward):
    sub_ensemble_size: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)


class FeynmannKac(Steering):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(intermediate_reward=omegaconf.MISSING),
        '_self_',
    ])
    ensemble_size: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    potential: FeynmannKacPotential = orm.make_field(orm.ColumnRequired(sa.Enum(FeynmannKacPotential)), default=omegaconf.MISSING)
    intermediate_reward = orm.OneToManyField(FeynmannKacIntermediateReward, default=omegaconf.MISSING)
    tilt: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)


class RpCN(Steering):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(intermediate_reward=omegaconf.MISSING),
        '_self_',
    ])
    intermediate_reward = orm.OneToManyField(FeynmannKacIntermediateReward, default=omegaconf.MISSING)
    chain_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    pcn_update_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    tilt: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)
