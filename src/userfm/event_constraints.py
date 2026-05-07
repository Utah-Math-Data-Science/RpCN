import jax.numpy as jnp

import conf.dataset


class InequalityConstraint:
    @classmethod
    def statistic(cls, x):
        raise NotImplementedError()

    @classmethod
    def constraint(cls, x):
        raise NotImplementedError()

    @classmethod
    def satisfies_constraint(cls, x):
        return cls.constraint(x) > 0

    @classmethod
    def reward_fn(cls, x, power=0, dtype=jnp.float32):
        c = cls.constraint(x)
        satisfies_constraint = (c > 0).astype(dtype)
        return (
            (satisfies_constraint - 1) * jnp.power(jnp.abs(c) + 1, power)
            + satisfies_constraint
        )
        return cls.satisfies_constraint(x).astype(dtype)

    @classmethod
    def potential_fn(cls, x, power=0, dtype=jnp.float32):
        return -cls.reward_fn(x, power=power, dtype=dtype)


class Lorenz63(InequalityConstraint):
    threshold = -.6

    @classmethod
    def statistic(cls, x):
        fourier_magnitudes = jnp.abs(jnp.fft.rfft(x[..., 0], axis=-1))
        return -fourier_magnitudes[..., 1:].mean(-1)

    @classmethod
    def constraint(cls, x):
        return cls.statistic(x) - cls.threshold


class FitzHughNagumo(InequalityConstraint):
    threshold = 2.5

    @classmethod
    def statistic(cls, x):
        return jnp.max(x[..., :2].mean(-1), -1)

    @classmethod
    def constraint(cls, x):
        return cls.statistic(x) - cls.threshold


def get_event_constraint(cfg: conf.dataset.Dataset):
    match cfg:
        case conf.dataset.Lorenz63():
            return Lorenz63
        case conf.dataset.FitzHughNagumo():
            return FitzHughNagumo
        case _:
            raise ValueError(f'No event constraint for dataset: {cfg}')
