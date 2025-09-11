import flax.linen as nn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from agents.simbaV2_layer import (
    HyperCategoricalValue,
    HyperEmbedder,
    HyperLERPBlock,
    HyperNormalTanhPolicy,
)

tfd = tfp.distributions
tfb = tfp.bijectors


class SimbaV2Actor(nn.Module):
    num_blocks: int
    hidden_dim: int
    action_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    c_shift: float

    def setup(self):
        self.embedder = HyperEmbedder(
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            c_shift=self.c_shift,
        )
        self.encoder = nn.Sequential(
            [
                HyperLERPBlock(
                    hidden_dim=self.hidden_dim,
                    scaler_init=self.scaler_init,
                    scaler_scale=self.scaler_scale,
                    alpha_init=self.alpha_init,
                    alpha_scale=self.alpha_scale,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.predictor = HyperNormalTanhPolicy(
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
    ) -> tfd.Distribution:
        x = observations
        y = self.embedder(x)
        z = self.encoder(y)
        dist, info = self.predictor(z, temperature)

        return dist, info


class SimbaV2Critic(nn.Module):
    num_blocks: int
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    c_shift: float
    num_bins: int
    min_v: float
    max_v: float

    def setup(self):
        self.embedder = HyperEmbedder(
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            c_shift=self.c_shift,
        )
        self.encoder = nn.Sequential(
            [
                HyperLERPBlock(
                    hidden_dim=self.hidden_dim,
                    scaler_init=self.scaler_init,
                    scaler_scale=self.scaler_scale,
                    alpha_init=self.alpha_init,
                    alpha_scale=self.alpha_scale,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.predictor = HyperCategoricalValue(
            hidden_dim=self.hidden_dim,
            num_bins=self.num_bins,
            min_v=self.min_v,
            max_v=self.max_v,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        x = jnp.concatenate((observations, actions), axis=1)
        y = self.embedder(x)
        z = self.encoder(y)
        q, info = self.predictor(z)
        return q, info


class SimbaV2DoubleCritic(nn.Module):
    """
    Vectorized Double-Q for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3
    """

    num_blocks: int
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    c_shift: float
    num_bins: int
    min_v: float
    max_v: float

    num_qs: int = 2

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        VmapCritic = nn.vmap(
            SimbaV2Critic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )

        qs, infos = VmapCritic(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            c_shift=self.c_shift,
            num_bins=self.num_bins,
            min_v=self.min_v,
            max_v=self.max_v,
        )(observations, actions)

        return qs, infos


class SimbaV2Temperature(nn.Module):
    initial_value: float = 0.01

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            name="log_temp",
            init_fn=lambda key: jnp.full(
                shape=(), fill_value=jnp.log(self.initial_value)
            ),
        )
        return jnp.exp(log_temp)


# --- 아래 내용을 simbaV2_network.py 맨 아래에 추가하세요 ---

import flax.linen as nn
import jax.numpy as jnp
from agents.simbaV2_layer import HyperEmbedder, HyperLERPBlock, HyperMLP, HyperCategoricalValue  # :contentReference[oaicite:2]{index=2}

class SimbaV2GoalRep(nn.Module):
    """
    φ([s; g]) 생성을 SimbaV2 블록으로 구현.
    출력은 L2 정규화된 rep_dim 차원 벡터.
    """
    num_blocks: int
    hidden_dim: int
    rep_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    c_shift: float

    def setup(self):
        self.embedder = HyperEmbedder(
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            c_shift=self.c_shift,
        )
        self.encoder = nn.Sequential([
            HyperLERPBlock(
                hidden_dim=self.hidden_dim,
                scaler_init=self.scaler_init,
                scaler_scale=self.scaler_scale,
                alpha_init=self.alpha_init,
                alpha_scale=self.alpha_scale,
            ) for _ in range(self.num_blocks)
        ])
        # HyperMLP는 마지막에 l2normalize까지 포함됨
        self.head = HyperMLP(
            hidden_dim=self.hidden_dim,
            out_dim=self.rep_dim,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def __call__(self, s_and_g: jnp.ndarray) -> jnp.ndarray:
        y = self.embedder(s_and_g)
        z = self.encoder(y)
        rep = self.head(z)  # l2 normalized
        return rep


class SimbaV2Value(nn.Module):
    """
    HIQL의 V(s, φ([s; g]))를 SimbaV2로 구현 (카테고리컬 밸류 헤드 사용).
    반환: (value_scalar, info={'log_prob': ...})
    """
    num_blocks: int
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    c_shift: float
    num_bins: int
    min_v: float
    max_v: float

    def setup(self):
        self.embedder = HyperEmbedder(
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            c_shift=self.c_shift,
        )
        self.encoder = nn.Sequential([
            HyperLERPBlock(
                hidden_dim=self.hidden_dim,
                scaler_init=self.scaler_init,
                scaler_scale=self.scaler_scale,
                alpha_init=self.alpha_init,
                alpha_scale=self.alpha_scale,
            ) for _ in range(self.num_blocks)
        ])
        self.predictor = HyperCategoricalValue(
            hidden_dim=self.hidden_dim,
            num_bins=self.num_bins,
            min_v=self.min_v,
            max_v=self.max_v,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def __call__(self, s_and_phi: jnp.ndarray):
        y = self.embedder(s_and_phi)
        z = self.encoder(y)
        v, info = self.predictor(z)
        return v, info


class SimbaV2DoubleValue(nn.Module):
    """
    HIQL이 기대하는 (v1, v2) 더블 밸류 출력을 위한 vmap 래퍼.
    """
    num_blocks: int
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    c_shift: float
    num_bins: int
    min_v: float
    max_v: float
    num_vs: int = 2

    @nn.compact
    def __call__(self, s_and_phi: jnp.ndarray):
        VmapValue = nn.vmap(
            SimbaV2Value,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_vs,
        )
        vs, infos = VmapValue(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            c_shift=self.c_shift,
            num_bins=self.num_bins,
            min_v=self.min_v,
            max_v=self.max_v,
        )(s_and_phi)
        # HIQL은 (v1, v2) 스칼라 튜플이 필요
        v1, v2 = vs[0], vs[1]
        return (v1, v2)
