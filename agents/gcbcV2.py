import math
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCDiscreteActor  # discrete는 그대로 유지

# ⬇️ SimbaV2 액터 사용
from agents.simbaV2_network import SimbaV2Actor

# ---- (새) Goal-conditioned Simba actor wrapper ----
class SimbaGCActor(nn.Module):
    """
    기존 GCActor처럼 (obs, goals)를 받아 내부에서 [s; g]를 구성해 SimbaV2Actor에 전달.
    optional gc_encoder가 있으면 그걸로 전처리 후 전달.
    """
    # SimbaV2 하이퍼파라미터
    num_blocks: int
    hidden_dim: int
    action_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    c_shift: float
    # optional encoder (GCEncoder)
    gc_encoder: Any = None

    def setup(self):
        self.actor = SimbaV2Actor(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            c_shift=self.c_shift,
        )
        # gc_encoder가 Module이면 플랙스가 자동으로 서브모듈로 등록함
        self.encoder = self.gc_encoder

    def __call__(self, observations, goals, temperature: float = 1.0):
        if self.encoder is not None:
            # GCEncoder는 (state_encoder/concat_encoder) 조합을 내부에서 처리
            x = self.encoder(observations, goals)
        else:
            x = jnp.concatenate([observations, goals], axis=-1)

        # SimbaV2Actor는 (dist, info)를 반환 -> 기존 인터페이스와 맞추기 위해 dist만 돌려줌
        dist, _ = self.actor(x, temperature=temperature)
        return dist


class GCBCAgentV2(flax.struct.PyTreeNode):
    """Goal-conditioned behavioral cloning (GCBC) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the BC actor loss."""
        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -log_prob.mean()

        actor_info = {
            'actor_loss': actor_loss,
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            # ⚠️ TransformedDistribution 대비: mode/scale_diag 대신 mean/stddev 사용
            pred_mean = dist.mean()
            pred_std = dist.stddev()
            actor_info.update(
                {
                    'mse': jnp.mean((pred_mean - batch['actions']) ** 2),
                    'std': jnp.mean(pred_std),
                }
            )

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # (선택) 비전 인코더 설정
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        # Actor: discrete는 기존, continuous는 SimbaV2 백본
        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            actor_def = SimbaGCActor(
                num_blocks=config['simba2_num_blocks'],
                hidden_dim=config['simba2_hidden_dim'],
                action_dim=action_dim,
                scaler_init=config['simba2_scaler_init'],
                scaler_scale=config['simba2_scaler_scale'],
                alpha_init=config['simba2_alpha_init'],
                alpha_scale=config['simba2_alpha_scale'],
                c_shift=config['simba2_c_shift'],
                gc_encoder=encoders.get('actor'),
            )

        network_info = dict(
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gcbcV2',  # Agent name.
            lr=1e-4,  # Learning rate.
            batch_size=512,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # (discrete 전용) 기존 GCActor용
            discount=0.99,  # (미사용) GCDataset 옵션 호환
            const_std=True,  # (discrete 무관) 기존 필드 유지
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='GCDataset',
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=False,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),

            # ⬇️ SimbaV2 하이퍼파라미터 (continuous 전용)
            simba2_num_blocks=4,
            simba2_hidden_dim=128,
            simba2_scaler_init=2.0/math.sqrt(128),
            simba2_scaler_scale=2.0/math.sqrt(128),
            simba2_alpha_init=1/(1+1),
            simba2_alpha_scale=1.0/math.sqrt(128),
            simba2_c_shift=3.0,
        )
    )
    return config
