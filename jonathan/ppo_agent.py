# PPO Agent — Proximal Policy Optimization
# Paper: Schulman et al. 2017  (https://arxiv.org/abs/1707.06347)
# GAE:   Schulman et al. 2015  (https://arxiv.org/abs/1506.02438)
#
# Architecture:
#   Shared CNN Backbone  →  512-dim feature vector
#                              ↙            ↘
#                        Actor Head      Critic Head
#                        P(action|s)      V(s)
#
# Loss:
#   L = L_CLIP  -  c1 * L_VF  +  c2 * L_ENT
#
# CarRacing-v3 discrete actions (continuous=False):
#   0: do nothing  1: steer left  2: steer right  3: gas  4: brake

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from jonathan.cnn import CNNFeatureExtractor

# ── Hyperparameters ────────────────────────────────────────────────────────────

CLIP_EPSILON   = 0.2    # how much the policy is allowed to change per update
GAMMA          = 0.99   # discount factor — how much future rewards matter
GAE_LAMBDA     = 0.95   # GAE smoothing — tradeoff between bias and variance
ENTROPY_COEF   = 0.05   # c2 — encourages exploration by keeping the policy spread out
VALUE_COEF     = 0.5    # c1 — how much the value loss contributes
N_EPOCHS       = 4      # passes over the rollout buffer per update
MINIBATCH_SIZE = 512    # samples per gradient step  →  4096/512 × 4 = 32 updates total
KL_TARGET      = 0.02   # early stopping: abort epoch if KL divergence exceeds this
N_ACTIONS      = 5      # CarRacing-v3 discrete action space


# ── Rollout Buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores one rollout (fixed number of steps) of experience.
    PPO is on-policy — the buffer is cleared after every update.
    """

    def __init__(self):
        self.states    = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.values    = []
        self.dones     = []

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ── PPO Agent ──────────────────────────────────────────────────────────────────

class PPOAgent(nn.Module):
    """
    Actor-Critic agent with shared CNN backbone.

    The CNN is trained end-to-end — gradients from both the actor loss
    and the critic loss flow back through the shared feature extractor.
    """

    def __init__(self):
        super().__init__()

        # Shared visual encoder — both heads see the same features
        self.cnn = CNNFeatureExtractor()

        # Actor head — small init so early logits are near-zero → uniform policy
        self.actor = nn.Linear(512, N_ACTIONS)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        # Critic head — orthogonal init, standard gain
        self.critic = nn.Linear(512, 1)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, state: torch.Tensor):
        """
        state: (batch, 4, 96, 96)
        returns: action_logits (batch, 5),  value (batch, 1)
        """
        features = self.cnn(state)                  # (batch, 512)
        action_logits = self.actor(features)        # (batch, 5)
        value = self.critic(features)               # (batch, 1)
        return action_logits, value

    # ── Action Sampling ───────────────────────────────────────────────────────

    @torch.no_grad()
    def get_action(self, state: torch.Tensor):
        """
        Samples an action from the current policy.
        No gradients needed here — this runs during environment interaction.

        returns: action (int), log_prob (scalar tensor), value (scalar tensor)
        """
        logits, value = self.forward(state)
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        # squeeze to scalars so buffer contains plain 0-dim tensors — no shape bugs
        return action.item(), log_prob.squeeze(), value.squeeze()

    # ── GAE — Generalised Advantage Estimation ────────────────────────────────

    def compute_gae(
        self,
        rewards:    list,
        values:     list,
        dones:      list,
        next_value: torch.Tensor,
        device:     torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Schulman et al. 2015 — computes advantages and discounted returns.

        A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
        δ_t = r_t + γ·V(s_{t+1}) - V(s_t)   ← TD error

        High λ → less bias, more variance  (closer to Monte Carlo)
        Low  λ → more bias, less variance  (closer to TD)
        """
        # Convert to plain float lists — no tensor shape confusion
        values_np = [v.item() for v in values]
        next_val  = next_value.squeeze().item()
        values_ext = values_np + [next_val]

        advantages_list = []
        gae = 0.0

        # Walk backwards through the rollout
        for t in reversed(range(len(rewards))):
            mask  = 1.0 - float(dones[t])           # 0 if terminal, 1 otherwise
            delta = rewards[t] + GAMMA * values_ext[t + 1] * mask - values_ext[t]
            gae   = delta + GAMMA * GAE_LAMBDA * mask * gae
            advantages_list.insert(0, gae)

        advantages = torch.tensor(advantages_list, dtype=torch.float32, device=device)
        values_t   = torch.tensor(values_np,       dtype=torch.float32, device=device)
        returns    = advantages + values_t           # A + V = Q ≈ return  (shape: T,)

        # Normalise advantages — stabilises training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    # ── PPO Update ────────────────────────────────────────────────────────────

    def update(
        self,
        buffer:     RolloutBuffer,
        next_value: torch.Tensor,
        optimizer:  torch.optim.Optimizer,
        device:     torch.device,
    ) -> dict:
        """
        Runs N_EPOCHS gradient updates on the collected rollout.

        PPO loss (Schulman et al. 2017, eq. 9):
          L = L_CLIP  -  c1 * L_VF  +  c2 * L_ENT
        """
        # ── Prepare batch tensors ──────────────────────────────────────────
        states     = torch.cat(buffer.states).to(device)                          # (T, 4, 96, 96)
        actions    = torch.tensor(buffer.actions, dtype=torch.long).to(device)    # (T,)
        old_lp     = torch.stack(buffer.log_probs).detach().squeeze().to(device)  # (T,)
        old_values = torch.stack(buffer.values).detach().squeeze().to(device)     # (T,)

        advantages, returns = self.compute_gae(
            buffer.rewards, buffer.values, buffer.dones, next_value, device
        )

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        n_updates         = 0
        T                 = states.shape[0]

        # ── N_EPOCHS passes, each with shuffled minibatches ───────────────
        for epoch in range(N_EPOCHS):

            # Shuffle indices so each minibatch sees different transitions
            indices = torch.randperm(T, device=device)

            for start in range(0, T, MINIBATCH_SIZE):
                mb_idx = indices[start : start + MINIBATCH_SIZE]

                mb_states      = states[mb_idx]
                mb_actions     = actions[mb_idx]
                mb_old_lp      = old_lp[mb_idx]
                mb_old_values  = old_values[mb_idx]
                mb_advantages  = advantages[mb_idx]
                mb_returns     = returns[mb_idx]

                logits, mb_values = self.forward(mb_states)
                mb_values = mb_values.squeeze()
                dist      = Categorical(logits=logits)
                new_lp    = dist.log_prob(mb_actions)
                entropy   = dist.entropy().mean()

                ratio = torch.exp(new_lp - mb_old_lp)

                # ── KL early stopping — abort epoch if policy changed too much ──
                # KL ≈ mean(old_lp - new_lp)  (reverse KL approximation)
                kl = (mb_old_lp - new_lp).mean().item()
                if kl > KL_TARGET:
                    break

                l_clip = torch.min(
                    ratio * mb_advantages,
                    torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * mb_advantages,
                ).mean()

                # ── Clipped value loss — prevents excessively large value updates ──
                # (same clipping idea as the policy loss, applied to the value function)
                v_clipped = mb_old_values + (mb_values - mb_old_values).clamp(
                    -CLIP_EPSILON, CLIP_EPSILON
                )
                l_vf = torch.max(
                    F.mse_loss(mb_values, mb_returns),
                    F.mse_loss(v_clipped, mb_returns),
                )

                loss = -l_clip + VALUE_COEF * l_vf - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                optimizer.step()

                total_policy_loss += l_clip.item()
                total_value_loss  += l_vf.item()
                total_entropy     += entropy.item()
                n_updates         += 1

            else:
                # Inner loop completed without KL break — continue to next epoch
                continue
            # Inner loop hit KL break — stop all epochs
            break

        buffer.clear()

        n_updates = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss":  total_value_loss  / n_updates,
            "entropy":     total_entropy     / n_updates,
            "n_updates":   n_updates,
        }
