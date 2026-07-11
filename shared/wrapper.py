import os
import gymnasium as gym
import numpy as np
import torch
from collections import deque

from jonathan.ppo_agent import PPOAgent, RolloutBuffer

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

ROLLOUT_STEPS = 4096   # how many steps to collect before each update
LR_INITIAL    = 3e-4   # starting learning rate — annealed linearly to 0


# ── Environment Wrapper ────────────────────────────────────────────────────────

class CarRacingEnv:
    """
    Wraps CarRacing-v3 and handles:
      - Grayscale conversion  (96x96x3 → 96x96x1)
      - Frame stacking        (4 consecutive frames → 4x96x96)
      - Pixel normalisation   (uint8 0-255 → float32 0-1)
    """

    N_STACK = 4
    MAX_GRASS_STEPS = 60   # Episode endet wenn Agent 60 Steps am Stück auf dem Gras ist

    def __init__(self, render: bool = False):
        render_mode = "human" if render else "rgb_array"
        self._env = gym.make(
            "CarRacing-v3",
            render_mode=render_mode,
            lap_complete_percent=0.95,
            domain_randomize=False,
            continuous=False,
        )
        self._frames = deque(maxlen=self.N_STACK)
        self._grass_steps = 0   # zählt aufeinanderfolgende Off-Track-Steps

    # -- helpers ---------------------------------------------------------------

    def _preprocess(self, obs: np.ndarray) -> np.ndarray:
        """96x96x3 uint8  →  96x96 float32 (grayscale, normalised)"""
        gray = obs[..., 0] * 0.299 + obs[..., 1] * 0.587 + obs[..., 2] * 0.114
        return (gray / 255.0).astype(np.float32)

    def _get_state(self) -> torch.Tensor:
        """Stack the last N frames into a (1, 4, 96, 96) tensor."""
        stacked = np.stack(self._frames, axis=0)          # (4, 96, 96)
        return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(device)

    # -- public API ------------------------------------------------------------

    def reset(self) -> torch.Tensor:
        obs, _ = self._env.reset()
        frame = self._preprocess(obs)
        for _ in range(self.N_STACK):                     # fill stack with same frame
            self._frames.append(frame)
        self._grass_steps = 0
        return self._get_state()

    def step(self, action: int):
        total_reward = 0.0
        terminated   = False
        truncated    = False
        for _ in range(4):                                # Frame Skipping: gleiche Aktion 4x
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        # Off-Track-Erkennung: CarRacing gibt -0.1/step auf dem Gras
        # Reward < -0.05 nach Frame-Skip (4 Steps × -0.1 = -0.4) → wahrscheinlich Gras
        if total_reward < -0.05:
            self._grass_steps += 1
        else:
            self._grass_steps = 0

        # Episode abbrechen wenn zu lange auf dem Gras — verhindert Kreis-Exploitation
        if self._grass_steps >= self.MAX_GRASS_STEPS:
            truncated = True

        done = terminated or truncated

        # Grund im info-Dict mitgeben
        if terminated:
            info["end_reason"] = "lap"        # lap_complete_percent erreicht
        elif self._grass_steps >= self.MAX_GRASS_STEPS:
            info["end_reason"] = "grass"      # zu lange off-track
        elif truncated:
            info["end_reason"] = "truncated"  # Zeitlimit
        else:
            info["end_reason"] = "running"

        self._frames.append(self._preprocess(obs))
        return self._get_state(), total_reward, done, info

    @property
    def n_actions(self) -> int:
        return self._env.action_space.n

    def close(self):
        self._env.close()


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path: str, agent, optimizer, episode: int, best_reward: float):
    torch.save({
        "episode":      episode,
        "best_reward":  best_reward,
        "model":        agent.state_dict(),
        "optimizer":    optimizer.state_dict(),
    }, path)
    print(f"  Checkpoint gespeichert: {path}  (Episode {episode})")


def load_checkpoint(path: str, agent, optimizer) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    agent.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    episode     = checkpoint["episode"]
    best_reward = checkpoint["best_reward"]
    print(f"Checkpoint geladen: {path}  (weiter ab Episode {episode + 1}, bisher bestes Reward: {best_reward:.1f})")
    return episode, best_reward


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(
    n_episodes:       int   = 1000,
    checkpoint_path:  str   = "checkpoint.pt",
    checkpoint_every: int   = 50,       # alle N Episoden speichern
    total_steps:      int   = 5_000_000,  # LR-Annealing: LR → 0 over this many env steps
):
    env       = CarRacingEnv(render=False)
    agent     = PPOAgent().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR_INITIAL)
    buffer    = RolloutBuffer()

    # ── Checkpoint laden falls vorhanden (Fallback auf best.pt) ──────────────
    start_episode = 0
    best_reward   = float("-inf")

    load_path = checkpoint_path if os.path.exists(checkpoint_path) else "best.pt"
    if os.path.exists(load_path):
        if load_path != checkpoint_path:
            print(f"checkpoint.pt nicht gefunden — lade best.pt als Fallback")
        start_episode, best_reward = load_checkpoint(load_path, agent, optimizer)
        start_episode += 1

    step = 0

    for episode in range(start_episode, n_episodes):
        state        = env.reset()
        episode_over = False
        total_reward = 0.0

        while not episode_over:
            action, log_prob, value = agent.get_action(state)
            next_state, reward, episode_over, info = env.step(action)

            buffer.store(state, action, log_prob, reward / 10.0, value, episode_over)
            state         = next_state
            total_reward += reward
            step         += 1

            if step % ROLLOUT_STEPS == 0:
                # LR-Annealing: linear decay from LR_INITIAL to 0
                lr_frac = max(1.0 - step / total_steps, 0.0)
                for pg in optimizer.param_groups:
                    pg["lr"] = LR_INITIAL * lr_frac

                with torch.no_grad():
                    _, next_value = agent.forward(state)
                losses = agent.update(buffer, next_value, optimizer, device)
                print(
                    f"  Update — "
                    f"policy: {losses['policy_loss']:.4f}  "
                    f"value: {losses['value_loss']:.4f}  "
                    f"entropy: {losses['entropy']:.4f}  "
                    f"updates: {losses['n_updates']}"
                )

        end_reason = info.get("end_reason", "?")
        is_best = total_reward > best_reward
        if is_best:
            best_reward = total_reward
            # Altes best.pt sichern bevor es überschrieben wird
            if os.path.exists("best.pt"):
                import shutil
                shutil.copy2("best.pt", "best_prev.pt")
            save_checkpoint("best.pt", agent, optimizer, episode, best_reward)

        print(f"Episode {episode + 1}/{n_episodes}  |  Reward: {total_reward:.1f}  |  Best: {best_reward:.1f}  |  [{end_reason}]{'  *' if is_best else ''}")

        # Regelmäßiger Checkpoint
        if (episode + 1) % checkpoint_every == 0:
            save_checkpoint(checkpoint_path, agent, optimizer, episode, best_reward)

    # Finaler Checkpoint am Ende
    save_checkpoint(checkpoint_path, agent, optimizer, n_episodes - 1, best_reward)
    env.close()


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    train(n_episodes=n)
