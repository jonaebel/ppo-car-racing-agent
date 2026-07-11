# Evaluation — lädt trainierte Weights und zeigt den Agent live
#
# Zeigt drei Fenster:
#   1. CarRacing Spiel (gymnasium render)
#   2. Die 4 gestackten Grayscale-Frames die das CNN sieht
#   3. Action-Wahrscheinlichkeiten als Balkendiagramm (live)

#python -m shared.eval agent.pt

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from shared.wrapper import CarRacingEnv
from jonathan.ppo_agent import PPOAgent

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

ACTION_LABELS = ["Nothing", "Left", "Right", "Gas", "Brake"]


# ── Visualisation Setup ────────────────────────────────────────────────────────

def build_dashboard() -> tuple:
    """
    Baut das Live-Dashboard mit 3 Bereichen:
      - Oben:         4 CNN-Input Frames nebeneinander
      - Unten links:  Action-Wahrscheinlichkeiten als Balkendiagramm
      - Unten rechts: Reward-Verlauf der aktuellen Episode
    """
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor("#1e1e1e")
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.4, wspace=0.3)

    # 4 Frame-Plots oben
    frame_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for i, ax in enumerate(frame_axes):
        ax.set_title(f"Frame t-{3 - i}", color="white", fontsize=8)
        ax.axis("off")
        ax.set_facecolor("#1e1e1e")

    # Balkendiagramm unten links (3 Spalten breit)
    ax_bar = fig.add_subplot(gs[1, :3])
    ax_bar.set_facecolor("#2d2d2d")
    ax_bar.set_title("Action Probabilities", color="white", fontsize=9)
    ax_bar.tick_params(colors="white")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#555555")

    bars = ax_bar.bar(
        ACTION_LABELS,
        [0.2] * 5,
        color=["#4c9be8", "#4c9be8", "#4c9be8", "#4c9be8", "#4c9be8"],
    )
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("P(action)", color="white", fontsize=8)

    # Reward-Verlauf unten rechts (2 Spalten breit)
    ax_reward = fig.add_subplot(gs[1, 3:])
    ax_reward.set_facecolor("#2d2d2d")
    ax_reward.set_title("Episode Reward", color="white", fontsize=9)
    ax_reward.tick_params(colors="white")
    ax_reward.set_xlabel("Step", color="white", fontsize=8)
    ax_reward.set_ylabel("Cumulative Reward", color="white", fontsize=8)
    for spine in ax_reward.spines.values():
        spine.set_edgecolor("#555555")

    reward_line, = ax_reward.plot([], [], color="#e87c4c", linewidth=1.5)

    plt.ion()
    plt.show()

    return fig, frame_axes, bars, ax_bar, ax_reward, reward_line


def update_dashboard(
    fig,
    frame_axes,
    bars,
    ax_bar,
    ax_reward,
    reward_line,
    state:        torch.Tensor,
    probs:        np.ndarray,
    chosen:       int,
    reward_hist:  list,
):
    """Aktualisiert alle drei Dashboard-Bereiche mit den aktuellen Werten."""

    # ── Frames ────────────────────────────────────────────────────────────────
    frames = state.squeeze(0).cpu().numpy()         # (4, 96, 96)
    for i, ax in enumerate(frame_axes):
        ax.clear()
        ax.imshow(frames[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Frame t-{3 - i}", color="white", fontsize=8)
        ax.axis("off")

    # ── Action Bars ───────────────────────────────────────────────────────────
    highlight = ["#e8c44c" if i == chosen else "#4c9be8" for i in range(5)]
    for bar, prob, color in zip(bars, probs, highlight):
        bar.set_height(prob)
        bar.set_color(color)
    ax_bar.set_ylim(0, 1)

    # ── Reward Line ───────────────────────────────────────────────────────────
    if reward_hist:
        reward_line.set_data(range(len(reward_hist)), reward_hist)
        ax_reward.relim()
        ax_reward.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()


# ── Eval Loop ─────────────────────────────────────────────────────────────────

def evaluate(weights_path: str = "agent.pt", n_episodes: int = 3):
    """
    Lädt die trainierten Weights und führt n_episodes aus.
    CarRacing rendert sich selbst (render_mode="human"),
    das Dashboard zeigt zusätzlich CNN-Input und Action-Probs.
    """

    # ── Model laden ───────────────────────────────────────────────────────────
    if not os.path.exists(weights_path):
        print(f"Fehler: '{weights_path}' nicht gefunden. Erst trainieren: python -m shared.wrapper")
        return

    agent = PPOAgent().to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    # Checkpoint ist ein Dict — nur den model-Teil laden
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    agent.load_state_dict(state_dict)
    agent.eval()
    episode_num = checkpoint.get("episode", "?")
    best_reward = checkpoint.get("best_reward", "?")
    print(f"Weights geladen: {weights_path}  (Episode {episode_num}, bestes Reward: {best_reward})")

    # ── Env mit render=True (gymnasium öffnet sein eigenes Fenster) ───────────
    env = CarRacingEnv(render=True)

    # ── Dashboard aufbauen ────────────────────────────────────────────────────
    fig, frame_axes, bars, ax_bar, ax_reward, reward_line = build_dashboard()

    for episode in range(n_episodes):
        state        = env.reset()
        episode_over = False
        total_reward = 0.0
        reward_hist  = []
        step         = 0

        print(f"\nEpisode {episode + 1}/{n_episodes}")

        while not episode_over:
            with torch.no_grad():
                logits, value = agent.forward(state)
                probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
                action = int(probs.argmax())            # greedy — kein Sampling beim Eval

            state, reward, episode_over, _ = env.step(action)
            total_reward += reward
            reward_hist.append(total_reward)
            step += 1

            # Dashboard alle 4 Steps aktualisieren (nicht jeden Frame — zu langsam)
            if step % 4 == 0:
                update_dashboard(
                    fig, frame_axes, bars, ax_bar, ax_reward, reward_line,
                    state, probs, action, reward_hist,
                )

        print(f"  Total Reward: {total_reward:.1f}  |  Steps: {step}")

    env.close()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "best.pt"
    evaluate(weights_path=path)
