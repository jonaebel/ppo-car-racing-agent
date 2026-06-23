# PPO Car Racing Agent

This repository is a continuation of my ML learning journey after building a DQN model for a 2D Pokémon game ([dqn-combat-agent](https://github.com/jonaebel/dqn-combat-agent)). With that project still being fine-tuned, I decided to take the next step and explore PPO (Proximal Policy Optimization).

Unlike DQN, which learns by estimating which action minimizes the loss (Q-values), PPO is a policy gradient method, it directly optimizes a policy that maps observations to actions, without needing an explicit value lookup or action classification step.

To learn this, I am using the [Gymnasium](https://gymnasium.farama.org/) framework, which provides a clean environment wrapper so I can focus on designing the model rather than managing the simulation.

## Concept

[Peer](https://github.com/PeerGrunow) and I ([Jonathan](https://github.com/jonaebel)) will each independently develop and train our own PPO agent on the `CarRacing-v3` environment. Afterwards we will evaluate and compare our results, discussing how our respective design choices affected each agent's performance.

## Project Structure

```
ppo-car-racing-agent/
├── shared/
│   ├── wrapper.py      # Environment setup and random-action baseline
│   └── eval.py         # Shared evaluation utilities
├── jonathan/           # Jonathans implementation (in progress)
│   ├── cnn.py          # CNN feature extractor
│   ├── ppo_agent.py    # PPO agent implementation
│   └── train.py        # Training loop
├── peer/               # Peer's implementation (in progress)
├── evaluation/         # Comparative evaluation results (in progress)
└── requirements.txt
```

## Setup

**Prerequisites:** Python 3.10+ and [Homebrew](https://brew.sh/) (macOS).

Install system dependencies (needed to build Box2D and pygame):

```bash
brew install swig sdl2 sdl2_image sdl2_mixer sdl2_ttf
```

Create a virtual environment and install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

The agent trains on [`CarRacing-v3`](https://gymnasium.farama.org/environments/box2d/car_racing/) from Gymnasium's Box2D suite.

- **Observation:** 96×96×3 RGB image of the track from a top-down view
- **Action space:** Continuous — `[steering, gas, brake]`
- **Reward:** +1000/N for each track tile visited (N = total tiles), −0.1 per frame as a time penalty
