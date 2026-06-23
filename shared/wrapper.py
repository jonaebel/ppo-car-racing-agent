import gymnasium as gym


# Create the Car racing environment
env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

observation, info = env.reset()
# oberservation -> what can the agent see
# info for logging and evaluation
print(f"Starting Observation: {observation}")


episode_over = False
total_reward = 0

while not episode_over:
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()