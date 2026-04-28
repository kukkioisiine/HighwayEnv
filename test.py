import gymnasium
import highway_env
from matplotlib import pyplot as plt

env = gymnasium.make('highway-v0', render_mode='rgb_array')
env.reset()

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

img = env.render()
plt.imshow(img)
plt.show()