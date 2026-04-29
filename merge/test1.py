#自作エージェントで可能かどうか

import time
import gymnasium
from custom_merge_env import CustomMergeEnv
import highway_env
from ppoagent import PPOAgent
from setting import UniqueEnv
#from highway_env.envs.merge_env import MergeEnv


config = {
    "lanes_count": 2,
    #"vehicles_count": 100,
    "collision_reward": -1000.0,
    #"right_lane_reward": -0.5,
    #"high_speed_reward": 1.0,
    #"lane_change_reward": 1.0,
    #"acceleration_reward": 1.0, 
    #"spacing_reward": 4.0,
}

agent = PPOAgent(n_envs=1, config=config)
agent.train(total_timesteps=int(1000))

# 登録せずに直接使う場合
#env = CustomMergeEnv(render_mode="rgb_array")
#env.configure({"lanes_count": 3})  # 動的にレーン数変更
#env.reset()
#agent = PPOAgent(n_envs=1, config=config)

# 評価用環境（描画あり）
test_env = UniqueEnv(render_mode="rgb_array", config=config)

while True:
    done = truncated = False
    obs, info = test_env.reset()
    while not (done or truncated):
        action, _states = agent.predict(obs)
        #print(action)
        obs, reward, done, truncated, info = test_env.step(action)
        test_env.render()
        time.sleep(0.2)
