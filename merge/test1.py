#自作エージェントで可能かどうか

import time
import gymnasium
from custom_merge_env import CustomMergeEnv
import highway_env
from myagent import MyAgent
#from highway_env.envs.merge_env import MergeEnv

# 登録せずに直接使う場合
env = CustomMergeEnv(render_mode="rgb_array")
env.configure({"lanes_count": 3})  # 動的にレーン数変更
#env.reset()
agent = MyAgent(env)

while True:
    done = truncated = False
    obs, info = env.reset() #coustom_merge_env.pyで，from highway_env.envs.merge_env import MergeEnvを読み込んでいるから，resetメソッドが利用可能になっている．
    #そして，resetメソッドは，車線を作る_make_roadと，車の配置を決める_make_vehiclesが実行されるようになっている．
    #CustomMergeEnvに_make_vehicles()がある場合は，指定した位置・台数・速度 に基づいて車両が配置される．
    #CustomMergeEnvに_make_vehicles()がない場合は，親クラスの MergeEnv._make_vehicles() によってデフォルトの車両配置がされる．( _make_roadでも同様)
    while not (done or truncated):
        action = agent.predict(obs) #ランダムに行動を選択する
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(0.2)
