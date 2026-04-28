######乱数シード対応######


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import highway_env
import gymnasium
from setting import UniqueEnv

class PPOAgent:
    def __init__(self, n_envs=1, tensorboard_log="test_ppo/", config=None, seed=42):
    #def __init__(self, n_envs=3, tensorboard_log="highway_ppo/", config=None, seed=[1,2,3,4,5]):

        # make_vec_envでUniqueEnvを使うようにする
        self.env = make_vec_env(lambda: UniqueEnv(config=config, render_mode=None), n_envs=n_envs, seed=seed)

        # SB3全体の乱数シードを固定
        set_random_seed(seed)

        #ステップ数を小さくすると学習データが少なくて，衝突しがちかな？

        # PPOモデル初期化
        self.model = PPO(
            "MlpPolicy",
            self.env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4, #文献購読の値
            #learning_rate=1e-4,
            #batch_size=128,
            batch_size=100,
            gamma=0.99,
            #n_steps=128,
            #n_steps=1024, #キリのいい数字で分けないといい結果が生まれなさそう
            n_steps=1000, #約10エピソードごとに更新
            n_epochs=10,
            #n_epochs=50,
            #gae_lambda=0.95,
            #ent_coef=0.01, #今までやっていた値
            #ent_coef=0.001,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed = seed,
        )


    def train(self, total_timesteps=100*10000, callback=None):
        #self.model.learn(total_timesteps=total_timesteps)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        self.model.save("mergetest/final_model")
        #self.model.save("ppo_highway/20万回学習エポックなど全て100")
        #self.model.save("ppo_highway/車間とスピードと衝突と快適さと左車線考慮/model")
        #self.model.save("ppo_highway/20万回学習安全性1-2(重み-1.0)")
        #self.model.save("ppo_highway/20万回学習安全性1-2と1-3")
        #self.model.save("ppo_highway/20万回学習車間距離の重み1.0")
        #self.model.save("ppo_highway/20万回学習安全性1-3")

    def load(self, path="ppo_highway/final_model"):
        self.model = PPO.load(path)

    def predict(self, observation, deterministic=True):
        return self.model.predict(observation, deterministic=deterministic)

    def get_env(self):
        return self.env
