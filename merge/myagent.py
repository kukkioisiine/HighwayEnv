#test1とセットで利用するエージェント

class MyAgent:
    def __init__(self, env):
        self.env = env
        # ここで必要な初期化処理を行う（例：学習済みパラメータの読み込みなど）

    def predict(self, observation):
        # 観測 observation をもとに，アクションを決定して返す
        # ここはアルゴリズムやルールベースによって変わる
        action = self.env.action_space.sample()  # ランダム行動の例（あとで置き換える）　#この例では，エージェントが取れる行動の集合(行動空間)からランダムに選ぶ．
        # self.env.action_spaceは，行動空間で，sampleがランダムに選ぶ
        return action
