#from __future__ import annotations

#import numpy as np
#import pandas as pd

#from highway_env.envs.common.abstract import AbstractEnv
#from highway_env.envs.common.observation import ObservationType
#from highway_env.envs.common.observation import KinematicObservation

#class IntentKinematicObservation(KinematicObservation):

    #def __init__(self, env, **kwargs):
        #super().__init__(env, **kwargs)

        # 🔥 intent分を追加
        #self.features = ["presence", "x", "y", "vx", "vy", "intent"]
        #print(self.features)
        #self.features = ["presence", "x", "y", "vx", "vy"]

    #def observe(self) -> np.ndarray:
        #if not self.env.road:
            #return np.zeros(self.space().shape)

        # Add ego-vehicle
        #df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])
        # Add nearby traffic
        #close_vehicles = self.env.road.close_objects_to(
            #self.observer_vehicle,
            #self.env.PERCEPTION_DISTANCE,
            #count=self.vehicles_count - 1,
            #see_behind=self.see_behind,
            #sort=self.order == "sorted",
            #vehicles_only=not self.include_obstacles,
        #)
        #if close_vehicles:
            #origin = self.observer_vehicle if not self.absolute else None
            #vehicles_df = pd.DataFrame.from_records(
                #[
                    #v.to_dict(origin, observe_intentions=self.observe_intentions)
                    #for v in close_vehicles[-self.vehicles_count + 1 :]
                #]
            #)
            #df = pd.concat([df, vehicles_df], ignore_index=True)

        #df = df[self.features]

        # Normalize and clip
        #if self.normalize:
            #df = self.normalize_obs(df)
        # Fill missing rows
        #if df.shape[0] < self.vehicles_count:
            #rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            #df = pd.concat(
                #[df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True
            #)
        # Reorder
        #df = df[self.features]
        #obs = df.values.copy()
        #if self.order == "shuffled":
            #self.env.np_random.shuffle(obs[1:])
        # Flatten
        #return obs.astype(self.space().dtype)
    
#def observation_factory(env: AbstractEnv, config: dict) -> ObservationType:
    #print("動いてるよん")
    #if config["type"] == "TimeToCollision":
    #    return TimeToCollisionObservation(env, **config)
    #if config["type"] == "Kinematics":
        #return KinematicObservation(env, **config)
    #elif config["type"] == "OccupancyGrid":
    #    return OccupancyGridObservation(env, **config)
    #if config["type"] == "KinematicsGoal":
        #return KinematicsGoalObservation(env, **config)
    #elif config["type"] == "GrayscaleObservation":
    #    return GrayscaleObservation(env, **config)
    #elif config["type"] == "AttributesObservation":
    #    return AttributesObservation(env, **config)
    #elif config["type"] == "MultiAgentObservation":
    #    return MultiAgentObservation(env, **config)
    #elif config["type"] == "TupleObservation":
    #    return TupleObservation(env, **config)
    #elif config["type"] == "LidarObservation":
    #    return LidarObservation(env, **config)
    #elif config["type"] == "ExitObservation":
    #    return ExitObservation(env, **config)
    #elif config["type"] == "IntentKinematics":
        #return IntentKinematicObservation(env, **config)
    #else:
        #raise ValueError("Unknown observation type")
    

#チャッピー丸投げ

from __future__ import annotations

import numpy as np
import pandas as pd

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import ObservationType
from highway_env.envs.common.observation import KinematicObservation


class IntentKinematicObservation(KinematicObservation):

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

        # 🔥 intentをfeaturesに追加（なければ）
        if "intent" not in self.features:
            self.features = list(self.features) + ["intent"]

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # =========================
        # ① ego車
        # =========================
        ego_dict = self.observer_vehicle.to_dict()

        # 👉 egoにもintentを保証
        if "intent" in self.features:
            ego_dict["intent"] = getattr(self.observer_vehicle, "intent", 0)

        df = pd.DataFrame.from_records([ego_dict])

        # =========================
        # ② 周囲車両
        # =========================
        close_vehicles = self.env.road.close_objects_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,
            see_behind=self.see_behind,
            sort=self.order == "sorted",
            vehicles_only=not self.include_obstacles,
        )

        #if close_vehicles:
            #origin = self.observer_vehicle if not self.absolute else None

            #records = []
            #for v in close_vehicles[-self.vehicles_count + 1:]:
                #d = v.to_dict(origin, observe_intentions=self.observe_intentions)

                # =========================
                # ★ intent_vectorを使用
                # =========================
                #print(v)
                #if hasattr(v, "intent_vector"):
                    #print("動いてるよ1")
                    #intent_vec = v.intent_vector
                    #intent_vec = 1
                    #print(intent_vec)
                #else:
                    #print("動いてるよ")
                    #intent_vec = [0, 0, 0, 0, 0]
                    #intent_vec = 0

                # 👉 各次元を展開
                #for i in range(len(intent_vec)):
                    #d[f"intent_{i}"] = intent_vec[i]

                #records.append(d)

            #vehicles_df = pd.DataFrame.from_records(records)
            #df = pd.concat([df, vehicles_df], ignore_index=True)

        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None

            records = []
            for v in close_vehicles[-self.vehicles_count + 1:]:
                #print(v)
                d = v.to_dict(origin, observe_intentions=self.observe_intentions)
                #print(d)

                # 👉 intentをここで統一
                # 1のやり方
                if "intent" in self.features:
                    #print("動いているよ")
                    #print(self.observe_intentions)
                    if hasattr(v, "intent"):
                        intent_map = {
                            "IDLE": 1,
                            "LANE_LEFT": 2,
                            "FASTER": 3,
                            "SLOWER": 4
                        }
                        d["intent"] = intent_map.get(v.intent, 0)
                        print(d["intent"])

                    # 2のやり方
                    #if hasattr(v, "intent"): #and self.observe_intentions:
                        #d["intent"] = v.intent
                        #print(d["intent"])

                    #これは共通(1のやり方と2のやり方両方に対応)
                    else:
                        d["intent"] = 0
                        print(d["intent"])

                records.append(d)

            vehicles_df = pd.DataFrame.from_records(records)
            df = pd.concat([df, vehicles_df], ignore_index=True)

        # =========================
        # ③ intent列を保証（超重要）
        # =========================
        if "intent" in self.features:
            df["intent"] = df.get("intent", 0)

        # =========================
        # ④ 必要な特徴量だけ
        # =========================
        df = df[self.features]

        # =========================
        # ⑤ 正規化
        # =========================
        if self.normalize:
            df = self.normalize_obs(df)

        # =========================
        # ⑥ 足りない分を0埋め
        # =========================
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = pd.concat(
                [df, pd.DataFrame(data=rows, columns=self.features)],
                ignore_index=True
            )

        # =========================
        # ⑦ 並び順
        # =========================
        df = df[self.features]
        obs = df.values.copy()

        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])  # ego以外

        return obs.astype(self.space().dtype)


def observation_factory(env: AbstractEnv, config: dict) -> ObservationType:
    if config["type"] == "Kinematics":
        return KinematicObservation(env, **config)

    elif config["type"] == "IntentKinematics":
        return IntentKinematicObservation(env, **config)

    else:
        raise ValueError("Unknown observation type")