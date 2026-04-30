from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.road.road import LaneIndex, Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

#IDMVehicleを継承して，意図を持つ車両のクラスを定義
from highway_env.vehicle.behavior import IDMVehicle

class IntentVehicle(IDMVehicle):
    def __init__(
        self,
        road,
        position,
        heading=0,
        speed=0,
        target_lane_index=None,
        target_speed=None,
        route=None,
        enable_lane_change=True,
        timer=None,
    ):
        super().__init__(
            road,
            position,
            heading,
            speed,
            target_lane_index,
            target_speed,
            route,
            enable_lane_change,
            timer,
        )

        # 🔥 ここが追加ポイント
        self.intent = "merge"   # 例：合流意図
        self.is_intention_sender = True
        #print("動いてるよ")

        # 🔥 意図候補（論文対応）
        self.intent_list = ["IDLE", "LANE_LEFT", "FASTER", "SLOWER"]

        # 🔥 ランダム選択（これが本体）
        self.intent = np.random.choice(self.intent_list)

        # one-hot表現も持たせる（論文用）
        self.intent_vector = {
            "IDLE":       [1, 0, 0, 0, 0],
            "LANE_LEFT":  [1, 1, 0, 0, 0],
            "FASTER":     [1, 0, 0, 1, 0],
            "SLOWER":     [1, 0, 0, 0, 1],
        }[self.intent]

        self.is_intention_sender = True

        print(f"[IntentVehicle] intent = {self.intent}")

    def act(self, action: dict | str = None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        #print(action)
        if self.crashed:
            return
        action = {}
        # Lateral: MOBIL
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()
        action["steering"] = self.steering_control(self.target_lane_index)
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )

        # Longitudinal: IDM
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
            self, self.lane_index
        )
        action["acceleration"] = self.acceleration(
            ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
        )
        # When changing lane, check both current and target lanes
        if self.lane_index != self.target_lane_index:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
                self, self.target_lane_index
            )
            target_idm_acceleration = self.acceleration(
                ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
            )
            action["acceleration"] = min(
                action["acceleration"], target_idm_acceleration
            )
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])
        action["acceleration"] = np.clip(
            action["acceleration"], -self.ACC_MAX, self.ACC_MAX
        )
        # Skip ControlledVehicle.act(), or the command will be overridden.
        #print(action)
        Vehicle.act(self, action)

    def step(self, dt: float):
        """
        Step the simulation.

        Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super().step(dt)

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        # まず通常のIDM
        acc = super().acceleration(ego_vehicle, front_vehicle, rear_vehicle)

        # 🔥 意図による制御
        #if self.intent == "SLOWER":
            #acc = min(acc, -1.5)  # 常に減速方向へ
        acc = min(acc, -5.5)  # 常に減速方向へ

        return acc