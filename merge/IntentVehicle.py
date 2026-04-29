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

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        # まず通常のIDM
        acc = super().acceleration(ego_vehicle, front_vehicle, rear_vehicle)

        # 🔥 意図による制御
        #if self.intent == "SLOWER":
            #acc = min(acc, -1.5)  # 常に減速方向へ
        acc = min(acc, -5.5)  # 常に減速方向へ

        return acc