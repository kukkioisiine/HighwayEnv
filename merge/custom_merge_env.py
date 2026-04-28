#こんな感じで独自に環境を設定することができる！(これは車線数を動的に変化させている)
#test2-2.pyで利用している
#実装できました！！！

#################################デフォルトのプログラムコード#################################
 #def _make_road(self) -> None:
        #"""
        #Make a road composed of a straight highway and a merging lane.

        #:return: the road
        #"""
        #net = RoadNetwork()

        # Highway lanes
        #ends = [150, 80, 80, 150]  # Before, converging, merge, after　#before (150 m)：合流開始前の区間　converging (80 m)：合流開始〜本線への接続区間　merge (80 m)：合流区間（合流車線が本線に合流する区間）　after (150 m)：合流完了後の区間
        #c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE #c:実線，s：破線，n：線なし
        #y = [0, StraightLane.DEFAULT_WIDTH] # 本線のy座標の設定
        ###車線の線種パターンの設定###
        #line_type = [[c, s], [n, c]]
        #line_type_merge = [[c, s], [n, s]]
        ##########################

        ###本線の車線を生成するループ###
        #for i in range(2):
            #net.add_lane(       #net.add_laneにより，区間aからb，bからc，cからdの3区間に分けて車線を追加
                #"a",
                #"b",
                #StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]),   #StraightLaneは直線を表し，座標はx軸方向は区間長，y軸方向は車線の高さ
            #)
            #net.add_lane(
                #"b",
                #"c",
                #StraightLane(
                    #[sum(ends[:2]), y[i]],
                    #[sum(ends[:3]), y[i]],
                    #line_types=line_type_merge[i],
                #),
            #)
            #net.add_lane(
                #"c",
                #"d",
                #StraightLane(
                    #[sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]
                #),
            #)
        #############################

        # Merging lane
        #amplitude = 3.25
        #ljk = StraightLane(
            #[0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True
        #)
        #lkb = SineLane(
            #ljk.position(ends[0], -amplitude),
            #ljk.position(sum(ends[:2]), -amplitude),
            #amplitude,
            #2 * np.pi / (2 * ends[1]),
            #np.pi / 2,
            #line_types=[c, c],
            #forbidden=True,
        #)
        #lbc = StraightLane(
            #lkb.position(ends[1], 0),
            #lkb.position(ends[1], 0) + [ends[2], 0],
            #line_types=[n, c],
            #forbidden=True,
        #)
        #net.add_lane("j", "k", ljk)
        #net.add_lane("k", "b", lkb)
        #net.add_lane("b", "c", lbc)
        #road = Road(
            #network=net,
            #np_random=self.np_random,
            #record_history=self.config["show_trajectories"],
        #)
        #road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        #self.road = road
#################################################################################################


import numpy as np
from highway_env import utils
from highway_env.envs.merge_env import MergeEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.objects import Obstacle


class CustomMergeEnv(MergeEnv):
    def _make_road(self) -> None:
        net = RoadNetwork()
        lanes_count = self.config.get("lanes_count", 2) # lanes_countが辞書にあれば，lanes_countに格納されている値を返す．無ければ2を返す．
        lane_width = StraightLane.DEFAULT_WIDTH

        # 各区間の長さ（十分に長く）
        before_merge = 150
        merge_zone = 80
        after_merge = 80
        post_merge = 1500
        ends = [before_merge, merge_zone, after_merge, post_merge]

        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        line_type = [[c, s], [n, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]

        # 本線を上に向かって敷く
        for i in range(lanes_count):
            y = i * lane_width
            net.add_lane("a", "b", StraightLane([0, y], [sum(ends[:2]), y], line_types = line_type[0] if i == 0 else (line_type[2] if i == lanes_count - 1 else line_type[1])))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y], [sum(ends[:3]), y], line_types = line_type[0] if i == 0 else (line_type[2] if i == lanes_count - 1 else line_type[1])))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y], [sum(ends), y], line_types = line_type[0] if i == 0 else (line_type[2] if i == lanes_count - 1 else line_type[1])))

        # 合流車線（本線のさらに1.7本分上）
        merging_y = (lanes_count + 1.7) * lane_width
        amplitude = 3.25

        ljk = StraightLane([0, merging_y], [ends[0], merging_y], line_types=[c, c], forbidden=True)
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(
            lkb.position(ends[1], 0),
            lkb.position(ends[1], 0) + [ends[2], 0],
            line_types=[n, c],
            forbidden=True,
        )

        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    ###プログラムを動かす際に利用される(車の配置の初期位置を決める)
    def _make_vehicles(self) -> None:
        road = self.road
        lanes_count = self.config.get("lanes_count", 2)

        #本線の場合，どの車線に配置させるかをランダムで決める
        ego_lane_index = self.np_random.integers(lanes_count) #合流車線の場合は利用しない(1車線しかないため)
        #自車両を本線に置く場合
        #ego_vehicle = self.action_type.vehicle_class(
            #road, road.network.get_lane(("a", "b", ego_lane_index)).position(30, 0), speed=30
        #)
        #自車両を合流車線に置く場合
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("j", "k", 0)).position(30, 0), speed=30
        )
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # 他車（本線）
        for position, speed in [(90, 29), (70, 31), (5, 31.5)]:
            lane_index = self.np_random.integers(lanes_count)
            lane = road.network.get_lane(("a", "b", lane_index))
            pos = lane.position(position + self.np_random.uniform(-5, 5), 0)
            road.vehicles.append(other_vehicles_type(road, pos, speed=speed + self.np_random.uniform(-1, 1)))

        # 合流車両（1台）
        #merging_lane = road.network.get_lane(("j", "k", 0))
        #merging_vehicle = other_vehicles_type(
            #road, merging_lane.position(110, 0), speed=20
        #)
        #merging_vehicle.target_speed = 30
        #road.vehicles.append(merging_vehicle)

        self.vehicle = ego_vehicle
