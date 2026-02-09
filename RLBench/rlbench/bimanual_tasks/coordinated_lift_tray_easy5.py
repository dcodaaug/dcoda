from collections import defaultdict
from typing import List, Tuple

import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task
from rlbench.backend.task import BimanualTask
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.conditions import Condition
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.spawn_boundary import SpawnBoundary
from pyrep.backend import sim

class LiftedCondition(Condition):

    def __init__(self, item: Shape, min_height: float):
        self.item = item
        self.min_height = min_height

    def condition_met(self):
        pos = self.item.get_position()
        return pos[2] >= self.min_height, False


class CoordinatedLiftTrayEasy5(BimanualTask):

    def init_task(self) -> None:
        self.item = Shape('item')
        self.tray = Shape('tray')

        self.register_graspable_objects([self.item])

        self.waypoint_mapping = defaultdict(lambda: 'left')
        for i in range(0, 7, 2):
            self.waypoint_mapping.update({f'waypoint{i}': 'right'})

        self.boundaries = Shape('workspace')

    def init_episode(self, index) -> List[str]:
        self._variation_index = index

        right_sensor = ProximitySensor('Panda_rightArm_gripper_attachProxSensor')
        left_sensor = ProximitySensor('Panda_leftArm_gripper_attachProxSensor')

        if sim.simGetObjectSizeFactor(Shape('workspace').get_handle()) >= 1.0:
            sim.simScaleObject(Shape('workspace').get_handle(), 0.9, 0.9, 0.9)

        b = SpawnBoundary([self.boundaries])
        b.clear()
        b.sample(Dummy('coordinated_lift_tray_easy5'), min_rotation=(0.0, 0.0, 0.0), max_rotation=(0.0, 0.0, 0.0), min_distance=0.0)

        #tray_visual = Shape('tray_visual')
        #print(self.item.get_position())
        #tray_visual.sample(self.item, min_distance=0.1, ignore_collisions=True)
        self.item.set_position([0.0, 0.0, 0.001], relative_to=self.tray, reset_dynamics=False)
        #print(self.item.get_position())

        self.register_success_conditions([
            LiftedCondition(self.tray, 1.2),
            LiftedCondition(self.item, 1.2),
            DetectedCondition(self.tray, right_sensor),
            DetectedCondition(self.tray, left_sensor)])

        return ['Lift the tray']

    def variation_count(self) -> int:
        return 1 #len(self._options)

    def boundary_root(self) -> Object:
        return Dummy('coordinated_lift_tray_easy5')

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, 0], [0, 0, 0]

    def is_static_workspace(self):
        return True