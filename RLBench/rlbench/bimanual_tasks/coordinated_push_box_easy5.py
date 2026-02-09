from collections import defaultdict
from typing import List, Tuple

import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task
from rlbench.backend.task import BimanualTask
from pyrep.backend import sim
from rlbench.backend.spawn_boundary import SpawnBoundary

class CoordinatedPushBoxEasy5(BimanualTask):

    def init_task(self) -> None:

        self.item = Shape('cube')
        self.target = Shape('target')
        self.register_graspable_objects([self.item])

        self.waypoint_mapping = defaultdict(lambda: 'left')
        self.waypoint_mapping.update({'waypoint0': 'right', 'waypoint2': 'right'})

        self.boundaries = Shape('workspace')

    def init_episode(self, index) -> List[str]:
        # coordinated_push_box_easy5
        self._variation_index = index

        #Dummy('waypoint0').set_position(position=(0,0,0),relative_to=self.target)
        #Dummy('waypoint0').set_position(position=(0,0,0),relative_to=self.target)

        if sim.simGetObjectSizeFactor(Shape('workspace').get_handle()) >= 1.0:
            sim.simScaleObject(Shape('workspace').get_handle(), 0.9, 0.9, 0.9)

        b = SpawnBoundary([self.boundaries])
        b.clear()
        b.sample(Dummy('coordinated_push_box_easy5'), min_rotation=(0.0, 0.0, 0.0), max_rotation=(0.0, 0.0, 0.0), min_distance=0.0)

        success_sensor = ProximitySensor('success0')
        self.register_success_conditions([DetectedCondition(self.item, success_sensor)])
        return ['push the box to the red area']

    def variation_count(self) -> int:
        return 1
