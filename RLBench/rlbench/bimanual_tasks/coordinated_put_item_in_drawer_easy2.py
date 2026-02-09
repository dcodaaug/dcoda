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

class CoordinatedPutItemInDrawerEasy2(BimanualTask):

    def init_task(self) -> None:
        self._options = ['bottom', 'middle', 'top']
        self._anchors = [Dummy('waypoint_anchor_%s' % opt)
                         for opt in self._options]
        self._joints = [Joint('drawer_joint_%s' % opt)
                        for opt in self._options]
        self._waypoint1 = Dummy('waypoint1')
        self._item = Shape('item')
        self.register_graspable_objects([self._item])
        self._index = 2

        self.boundaries = Shape('workspace')
        self.waypoint_mapping = defaultdict(lambda: 'right')
        self.waypoint_mapping.update({'waypoint0': 'left', 'waypoint1': 'left', 'waypoint2': 'left'})


    def init_episode(self, index) -> List[str]:
        # if sim.simGetObjectSizeFactor(Shape('workspace').get_handle()) >= 1.0:
        #     sim.simScaleObject(Shape('workspace').get_handle(), 1.0, 1.0, 1.0)
        
        b = SpawnBoundary([self.boundaries])
        b.clear()
        b.sample(Dummy('coordinated_put_item_in_drawer_easy2'), min_rotation=(0.0, 0.0, 0.0), max_rotation=(0.0, 0.0, 0.0), min_distance=0.0)

        self._variation_index = self._index
        option = self._options[self._index]
        anchor = self._anchors[self._index]
        self._waypoint1.set_position(anchor.get_position())
        success_sensor = ProximitySensor('success_' + option)
        self.register_success_conditions(
            [DetectedCondition(self._item, success_sensor)])
        return ['put the item in the %s drawer' % option,
                'put the block away in the %s drawer' % option,
                'open the %s drawer and place the block inside of it' % option,
                'leave the block in the %s drawer' % option]

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, 0], [0, 0, 0]

    def is_static_workspace(self):
        return True