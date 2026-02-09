from typing import List, Tuple
from pyrep.objects.joint import Joint
from rlbench.backend.conditions import JointCondition
from rlbench.backend.task import Task
from rlbench.backend.task import BimanualTask
from collections import defaultdict
from pyrep.backend import sim

from pyrep.objects.shape import Shape
from rlbench.backend.conditions import Condition
from rlbench.backend.spawn_boundary import SpawnBoundary
from pyrep.objects.dummy import Dummy

class LiftedCondition(Condition):

    def __init__(self, item: Shape, min_height: float):
        self.item = item
        self.min_height = min_height

    def condition_met(self):
        pos = self.item.get_position()
        return pos[2] >= self.min_height, False



class BimanualPickLaptopEasy2(BimanualTask):

    def init_task(self) -> None:
        self.register_success_conditions([LiftedCondition(Shape('lid'), 1.0)])
        self.waypoint_mapping = defaultdict(lambda: 'left')
        for i in range(1, 7, 2):
            self.waypoint_mapping.update({f'waypoint{i}': 'right'})
        self.boundaries = Shape('workspace')
        if sim.simGetObjectSizeFactor(Shape('workspace').get_handle()) >= 1.0:
            sim.simScaleObject(Shape('workspace').get_handle(), 0.8, 0.8, 0.8)

        b = SpawnBoundary([self.boundaries])
        b.clear()
        b.sample(Dummy('bimanual_pick_laptop_easy2'), min_rotation=(0.0, 0.0, 0.0), max_rotation=(0.0, 0.0, 0.0), min_distance=0.0)


    def init_episode(self, index: int) -> List[str]:
        return ['pick up the laptop']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self):
        return True