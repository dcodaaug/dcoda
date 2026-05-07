from typing import List
import math
from collections import defaultdict

from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint

from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import JointCondition, ConditionSet
from rlbench.backend.task import BimanualTask


FIXED_BUTTON_COLOR_NAME = 'blue'
FIXED_BUTTON_COLOR_RGB = (0.0, 0.0, 1.0)
HIDDEN_BUTTON_POS = (-10.0, -10.0, -10.0)


class UnimanualPushSingleButtonLeft(BimanualTask):

    def init_task(self) -> None:
        self.buttons_pushed = 0
        self.target_buttons = [Shape('push_buttons_target%d' % i)
                               for i in range(3)]
        self.target_topPlates = [Shape('target_button_topPlate%d' % i)
                                 for i in range(3)]
        self.target_joints = [Joint('target_button_joint%d' % i)
                              for i in range(3)]
        self.target_wraps = [Shape('target_button_wrap%d' % i)
                             for i in range(3)]
        self.boundaries = Shape('push_buttons_boundary')
        self.goal_conditions = [JointCondition(self.target_joints[n], 0.001)
                                for n in range(3)]

        self.waypoint_mapping = defaultdict(lambda: 'left')
        self.waypoint_mapping.update({'waypoint2': 'right'})

    def _hide_unused_buttons(self) -> None:
        for i in range(1, len(self.target_buttons)):
            self.target_buttons[i].set_renderable(False)
            self.target_buttons[i].set_collidable(False)
            self.target_buttons[i].set_position(HIDDEN_BUTTON_POS)
            self.target_topPlates[i].set_renderable(False)
            self.target_topPlates[i].set_collidable(False)
            self.target_wraps[i].set_renderable(False)
            self.target_wraps[i].set_collidable(False)

    def init_episode(self, index: int) -> List[str]:
        del index

        for tp in self.target_topPlates:
            tp.set_color([1.0, 0.0, 0.0])
        for w in self.target_wraps:
            w.set_color([1.0, 0.0, 0.0])

        self.buttons_to_push = 1
        self.color_names = [FIXED_BUTTON_COLOR_NAME]
        self.color_rgbs = [FIXED_BUTTON_COLOR_RGB]
        self.target_buttons[0].set_color(FIXED_BUTTON_COLOR_RGB)
        self._hide_unused_buttons()

        self.success_conditions = [self.goal_conditions[0]]
        self.register_success_conditions(
            [ConditionSet(self.success_conditions, True, False)])

        rtn0 = 'push the %s button' % self.color_names[0]
        rtn1 = 'press the %s button' % self.color_names[0]
        rtn2 = 'push down the button with the %s base' % self.color_names[0]

        b = SpawnBoundary([self.boundaries])
        b.sample(self.target_buttons[0], min_distance=0.1)

        left_tcp = Dummy('Panda_rightArm_tip')
        Dummy('waypoint2').set_position(position=(0, 0, 0), relative_to=left_tcp)
        Dummy('waypoint2').set_orientation(
            orientation=(0, 0, 0), relative_to=left_tcp)

        w0 = Dummy('waypoint0')
        x, y, z = self.target_buttons[0].get_position()
        w0.set_position([x, y, z + 0.083])
        w0.set_orientation([math.pi, 0, math.pi])

        return [rtn0, rtn1, rtn2]

    def variation_count(self) -> int:
        return 1

    def step(self) -> None:
        if self.goal_conditions[0].condition_met() == (True, True):
            self.target_topPlates[0].set_color([0.0, 1.0, 0.0])
            self.target_wraps[0].set_color([0.0, 1.0, 0.0])

    def cleanup(self) -> None:
        self.buttons_pushed = 0

    def is_static_workspace(self):
        return True
