from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition

from rlbench.const import colors
import numpy as np
import random


class SlideBlockToSymmetricalColorTarget(Task):

    def init_task(self) -> None:
        self._waypoint_paths = {
            0: [Dummy('point0a'),
                Dummy('point0b')],

            1: [Dummy('point1a'),
                Dummy('point1b')],

            2: [Dummy('point2a'),
                Dummy('point2b')],

            3: [Dummy('point3a'),
                Dummy('point3b')]
        }


    def init_episode(self, index: int) -> List[str]:
        # target_index = np.random.randint(4)
        target_index = 3 # use the last target for now
        self.register_success_conditions([
            DetectedCondition(Shape('block'), 
                ProximitySensor(f'success{target_index}'))])
        self.target = Shape('target%d'%target_index)
        distractor_index = list(range(target_index)) + list(range(target_index + 1, 4))
        random.shuffle(distractor_index)
        self.distractor1, self.distractor2, self.distractor3 = Shape('target%d'%distractor_index[0]), Shape('target%d'%distractor_index[1]), Shape('target%d'%distractor_index[2])

        self._variation_index = index
        color_name, color_rgb = colors[index]
        self.target.set_color(color_rgb)
        color_choices = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=3, replace=False)
        for ob, i in zip([self.distractor1, self.distractor2, self.distractor3], color_choices):
            name, rgb = colors[i]
            ob.set_color(rgb)

        target_waypoints = self._waypoint_paths[target_index]
        self._waypoints = [Dummy('waypoint%d'%(i))
                           for i in range(2)]
        for i in range(len(target_waypoints)):
            self._waypoints[i].set_pose(target_waypoints[i].get_pose())
        self.register_stop_at_waypoint(i+1)
           
        return ['the %s square' % (color_name)]


    def variation_count(self) -> int:
        return len(colors)
