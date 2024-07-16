from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition

SHAPE_NAMES = ['cube', 'cylinder', 'triangular prism', 'star', 'moon']
Lang_NAMES = ['cube', 'cylinder', 'triangular prism', 'star', 'crescent moon']


class ReachShape(Task):

    def init_task(self) -> None:
        self.shapes = [Shape(ob.replace(' ', '_')) for ob in SHAPE_NAMES]
        self.boundary = SpawnBoundary([Shape('boundary')])
        
    def init_episode(self, index) -> List[str]:
        #[shape.set_color((1,0,0)) for shape in self.shapes]
        self.variation_index = index
        shape = SHAPE_NAMES[index]
        self.success_sensor = ProximitySensor('%s_success'% shape.replace(' ', '_'))
        
        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), self.success_sensor)])

        self.boundary.clear()
        [self.boundary.sample(s, min_distance=0.1) for s in self.shapes]

        self._waypoints = [Dummy('waypoint0')]
        self.point = Dummy('%s_point'% shape.replace(' ', '_'))
        self._waypoints[0].set_pose(self.point.get_pose())

        shape_lang = Lang_NAMES[index]
        return ['the %s ' % shape_lang]

    def variation_count(self) -> int:
        return len(SHAPE_NAMES)

    def is_static_workspace(self) -> bool:
        return True
