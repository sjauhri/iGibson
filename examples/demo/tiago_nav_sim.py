from gibson2.core.physics.robot_locomotors import Tiago_Dual
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene
from gibson2.core.physics.interactive_objects import YCBObject
from gibson2.utils.utils import parse_config
import pybullet as p
import numpy as np
from gibson2.core.render.profiler import Profiler


def main():
    config = parse_config('../configs/tiago_dual_p2p_nav.yaml')
    s = Simulator(mode='gui', image_width=512, image_height=512)
    scene = BuildingScene('Rs',
                          build_graph=True,
                          pybullet_load_texture=True)
    s.import_scene(scene)
    robot = Tiago_Dual(config)
    s.import_robot(robot)
    robot.robot_specific_reset()

    for _ in range(10):
        obj = YCBObject('003_cracker_box')
        s.import_object(obj)
        obj.set_position_orientation(np.random.uniform(low=0, high=2, size=3), [0,0,0,1])

    action = np.zeros(robot.action_dim)
    #action[:robot.wheel_dim] = 0.1
    action[0] = 0.1
    action[1] = 0.5
    for i in range(10000):
        with Profiler('Simulator step'):
            robot.apply_action(action)
            s.step()
            rgb = s.renderer.render_robot_cameras(modes=('rgb'))

    s.disconnect()


if __name__ == '__main__':
    main()
