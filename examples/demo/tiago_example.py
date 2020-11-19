from gibson2.core.physics.robot_locomotors import Tiago_Dual
from gibson2.utils.utils import parse_config
import os
import time
import numpy as np
import pybullet as p
import pybullet_data

def main():
    p.connect(p.GUI)
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./240.)

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    p.loadMJCF(floor)

    robots = []
    config = parse_config('../configs/tiago_p2p_nav.yaml')
    tiago = Tiago_Dual(config)
    robots.append(tiago)

    positions = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ]

    for robot, position in zip(robots, positions):
        robot.load()
        robot.set_position(position)
        robot.robot_specific_reset()
        robot.keep_still()

    for _ in range(2400):  # keep still for 10 seconds
        p.stepSimulation()
        time.sleep(1./240.)

    for _ in range(2400):  # move with small random actions for 10 seconds
        for robot, position in zip(robots, positions):
            action = np.random.uniform(-1, 1, robot.action_dim)
            robot.apply_action(action)
        p.stepSimulation()
        time.sleep(1./240.0)

    p.disconnect()


if __name__ == '__main__':
    main()

