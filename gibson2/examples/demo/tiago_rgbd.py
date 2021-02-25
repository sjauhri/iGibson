from gibson2.robots.tiago_dual_robot import Tiago_Dual
from gibson2.simulator import Simulator
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.utils.utils import parse_config
from gibson2.render.profiler import Profiler

import pybullet as p
from gibson2.external.pybullet_tools.utils import set_joint_positions, joints_from_names, get_joint_positions, \
    get_max_limits, get_min_limits, get_sample_fn, get_movable_joints

import numpy as np
import gibson2
import os

import cv2

def main():
    config = parse_config(os.path.join(gibson2.example_config_path, 'tiago_dual_point_nav.yaml'))
    s = Simulator(mode='gui', physics_timestep=1 / 240.0)
    scene = EmptyScene()
    s.import_scene(scene)
    tiago = Tiago_Dual(config)
    s.import_robot(tiago)

    robot_id = tiago.robot_ids[0]

    tiago.robot_body.reset_position([0, 0, 0])
    tiago.robot_body.reset_orientation([0, 0, 1, 0])

    # get camera intrinsics (camera matrix)
    K = s.renderer.get_intrinsics()
    print("== camera intrinsics ==")
    print(K)

    # render image, note: RGB image and SEG image contain alpha channel
    rgb_im = np.asarray(s.renderer.render_robot_cameras(modes=('rgb')))[0]
    seg_im = np.asarray(s.renderer.render_robot_cameras(modes=('seg')))[0]

    # reconstruct depth from point cloud
    pc = np.asarray(s.renderer.render_robot_cameras(modes=('3d')))[0]
    depth_im = - pc[:, :, 2]  # extract z buffer from point cloud
    # add noise to simulate real camera
    valid_mask = np.random.choice(2, depth_im.shape, p=[0, 1])
    depth_im[valid_mask == 0] = 0.0

    print("== img shapes ==")
    print("RGB: {}, D: {}, SEG: {}".format(rgb_im.shape, depth_im.shape, seg_im.shape))

    #while True:
    #    with Profiler("Simulation step"):
    #        tiago.robot_body.reset_position([0, 0, 0])
    #        tiago.robot_body.reset_orientation([0, 0, 1, 0])

    #        s.step()

    s.disconnect()


if __name__ == '__main__':
    main()
