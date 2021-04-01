import os
import numpy as np
import gibson2

from gibson2.robots.tiago_dual_robot import Tiago_Dual
from gibson2.simulator import Simulator
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.utils.utils import parse_config
from gibson2.render.profiler import Profiler

import pybullet as p
from gibson2.external.pybullet_tools.utils import set_joint_positions, joints_from_names, get_joint_positions, \
    get_max_limits, get_min_limits, get_sample_fn, get_movable_joints

import cv2
import torch
from torchvision import transforms
from gibson2.utils.nn.unet import UNet


def load_model(model_path="/home/student/unet_best2_grasp_point_jacquard.pt"):
    net = UNet(1,3)
    chkpt = torch.load(model_path, map_location=torch.device('cuda'))
    net.load_state_dict(chkpt['model_state_dict'], strict=False)
    net = net.cuda()
    return net


def find_grasp(net, depth, crop_size=350, scale_size=320):
    H, W = depth.shape

    crop_margin_y, crop_margin_x = (H-crop_size)//2, (W-crop_size)//2
    depth_crop = cv2.resize(depth[crop_margin_y:crop_size+crop_margin_y, crop_margin_x:crop_size+crop_margin_x], (scale_size, scale_size))

    # normalize depth
    depth_norm = cv2.normalize(depth_crop, None, 0, 1, cv2.NORM_MINMAX)

    # convert to tensor and clamp
    depth_img = transforms.functional.to_tensor(depth_norm).float()
    depth_img = torch.clamp(depth_img - depth_img.mean(), -1, 1)

    # run inference
    with torch.no_grad():
       pos, cos, sin, width, graspness, bins = net(depth_img.unsqueeze(0).cuda())
       pos, cos, sin, width, graspness, bins = pos.cpu(), cos.cpu(), sin.cpu(), width.cpu(), graspness.cpu(), bins.cpu()
    pos_g = pos * torch.sigmoid(graspness.unsqueeze(1))
    pos_f = pos_g * torch.sigmoid(bins)

    # Calculate the angle map.
    cos_out = cos.squeeze().numpy()
    sin_out = sin.squeeze().numpy()
    ang_out = np.arctan2(sin_out, cos_out)/2.0

    width_out = width.squeeze().numpy() *150  # Scaled 0-150:0-1

    temp = pos_f.squeeze()
    temp = temp.numpy()
    max_pix_raw = np.unravel_index(np.argmax(temp, axis=None), temp.shape)

    ang = ang_out[max_pix_raw[0], max_pix_raw[1], max_pix_raw[2]]
    width = width_out[max_pix_raw[0], max_pix_raw[1], max_pix_raw[2]]

    max_pix = ((np.array(max_pix_raw[1:2+1]) / scale_size * crop_size) + np.array([crop_margin_y, crop_margin_x]))
    max_pix = np.round(max_pix).astype(np.int)

    x = max_pix[1]
    y = max_pix[0]

    return x, y


def main():
    config = parse_config(os.path.join(gibson2.example_config_path, 'tiago_dual_point_nav.yaml'))
    s = Simulator(mode='gui',
                    physics_timestep=1/240.0,
                    image_width=512,
                    image_height=512,
                    render_to_tensor=False)

    net = load_model()

    scene = EmptyScene()
    s.import_scene(scene)
    tiago = Tiago_Dual(config)
    s.import_robot(tiago)

    robot_id = tiago.robot_ids[0]

    arm_joints = joints_from_names(robot_id, ['arm_left_1_joint',
                                               'arm_left_2_joint',
                                               'arm_left_3_joint',
                                               'arm_left_4_joint',
                                               'arm_left_5_joint',
                                               'arm_left_6_joint',
                                               'arm_left_7_joint'])
    gripper_joints = joints_from_names(robot_id, [
                                               'gripper_left_right_finger_joint',
                                               'gripper_left_left_finger_joint'])

    tiago.robot_body.reset_position([0, 0, 0])
    tiago.robot_body.reset_orientation([0, 0, 1, 0])

    while True:
        with Profiler("Simulation step"):
            # find grasp point from depth image
            pc = np.asarray(s.renderer.render_robot_cameras(modes=('3d')))[0]
            depth = - pc[:, :, 2]  # extract z from point cloud
            xx, yy = find_grasp(net, depth)
            X = pc[yy, xx, :4]  # point in 3d
            print("Grasp at {} {} {}".format(X[0], X[1], X[2]))

            tiago.robot_body.reset_position([0, 0, 0])
            tiago.robot_body.reset_orientation([0, 0, 1, 0])
            s.step()

    s.disconnect()


if __name__ == '__main__':
    main()
