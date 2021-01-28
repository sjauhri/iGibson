import sys
import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, JR2_Kinova, Fetch, Tiago_Dual
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import EmptyScene, GraspScene
from gibson2.core.physics.interactive_objects import InteractiveObj, BoxShape, YCBObject
from gibson2.utils.utils import parse_config
from gibson2.core.render.profiler import Profiler
from gibson2.utils.gl_transform import GL_Transform

import pytest
import pybullet as p
import numpy as np
from gibson2.external.pybullet_tools.utils import set_base_values, joint_from_name, set_joint_position, \
    set_joint_positions, add_data_path, connect, plan_base_motion, plan_joint_motion, enable_gravity, \
    joint_controller, dump_body, load_model, joints_from_names, user_input, disconnect, get_joint_positions, \
    get_link_pose, link_from_name, HideOutput, get_pose, wait_for_user, dump_world, plan_nonholonomic_motion, \
    set_point, create_box, stable_z, control_joints, get_max_limits, get_min_limits, get_sample_fn, get_movable_joints, get_joint_names

import time
import numpy as np

import cv2

import torch
from torchvision import transforms
from gibson2.utils.unet import UNet 


# TODO: cleanup

# Load the Network.
net = UNet(1,3)
MODEL_FILE = '/home/hypatia/projects/m2ore/models_trained/unet_best2_grasp_point_jacquard.pt'
chkpt = torch.load(MODEL_FILE, map_location=torch.device('cuda'))
net.load_state_dict(chkpt['model_state_dict'], strict=False)
net = net.cuda()

H, W = 512, 512
base_pos = [0.9, 0, 0]

gltrans = GL_Transform()

def coords_from_pointcloud(pc, x, y):
    X = pc[y, x, :4]  # homogenous
    return X

def depth_from_pointcloud(pc, min_depth=0.5, max_depth=5.0):
    depth = - pc[:, :, 2]  # extract z from point cloud
    # 0.0 is a special value for invalid entries
    #depth[depth < min_depth] = 0.0
    #depth[depth > max_depth] = 0.0

    ## re-scale depth to [0.0, 1.0]
    #depth /= max_depth

    # simulate real camera by adding noise
    valid_mask = np.random.choice(2, depth.shape, p=[0, 1])
    depth[valid_mask == 0] = 0.0

    return depth


def m2ore_grasp(depth, crop_size=350, scale_size=320):
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

    #print("grasp_pixel", max_pix[0], max_pix[1])
    #cv2.circle(depth_norm, center=(max_pix_raw[1], max_pix_raw[2]), radius=3, color=[255,102,0], thickness=2)
    #cv2.imshow('depth_image', depth_norm)

    max_pix = ((np.array(max_pix_raw[1:2+1]) / scale_size * crop_size) + np.array([crop_margin_y, crop_margin_x]))
    max_pix = np.round(max_pix).astype(np.int)
    #cv2.circle(depth, center=(max_pix[0], max_pix[1]), radius=3, color=[255,102,0], thickness=2)
    #cv2.imshow('depth_image', depth)

    #point_depth = float(depth[max_pix[0], max_pix_raw[1]])
    #print("depth:", point_depth)

    x = max_pix[0]
    y = max_pix[1]

    return x, y


def main():
    config = parse_config('../configs/tiago_dual_p2p_nav.yaml')
    s = Simulator(mode='gui', timestep=1 / 240.0, image_width=W, image_height=H)
    scene = GraspScene()
    s.import_scene(scene)
    robot = Tiago_Dual(config)
    s.import_robot(robot)

    obj = YCBObject('003_cracker_box')
    s.import_object(obj)
    # obj.set_position_orientation(np.random.uniform(low=0, high=0.5, size=3), [0,0,0,1])
    obj.set_position_orientation([0.3, -0.1, 0.8], [0,0,0,1])

    #robot.apply_action(np.zeros(robot.action_dim))
    robot.robot_specific_reset()

    robot_id = robot.robot_ids[0]
    
    torso_joints = joints_from_names(robot_id, ['head_1_joint', 'head_2_joint', 'torso_lift_joint'])

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
    arm_right_joints = joints_from_names(robot_id, ['arm_right_1_joint',
                                               'arm_right_2_joint',
                                               'arm_right_3_joint',
                                               'arm_right_4_joint',
                                               'arm_right_5_joint',
                                               'arm_right_6_joint',
                                               'arm_right_7_joint'])
    start_index = robot.wheel_dim + robot.torso_lift_dim + robot.head_dim
    end_index = start_index + robot.arm_left_dim
    end_index_gripper = end_index + robot.gripper_dim
    end_index_right = end_index_gripper + robot.arm_right_dim
    robot.robot_body.reset_position(base_pos)
    robot.robot_body.reset_orientation([0, 0, 1, 0])

    all_joints = get_movable_joints(robot_id)
    all_joint_names = get_joint_names(robot_id, all_joints)
    valid_joints = [j.joint_index for j in robot.ordered_joints]
    joint_mask = []
    for j in all_joints:
        if j in valid_joints:
            joint_mask += [True]
        else:
            joint_mask += [False]

    set_joint_positions(robot_id, torso_joints, [-0.2, -0.8, 0.5])


    #set_joint_positions(robot_id, joints, np.zeros(len(joints)))

    x,y,z = robot.get_end_effector_position()
    #set_joint_positions(robot_id, finger_joints, [0.04,0.04])
    #print(x,y,z)

    visual_marker = p.createVisualShape(p.GEOM_SPHERE, radius = 0.02)
    marker = p.createMultiBody(baseVisualShapeIndex = visual_marker)

    visual_marker_grasp = p.createVisualShape(p.GEOM_SPHERE, radius = 0.02)
    marker_grasp = p.createMultiBody(baseVisualShapeIndex = visual_marker_grasp)

    max_limits = get_max_limits(robot_id, all_joints)
    min_limits = get_min_limits(robot_id, all_joints)
    #for jname, min_, max_ in zip(all_joint_names, min_limits, max_limits):
    #    print(jname, min_, max_)
    rest_position = get_joint_positions(robot_id, all_joints)
    #rest_position = robot.rest_position #[0,0] + list(get_joint_positions(robot_id, arm_joints))
    joint_range = list(np.array(max_limits) - np.array(min_limits))
    #joint_range = [item + 1 for item in joint_range]  # TODO: what is this for?
    jd = [0.1 for item in joint_range]

    def accurateCalculateInverseKinematics(robotid, endEffectorId, targetPos, threshold, maxIter):
        #sample_fn = get_sample_fn(robotid, arm_joints)
        #set_joint_positions(robotid, arm_joints, sample_fn())

        jointPoses = p.calculateInverseKinematics(robotid, endEffectorId, targetPos,
                                                    maxNumIterations=maxIter,
                                                    residualThreshold=threshold,
                                                      lowerLimits = min_limits,
                                                      upperLimits = max_limits,
                                                      jointRanges = joint_range,
                                                      restPoses = rest_position,
                                                      jointDamping = jd)
        jointPoses = np.asarray(jointPoses)
        #for jname, min_, max_, pose in zip(all_joint_names, min_limits, max_limits, jointPoses):
        #    print(jname, min_, max_, pose)

        set_joint_positions(robotid, valid_joints, jointPoses[joint_mask])
        set_joint_positions(robotid, gripper_joints, [0.01] * len(gripper_joints))
        #set_joint_positions(robotid, arm_joints, np.zeros(len(arm_joints)))
        #set_joint_positions(robotid, arm_right_joints, np.zeros(len(arm_right_joints)))
        ls = p.getLinkState(robotid, endEffectorId)
        newPos = ls[4]

        return jointPoses

    est_freq = 60
    est_i = 0

    x,y,z = robot.get_end_effector_position()
    while True:
        #K = s.renderer.get_intrinsics()  # camera matrix
        gltrans.V = s.renderer.V  # view matrix
        gltrans.P = s.renderer.P  # projection matrix

        with Profiler("Simulation step"):
            if est_i != est_freq:
                est_i += 1
            else:
                est_i = 0

                pc = np.asarray(s.renderer.render_robot_cameras(modes=('3d')))[0]
                depth = depth_from_pointcloud(pc)
                #cv2.imshow("depth_image", depth)
                #cv2.imwrite('depth_img.png', depth_img)

                xx, yy = m2ore_grasp(depth)

                X = coords_from_pointcloud(pc, xx, yy)

                #depth[yy-2:yy+2, xx-2:xx+2] = 0
                cv2.circle(depth, center=(xx, yy), radius=3, color=[255,102,0], thickness=2)
                cv2.imshow('depth_image', depth)

                X = gltrans.cam_to_world(X)

                print("m2ore found grasp point! X: {:.4f}, Y: {:.4f}, Z: {:.4f}".format(X[0], X[1], X[2]))

                p.resetBasePositionAndOrientation(marker_grasp, X, [0,0,0,1])

            #time.sleep(1/240)
            robot.robot_body.reset_position(base_pos)
            robot.robot_body.reset_orientation([0, 0, 1, 0])
            threshold = 0.01
            maxIter = 100
            joint_pos = accurateCalculateInverseKinematics(robot_id, robot.get_end_effector_index(), [x, y, z],
                                                           threshold, maxIter)[2:10]

            #set_joint_positions(robot_id, finger_joints, [0.04, 0.04])
            s.step()
            keys = p.getKeyboardEvents()
            for k, v in keys.items():
                if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_IS_DOWN)):
                    y -= 0.01
                if (k == p.B3G_LEFT_ARROW and (v & p.KEY_IS_DOWN)):
                    y += 0.01
                if (k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN)):
                    x += 0.01
                if (k == p.B3G_DOWN_ARROW and (v & p.KEY_IS_DOWN)):
                    x -= 0.01
                if (k == ord('z') and (v & p.KEY_IS_DOWN)):
                    z += 0.01
                if (k == ord('x') and (v & p.KEY_IS_DOWN)):
                    z -= 0.01
            p.resetBasePositionAndOrientation(marker, [x,y,z], [0,0,0,1])

    s.disconnect()


if __name__ == '__main__':
    main()

