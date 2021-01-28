import sys
import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, JR2_Kinova, Fetch, Tiago_Dual
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import EmptyScene, GraspScene
from gibson2.core.physics.interactive_objects import InteractiveObj, BoxShape, YCBObject
from gibson2.utils.utils import parse_config
from gibson2.core.render.profiler import Profiler

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


def main():
    base_pos = [1, 0, 0]

    config = parse_config('../configs/tiago_dual_p2p_nav.yaml')
    s = Simulator(mode='gui', timestep=1 / 240.0, image_width=512, image_height=512)
    scene = GraspScene()
    s.import_scene(scene)
    robot = Tiago_Dual(config)
    s.import_robot(robot)

    obj = YCBObject('003_cracker_box')
    s.import_object(obj)
    #obj.set_position_orientation(np.random.uniform(low=0, high=2, size=3), [0,0,0,1])
    obj.set_position_orientation([0.3, -0.1, 1.0], [0,0,0,1])

    #robot.apply_action(np.zeros(robot.action_dim))
    robot.robot_specific_reset()

    robot_id = robot.robot_ids[0]

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

    #set_joint_positions(robot_id, joints, np.zeros(len(joints)))

    x,y,z = robot.get_end_effector_position()
    #set_joint_positions(robot_id, finger_joints, [0.04,0.04])
    print(x,y,z)

    visual_marker = p.createVisualShape(p.GEOM_SPHERE, radius = 0.02)
    marker = p.createMultiBody(baseVisualShapeIndex = visual_marker)

    max_limits = get_max_limits(robot_id, all_joints)
    min_limits = get_min_limits(robot_id, all_joints)
    for jname, min_, max_ in zip(all_joint_names, min_limits, max_limits):
        print(jname, min_, max_)
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
        for jname, min_, max_, pose in zip(all_joint_names, min_limits, max_limits, jointPoses):
            print(jname, min_, max_, pose)

        set_joint_positions(robotid, valid_joints, jointPoses[joint_mask])
        set_joint_positions(robotid, gripper_joints, [0.01] * len(gripper_joints))
        #set_joint_positions(robotid, arm_joints, np.zeros(len(arm_joints)))
        #set_joint_positions(robotid, arm_right_joints, np.zeros(len(arm_right_joints)))
        ls = p.getLinkState(robotid, endEffectorId)
        newPos = ls[4]

        return jointPoses

    x,y,z = robot.get_end_effector_position()
    while True:
        with Profiler("Simulation step"):
            img = s.renderer.render_robot_cameras(modes=('rgb'))

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

