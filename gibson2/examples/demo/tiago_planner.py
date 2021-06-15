from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
import time
import gibson2
import os
from gibson2.render.profiler import Profiler
import logging
import numpy as np
from gibson2.objects.ycb_object import YCBObject
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.robots.fetch_robot import Fetch

target_pos = [0.6,0.2,0.8]
target_orn = [0,0,-1]

def main():
    config_filename = os.path.join(gibson2.example_config_path, 'tiago_dual_point_nav.yaml')
    env = iGibsonEnv(config_file=config_filename, mode='gui')
    #Load Motion Planner
    motion_planner = MotionPlanningWrapper(env)
    
    
    #time.sleep(2)    
    motion_planner.set_marker_position_yaw(target_pos, 0)
    plan = motion_planner.plan_base_motion([0.2,-1,5])
    #print("Plan Base Motion------------------------", plan)

    arm_plan = motion_planner.plan_arm_push(target_pos, target_orn)
    print("Arm Planning-------------", arm_plan)
    #if arm_plan is not None and len(plan) > 0:
    execute_arm_plan = motion_planner.execute_arm_push(arm_plan, target_pos, target_orn)
    print("Execute Arm Motion------------------------", execute_arm_plan)

    #print("Print the robot id++++++++++",env.robots[0].robot_ids[0])
    robot = env.robots[0]


    while True:
        #with Profiler('Environment action step'):
        action = np.zeros(robot.action_dim)
        """
        action = np.zeros(robot.action_dim)
        x = 0
        y = robot.wheel_dim
        action[x:y] = 0.01
        x = y
        y += robot.torso_lift_dim
        action[x:y] = 0.01
        x = y
        #y += robot.head_dim
        action[x:y] = 0.03
        y += robot.gripper_dim
        action[x:y] = 0.3

        # action[y+3] = 0.3
        """
        robot.apply_action(action)
        state, reward, done, info = env.step(action)
        time.sleep(1./240.)

    #env.reset()


if __name__ == "__main__":
    main()
