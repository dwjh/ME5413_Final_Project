#!/usr/bin/env python3

import rospy
import smach
import smach_ros
import yaml
import os

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib import SimpleActionClient

# 初始化状态
class InitializeState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['initialized'])

    def execute(self, userdata):
        rospy.loginfo('State: INITIALIZE')
        rospy.sleep(2.0)  # 等待定位稳定 (可调整时间)
        return 'initialized'

# 从YAML读取导航目标并导航
class NavigateToGoalFromYAML(smach.State):
    def __init__(self, goal_name, yaml_path):
        smach.State.__init__(self, outcomes=['reached', 'failed'])
        self.goal_name = goal_name
        self.yaml_path = os.path.expanduser(yaml_path)
        self.client = SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base server...")
        self.client.wait_for_server()

    def execute(self, userdata):
        rospy.loginfo(f'State: NAVIGATE_TO_GOAL → {self.goal_name}')
        try:
            with open(self.yaml_path, 'r') as f:
                all_goals = yaml.safe_load(f)
            goal_data = all_goals['navigation_goals'][self.goal_name]
        except Exception as e:
            rospy.logerr(f"Failed to load goal from YAML: {e}")
            return 'failed'

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        pos = goal_data['position']
        ori = goal_data['orientation']

        goal.target_pose.pose.position.x = pos['x']
        goal.target_pose.pose.position.y = pos['y']
        goal.target_pose.pose.position.z = pos['z']
        goal.target_pose.pose.orientation.x = ori['x']
        goal.target_pose.pose.orientation.y = ori['y']
        goal.target_pose.pose.orientation.z = ori['z']
        goal.target_pose.pose.orientation.w = ori['w']

        self.client.send_goal(goal)

        if self.client.wait_for_result(rospy.Duration(60)):
            rospy.loginfo("Reached YAML goal successfully.")
            return 'reached'
        else:
            rospy.logwarn("Failed to reach YAML goal.")
            return 'failed'

def main():
    rospy.init_node('simple_navigation_sm')

    sm = smach.StateMachine(
        outcomes=['navigation_completed', 'navigation_failed'])

    # 设置 YAML 路径（可根据你实际路径调整）
    yaml_path = '~/ME5413_Final_Project/src/auto_task_fsm/config/navigation_goals.yaml'

    with sm:
        smach.StateMachine.add('INITIALIZE', InitializeState(),
                               transitions={'initialized': 'NAVIGATE_YAML_GOAL'})

        # 可修改目标名为 test / bridge / target_box
        smach.StateMachine.add('NAVIGATE_YAML_GOAL',
                               NavigateToGoalFromYAML('test2', yaml_path),
                               transitions={'reached': 'navigation_completed',
                                          'failed': 'navigation_failed'})

    # 状态机可视化工具
    sis = smach_ros.IntrospectionServer('navigation_sm', sm, '/SM_ROOT')
    sis.start()

    outcome = sm.execute()

    rospy.spin()
    sis.stop()

if __name__ == '__main__':
    main()
