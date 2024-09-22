import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from motion_planning_v3 import Motion_planning
from scipy.interpolate import CubicSpline
import numpy as np


class PathPlanningNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.pub = self.create_publisher(Pose, "/lbr/command/pose", 10)
        self.sub_1 = self.create_subscription(
            Pose, "/lbr/state/pose", self.callback_pose, 10
        )
        self.sub_2 = self.create_subscription(Pose, "moveto", self.callback_grasp, 10)
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.first_message_received = False  # 标志，用于指示是否已接收到第一个消息
        self.goal_pos = [
            0.45717,
            0.09800,
            0.31973,
        ]  # 门把手的目标位置
        self.door_pos = [0, 0, 0]
        self.cabinet_pos = [0, 0, 0]
        self.current_index = 0

        self.x, self.y, self.z = [], [], []  # 轨迹坐标

    def callback_pose(self, msg):
        # 处理接收到的消息
        self.init_pose = [msg.position.x, msg.position.y, msg.position.z]
        if not self.first_message_received:
            # 初始化轨迹
            self.x, self.y, self.z = self.trace_trajectory_Astar(
                d0=self.init_pose, goal_pos=self.goal_pos, tf=10.0, freq=200
            )
            # 添加第二段轨迹
            self.add_second_segment()
            # 取消订阅
            self.destroy_subscription(self.sub_1)
            self.first_message_received = True

    def callback_grasp(self, msg):
        pass

    def timer_callback(self):
        if self.current_index < len(self.x):
            pose = Pose()
            pose.position.x = self.x[self.current_index]
            pose.position.y = self.y[self.current_index]
            pose.position.z = self.z[self.current_index]
            pose.orientation.x = 0.0
            pose.orientation.y = 1.0
            pose.orientation.z = 0.0
            pose.orientation.w = 0.0

            self.pub.publish(pose)
            self.current_index += 1
        else:
            self.timer.cancel()

    def trace_trajectory_Astar(self, d0, goal_pos, tf, freq, gri_open=True):
        plan = Motion_planning(
            self.door_pos,
            self.cabinet_pos,
            dx=0.01,
            dy=0.01,
            dz=0.01,
            gripper_open=gri_open,
        )
        Points_recover = plan.path_searching(start=d0, end=goal_pos)
        if Points_recover is not None:
            x, y, z = plan.path_smoothing(
                Path_points=Points_recover, t_final=tf, freq=freq
            )  # 使用三次样条曲线平滑轨迹
            return x, y, z

        else:
            print("未找到路径!!")
            return None

    def add_second_segment(self):
        # 第二段目标沿y轴移动-0.18m
        second_goal_pos = [self.goal_pos[0], self.goal_pos[1] - 0.2, self.goal_pos[2]]

        # 第二段轨迹的时间参数
        t_start = 0
        t_end = 5  # 第二段轨迹的持续时间为5秒
        t_points = np.linspace(t_start, t_end, int(t_end * 200))  # 100 Hz频率

        # 使用线性插值计算第二段的轨迹点
        second_x = np.linspace(self.goal_pos[0], second_goal_pos[0], len(t_points))
        second_y = np.linspace(self.goal_pos[1], second_goal_pos[1], len(t_points))
        second_z = np.linspace(self.goal_pos[2], second_goal_pos[2], len(t_points))

        # 将第二段轨迹附加到现有轨迹
        self.x = np.concatenate((self.x, second_x))
        self.y = np.concatenate((self.y, second_y))
        self.z = np.concatenate((self.z, second_z))


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningNode("path_planning_node")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
