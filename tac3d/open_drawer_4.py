import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import numpy as np
from scipy.interpolate import CubicSpline


class PathPlanningNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.pub = self.create_publisher(Pose, "/lbr/command/pose", 10)
        self.sub_1 = self.create_subscription(
            Pose, "/lbr/state/pose", self.callback_pose, 10
        )
        self.sub_2 = self.create_subscription(Pose, "moveto", self.callback_grasp, 10)
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.first_message_received = False  # 标记是否已经接收到第一次消息
        self.goal_pos = [0.47683, 0, 0.37926]  # 放置杯子的目标位置
        self.current_index = 0
        self.x, self.y, self.z = [], [], []  # 轨迹的坐标
        self.orientation = None  # 用于存储初始姿态

    def callback_pose(self, msg):
        # 处理接收到的初始位置消息
        self.init_pose = [msg.position.x, msg.position.y, msg.position.z]
        self.orientation = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ]  # 保存姿态
        if not self.first_message_received:
            # 生成向上移动5cm的轨迹
            upward_goal = [
                self.init_pose[0],
                self.init_pose[1],
                self.init_pose[2] + 0.05,
            ]
            upward_x, upward_y, upward_z = self.trace_trajectory_spline(
                start=self.init_pose, goal=upward_goal, tf=2.0, freq=100
            )
            # 生成从上升5cm后的轨迹到目标位置的轨迹
            main_x, main_y, main_z = self.trace_trajectory_spline(
                start=upward_goal, goal=self.goal_pos, tf=8.0, freq=100
            )
            # 将两个轨迹合并
            self.x = np.concatenate((upward_x, main_x))
            self.y = np.concatenate((upward_y, main_y))
            self.z = np.concatenate((upward_z, main_z))

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

            # 保持初始姿态
            pose.orientation.x = self.orientation[0]
            pose.orientation.y = self.orientation[1]
            pose.orientation.z = self.orientation[2]
            pose.orientation.w = self.orientation[3]

            self.pub.publish(pose)
            self.current_index += 1
        else:
            self.timer.cancel()

    def trace_trajectory_spline(self, start, goal, tf, freq):
        # 根据给定的起点和终点生成三次样条插值路径
        t = np.linspace(0, tf, int(tf * freq))  # 生成时间点
        # 对 x, y, z 分别进行三次样条插值
        x_spline = CubicSpline([0, tf], [start[0], goal[0]])
        y_spline = CubicSpline([0, tf], [start[1], goal[1]])
        z_spline = CubicSpline([0, tf], [start[2], goal[2]])

        x_traj = x_spline(t)
        y_traj = y_spline(t)
        z_traj = z_spline(t)

        return x_traj, y_traj, z_traj


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningNode("path_planning_node")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
