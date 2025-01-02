import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from motion_planning_v3 import Motion_planning
from scipy.interpolate import CubicSpline
import numpy as np
import subprocess


class PathPlanningNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.pub = self.create_publisher(Pose, "/lbr/command/pose", 10)
        self.sub_1 = self.create_subscription(
            Pose, "/lbr/state/pose", self.callback_pose, 10
        )
        self.sub_2 = self.create_subscription(Pose, "moveto", self.callback_grasp, 10)
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.first_message_received = False  # 用于标记是否已经接收到第一次消息
        # self.goal_pos = [0.63016, -0.18632, 0.38653]  # 杯子目标位置0.36437
        self.goal_pos = [0.58225, -0.14936, 0.39037]  # 杯子目标位置
        self.door_pos = [0, 0, 0]
        self.cabinet_pos = [0, 0, 0]
        self.current_index = 0

        # 将 self.x, self.y, self.z 初始化为空的 numpy 数组
        self.x, self.y, self.z = np.array([]), np.array([]), np.array([])

    def callback_pose(self, msg):
        # 处理接收到的消息
        self.init_pose = [msg.position.x, msg.position.y, msg.position.z]
        if not self.first_message_received:
            # 1. 生成向上运动10cm的轨迹
            self.x, self.y, self.z = self.generate_initial_upward_trajectory(
                d0=self.init_pose, delta_z=0.17, tf=2.0, freq=300
            )

            # 2. 生成沿y轴正方向运动10cm的轨迹
            x_y_traj, y_y_traj, z_y_traj = self.generate_y_positive_trajectory(
                d0=[self.x[-1], self.y[-1], self.z[-1]],
                delta_y=-(0.09349 - (0.09800 - 0.2)),
                tf=2.0,
                freq=300,
            )

            # 将沿y轴运动的轨迹拼接到向上运动之后
            self.x = np.concatenate((self.x, x_y_traj))
            self.y = np.concatenate((self.y, y_y_traj))
            self.z = np.concatenate((self.z, z_y_traj))

            # 3. 使用A*算法生成到目标位置的轨迹
            astar_x, astar_y, astar_z = self.trace_trajectory_Astar(
                d0=[self.x[-1], self.y[-1], self.z[-1]],
                goal_pos=self.goal_pos,
                tf=10.0,
                freq=100,
            )

            if astar_x is not None:
                # 使用 np.concatenate 拼接 numpy 数组
                self.x = np.concatenate((self.x, astar_x))
                self.y = np.concatenate((self.y, astar_y))
                self.z = np.concatenate((self.z, astar_z))

            # 取消订阅
            self.destroy_subscription(self.sub_1)
            self.first_message_received = True

    def callback_grasp(self, msg):
        pass

    def timer_callback(self):
        if len(self.x) == 0:
            return
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
            subprocess.run(["pkill", "-f", "open_drawer_4"])  # 强制关闭节点

    def generate_initial_upward_trajectory(self, d0, delta_z, tf, freq):
        """生成向上移动10cm的轨迹，使用三次样条插值"""
        t = np.linspace(0, tf, int(tf * freq))
        z0 = d0[2]
        z_target = z0 + delta_z

        # 生成z方向的三次样条插值
        z_spline = CubicSpline([0, tf], [z0, z_target])

        # 生成整个轨迹
        x_traj = np.full(len(t), d0[0])  # x保持不变
        y_traj = np.full(len(t), d0[1])  # y保持不变
        z_traj = z_spline(t)  # z根据样条插值生成

        return x_traj, y_traj, z_traj

    def generate_y_positive_trajectory(self, d0, delta_y, tf, freq):
        """生成沿y轴正方向运动10cm的轨迹，使用三次样条插值"""
        t = np.linspace(0, tf, int(tf * freq))
        y0 = d0[1]
        y_target = y0 + delta_y

        # 生成y方向的三次样条插值
        y_spline = CubicSpline([0, tf], [y0, y_target])

        # 生成整个轨迹
        x_traj = np.full(len(t), d0[0])  # x保持不变
        y_traj = y_spline(t)  # y根据样条插值生成
        z_traj = np.full(len(t), d0[2])  # z保持不变

        return x_traj, y_traj, z_traj

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
            )  # 轨迹使用三次B样条曲线进行平滑处理
            return x, y, z

        else:
            print("the path is not found!!")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningNode("path_planning_node")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
