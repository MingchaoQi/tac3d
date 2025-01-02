import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from motion_planning_v3 import Motion_planning
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp
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
        self.door_pos = [0.5, 0.5, 0.5]
        self.cabinet_pos = [0.5, 0.5, 0.5]
        self.goal_pos = [0.47983, -0.51596, 0.27926]  # 杯子的目标位置
        self.current_index = 0

        # 工具坐标系相对于基坐标系的四元数 (0, 1, 0, 0)
        self.q_tool_to_base = np.array([0.0, 1.0, 0.0, 0.0])

        self.x, self.y, self.z = [], [], []  # 轨迹的坐标
        self.quat = []  # 姿态轨迹的四元数

    def callback_pose(self, msg):
        # 提取当前机械臂姿态的四元数
        current_quat = np.array(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        )

        # 处理接收到的消息
        self.init_pose = [msg.position.x, msg.position.y, msg.position.z]
        if not self.first_message_received:
            # 计算偏移后的初始位置
            offset_pose = [
                self.init_pose[0] + 0.10,  # x方向偏移10cm
                self.init_pose[1] + 0.10,  # y方向偏移10cm
                self.init_pose[2] + 0.10,  # z方向偏移10cm
            ]

            # 生成从初始位置到偏移后位置的平滑轨迹（使用三次样条插值）
            smooth_x, smooth_y, smooth_z = self.generate_smooth_trajectory(
                self.init_pose, offset_pose, tf=2.0, freq=200
            )

            # 生成姿态轨迹，从当前姿态到目标姿态
            target_quat = self.calculate_target_orientation()

            # 姿态插值在 1 秒内完成
            smooth_quat = self.generate_smooth_orientation_trajectory(
                current_quat, target_quat, tf=10.0, freq=300
            )

            # 从偏移后的位置规划到目标位置
            plan_x, plan_y, plan_z = self.trace_trajectory_Astar(
                d0=offset_pose, goal_pos=self.goal_pos, tf=10.0, freq=300
            )

            # 姿态保持目标姿态不变
            plan_quat = np.tile(target_quat, (len(plan_x), 1))

            # 拼接两段轨迹
            self.x = np.concatenate((smooth_x, plan_x))
            self.y = np.concatenate((smooth_y, plan_y))
            self.z = np.concatenate((smooth_z, plan_z))
            self.quat = np.concatenate((smooth_quat, plan_quat))

            # 取消订阅
            self.destroy_subscription(self.sub_1)
            self.first_message_received = True

    def generate_smooth_trajectory(self, start_pos, end_pos, tf, freq):
        """
        生成从 start_pos 到 end_pos 的平滑插值轨迹。
        使用三次样条插值生成平滑轨迹。
        """
        t = np.linspace(0, tf, int(tf * freq))  # 时间数组
        t_points = [0, tf]  # 起始和结束时间点

        # 为 x, y, z 使用三次样条插值
        spline_x = CubicSpline(t_points, [start_pos[0], end_pos[0]])
        spline_y = CubicSpline(t_points, [start_pos[1], end_pos[1]])
        spline_z = CubicSpline(t_points, [start_pos[2], end_pos[2]])

        # 计算插值点
        x = spline_x(t)
        y = spline_y(t)
        z = spline_z(t)

        return x, y, z

    def generate_smooth_orientation_trajectory(self, start_quat, end_quat, tf, freq):
        """
        生成从 start_quat 到 end_quat 的四元数平滑插值轨迹。
        使用 Slerp 插值生成平滑姿态。
        """
        t = np.linspace(0, tf, int(tf * freq))  # 时间数组

        # 定义起点和终点的四元数
        r_start = R.from_quat(start_quat)
        r_end = R.from_quat(end_quat)

        # 创建 Slerp 插值对象，注意传入的是 `Rotation` 对象列表
        slerp = Slerp([0, tf], R.from_quat([start_quat, end_quat]))

        # 对时间进行插值
        r_interpolated = slerp(t)

        # 返回插值后的四元数列表
        return r_interpolated.as_quat()

    def calculate_target_orientation(self):
        """
        计算目标末端姿态的四元数
        """
        angles = [-90, 26.00]  # 绕 Z 轴 -90 度，绕 X 轴 -25 度
        angles_rad = np.radians(angles)  # 将角度转换为弧度

        # 创建旋转对象，旋转顺序为 z -> x，表示先绕 Z 再绕 X，固定坐标系旋转
        r = R.from_euler("zx", angles_rad, degrees=False).as_quat()

        r_final = R.from_quat(self.q_tool_to_base) * R.from_quat(r)
        q_final = r_final.as_quat()

        return q_final

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

            quat = self.quat[self.current_index]
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]

            self.pub.publish(pose)
            self.current_index += 1
        else:
            self.timer.cancel()
            subprocess.run(["pkill", "-f", "open_door_3"])  # 强制关闭节点

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
