import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R


class OpenDoor(Node):
    def __init__(self):
        super().__init__("open_door")
        self.pub = self.create_publisher(Pose, "/lbr/command/pose", 10)
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.sub = self.create_subscription(
            Pose, "/lbr/state/pose", self.callback_pose, 10
        )
        self.index = 0

        # 圆弧轨迹参数
        self.center = [0.30577, -0.51593, 0.36816]
        self.theta_start = 0
        self.theta_end = np.pi * 87 / 180
        self.num_points = 2000
        self.radius = 0.28

        self.first_message_received = False  # 还未收到位姿信息

        # 工具坐标系相对于基坐标系的四元数 (0, 1, 0, 0)
        self.q_tool_to_base = np.array([0.0, 1.0, 0.0, 0.0])

    def callback_pose(self, msg):
        # 处理接收到的消息
        self.init_pose = [msg.position.x, msg.position.y, msg.position.z]
        if not self.first_message_received:
            # 收到初始位姿后，生成轨迹
            self.transition_points = self.generate_transition_points()
            self.arc_points = self.generate_arc_points()
            self.points = np.concatenate(
                (self.transition_points, self.arc_points), axis=0
            )

            # 计算切线方向
            self.tangents = self.calculate_tangents(self.arc_points)

            self.first_message_received = True
            # 可以取消订阅以节省资源
            self.destroy_subscription(self.sub)

    def timer_callback(self):
        if self.first_message_received and self.index < len(self.points):
            point = self.points[self.index]

            pose = Pose()
            pose.position.x = point[0]
            pose.position.y = point[1]
            pose.position.z = point[2]

            if self.index < len(self.transition_points):
                # 在过渡阶段使用固定四元数
                pose.orientation.x = 0.0
                pose.orientation.y = 1.0
                pose.orientation.z = 0.0
                pose.orientation.w = 0.0
            else:
                # 绕工具坐标系 z 轴旋转角度 angle（单位：弧度）
                tangent = self.tangents[self.index - len(self.transition_points)]
                angle = np.pi / 2 - np.arctan2(tangent[1], tangent[0])
                # 生成绕Z轴的四元数
                q_rot = R.from_euler("z", angle).as_quat()  # [x, y, z, w]

                # 创建四元数旋转对象
                r_tool_to_base = R.from_quat(self.q_tool_to_base)
                r_rot = R.from_quat(q_rot)

                # 等价于基坐标系下的旋转
                # r_final = r_tool_to_base * r_rot * r_tool_to_base.inv()
                r_final = r_tool_to_base * r_rot
                quaternion = r_final.as_quat()  # [x, y, z, w]

                pose.orientation.x = quaternion[0]
                pose.orientation.y = quaternion[1]
                pose.orientation.z = quaternion[2]
                pose.orientation.w = quaternion[3]

            self.pub.publish(pose)
            self.index += 1
        elif not self.first_message_received:
            self.get_logger().info("Waiting for initial pose...")
        else:
            self.get_logger().info("Arc trajectory finished.")

    def generate_transition_points(self):
        """生成从初始位置到圆弧轨迹起点的平滑过渡点"""
        arc_start_point = [
            self.center[0] + self.radius * np.cos(self.theta_start),
            self.center[1] + self.radius * np.sin(self.theta_start),
            self.center[2],
        ]

        # 生成从初始点到圆弧起点的线性过渡轨迹
        transition_points = np.linspace(self.init_pose, arc_start_point, 100)
        return transition_points

    def generate_arc_points(self):
        """生成圆弧轨迹点"""
        theta = np.linspace(self.theta_start, self.theta_end, self.num_points)
        x_c, y_c, z_c = self.center
        x = x_c + self.radius * np.cos(theta)
        y = y_c + self.radius * np.sin(theta)
        z = np.full_like(theta, z_c)
        return np.stack([x, y, z], axis=1)

    def calculate_tangents(self, points):
        """计算每个点的切线方向"""
        tangents = []
        for i in range(len(points) - 1):
            tangent = points[i + 1] - points[i]
            tangent_norm = tangent / np.linalg.norm(tangent)  # 归一化
            tangents.append(tangent_norm)
        return tangents


def main(args=None):
    rclpy.init(args=args)
    node = OpenDoor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
