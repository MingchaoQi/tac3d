import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R, Slerp


class PoseControl(Node):
    def __init__(self):
        super().__init__("pose_control")

        # 发布目标位姿的发布者
        self.pose_pub = self.create_publisher(Pose, "/lbr/command/pose", 10)

        # 订阅当前末端执行器的位姿
        self.pose_sub = self.create_subscription(
            Pose, "/lbr/state/pose", self.pose_callback, 10
        )

        # 用于存储当前位置和目标位置
        self.current_pose = None
        self.desired_pose = None
        self.timer = self.create_timer(0.01, self.publish_intermediate_pose)  # 100 Hz

        self.q_tool_to_base = np.array([0.0, 1.0, 0.0, 0.0])

        # 设定最大步长
        self.max_step = 0.01  # 每次的最大位移步长
        self.max_angle_step = 0.01  # 每次的最大角度步长（弧度）

    def pose_callback(self, msg):
        # 存储接收到的当前位姿
        self.current_pose = msg

        # 基于固定角度计算目标姿态
        self.desired_pose = self.simple_transform_and_align(self.current_pose)

    def simple_transform_and_align(self, current_pose):
        # 固定角度调整（绕y轴旋转）
        fixed_angle = 0.555  # 固定旋转角度，单位：弧度

        # 生成绕y轴的旋转四元数
        rotation_quat = R.from_euler("y", fixed_angle, degrees=False).as_quat()

        # 获取当前姿态的四元数
        current_quat = np.array(
            [
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w,
            ]
        )

        # 将固定旋转四元数应用到当前姿态

        # new_quat = R.from_quat(current_quat) * R.from_quat(rotation_quat)
        r_final = R.from_quat(self.q_tool_to_base) * R.from_quat(rotation_quat)

        # 创建新的目标姿态
        desired_pose = Pose()
        desired_pose.position = current_pose.position  # 保持当前位置不变

        # 设置目标姿态的四元数
        new_quat_arr = r_final.as_quat()
        desired_pose.orientation.x = new_quat_arr[0]
        desired_pose.orientation.y = new_quat_arr[1]
        desired_pose.orientation.z = new_quat_arr[2]
        desired_pose.orientation.w = new_quat_arr[3]

        return desired_pose

    def publish_intermediate_pose(self):
        if self.current_pose is None or self.desired_pose is None:
            return

        # 插值位置
        position_diff = np.array(
            [
                self.desired_pose.position.x - self.current_pose.position.x,
                self.desired_pose.position.y - self.current_pose.position.y,
                self.desired_pose.position.z - self.current_pose.position.z,
            ]
        )

        distance = np.linalg.norm(position_diff)
        if distance > self.max_step:
            step = position_diff / distance * self.max_step
        else:
            step = position_diff

        # 插值四元数
        current_quat = np.array(
            [
                self.current_pose.orientation.x,
                self.current_pose.orientation.y,
                self.current_pose.orientation.z,
                self.current_pose.orientation.w,
            ]
        )

        desired_quat = np.array(
            [
                self.desired_pose.orientation.x,
                self.desired_pose.orientation.y,
                self.desired_pose.orientation.z,
                self.desired_pose.orientation.w,
            ]
        )

        angle_diff = (
            np.arccos(np.clip(np.dot(current_quat, desired_quat), -1.0, 1.0)) * 2.0
        )
        if angle_diff > self.max_angle_step:
            t = self.max_angle_step / angle_diff
        else:
            t = 1.0

        # 使用 Slerp 进行四元数插值
        key_rots = R.from_quat([current_quat, desired_quat])
        slerp = Slerp([0, 1], key_rots)
        interpolated_quat = slerp(t).as_quat()

        # 创建新的中间目标位置
        intermediate_pose = Pose()
        intermediate_pose.position.x = self.current_pose.position.x + step[0]
        intermediate_pose.position.y = self.current_pose.position.y + step[1]
        intermediate_pose.position.z = self.current_pose.position.z + step[2]

        # 插值后的四元数
        intermediate_pose.orientation.x = interpolated_quat[0]
        intermediate_pose.orientation.y = interpolated_quat[1]
        intermediate_pose.orientation.z = interpolated_quat[2]
        intermediate_pose.orientation.w = interpolated_quat[3]

        # 发布中间目标位置
        self.pose_pub.publish(intermediate_pose)


def main(args=None):
    rclpy.init(args=args)
    pose_control = PoseControl()
    rclpy.spin(pose_control)
    pose_control.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
