import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray
from scipy.spatial.transform import Rotation as R, Slerp

class PoseControl(Node):
    def __init__(self):
        super().__init__('pose_control')

        # 发布目标位姿的发布者
        self.pose_pub = self.create_publisher(Pose, '/lbr/command/pose', 10)

        # 订阅当前末端执行器的位姿
        self.pose_sub = self.create_subscription(
            Pose, '/lbr/state/pose', self.pose_callback, 10
        )

        # 订阅Gelsight的二维方向数据
        self.orientation_sub = self.create_subscription(
            Float32MultiArray, '/gelsight/orientation', self.control_callback, 10
        )

        # 用于存储当前位置
        self.current_pose = None

        # 设定最大步长
        self.max_step = 0.01  # 每次的最大位移步长
        self.max_angle_step = 0.1  # 每次的最大角度步长（弧度）

    def pose_callback(self, msg):
        # 存储接收到的当前位姿
        self.current_pose = msg

    def control_callback(self, msg):
        if self.current_pose is None:
            self.get_logger().warn("Current pose is not available yet.")
            return

        # 获取从夹爪平面传来的二维方向向量
        direction_2d = np.array([msg.data[0], msg.data[1]])

        # 使用存储的当前位置
        current_pose = self.current_pose

        # 将二维方向向量转化为三维空间中的向量
        desired_pose = self.simple_transform_and_align(direction_2d, current_pose)

        # 确保物体竖直对齐
        # desired_pose = self.ensure_vertical_alignment(direction_3d, current_pose)

        # 插值并发布目标位姿
        self.move_towards_pose(current_pose, desired_pose)

    # def transform_to_3d(self, direction_2d, current_pose):
    #     # 将2D方向向量映射到3D空间，假设夹爪平面是XZ平面
    #     # direction_2d 的第一个值表示x方向，第二个值表示z方向
    #     direction_3d_local = np.array([direction_2d[0], 0.0, direction_2d[1]])  # 在局部坐标系中的方向向量
    #
    #     # 从 current_pose 中提取四元数 (x, y, z, w)
    #     quat = np.array([
    #         current_pose.orientation.x,
    #         current_pose.orientation.y,
    #         current_pose.orientation.z,
    #         current_pose.orientation.w
    #     ])
    #
    #     # 使用四元数旋转将局部坐标系中的3D方向向量转换为全局坐标系中的向量
    #     # 这里我们使用scipy的Rotation类
    #     rotation = R.from_quat(quat)
    #
    #     # 将局部方向向量旋转到全局坐标系中
    #     direction_3d_global = rotation.apply(direction_3d_local)
    #
    #     return direction_3d_global

    def simple_transform_and_align(self, direction_2d, current_pose):
        # 将2D方向向量映射到夹角 (与x轴的夹角)
        angle = np.arctan2(direction_2d[1], direction_2d[0])  # 计算与x轴的夹角，返回值范围为 [-pi, pi]
        print(angle)
        angle = np.pi - angle

        # 生成绕z轴的旋转四元数
        rotation_quat = R.from_euler('y', angle, degrees=False).as_quat(False)  # 生成绕z轴的旋转

        # 获取当前姿态的四元数
        current_quat = np.array([
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ])

        # 将绕z轴的旋转四元数应用到当前姿态
        new_quat = R.from_quat(current_quat) * R.from_quat(rotation_quat)

        # 创建新的目标姿态
        desired_pose = Pose()
        desired_pose.position = current_pose.position  # 保持当前位置不变

        # 设置目标姿态的四元数
        new_quat_arr = new_quat.as_quat()
        desired_pose.orientation.x = new_quat_arr[0]
        desired_pose.orientation.y = new_quat_arr[1]
        desired_pose.orientation.z = new_quat_arr[2]
        desired_pose.orientation.w = new_quat_arr[3]

        return desired_pose

    def move_towards_pose(self, current_pose, desired_pose):
        # 插值位置
        position_diff = np.array([
            desired_pose.position.x - current_pose.position.x,
            desired_pose.position.y - current_pose.position.y,
            desired_pose.position.z - current_pose.position.z
        ])

        distance = np.linalg.norm(position_diff)
        if distance > self.max_step:
            step = position_diff / distance * self.max_step
        else:
            step = position_diff

        # 插值四元数（使用球面线性插值）
        current_quat = np.array([
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ])

        desired_quat = np.array([
            desired_pose.orientation.x,
            desired_pose.orientation.y,
            desired_pose.orientation.z,
            desired_pose.orientation.w
        ])

        # 计算插值比例
        angle_diff = np.arccos(np.clip(np.dot(current_quat, desired_quat), -1.0, 1.0)) * 2.0
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
        intermediate_pose.position.x = current_pose.position.x + step[0]
        intermediate_pose.position.y = current_pose.position.y + step[1]
        intermediate_pose.position.z = current_pose.position.z + step[2]

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

if __name__ == '__main__':
    main()

