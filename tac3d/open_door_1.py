import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from motion_planning_v3 import Motion_planning


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
        # self.goal_pos = [0.45717, 0.08653, 0.31973]
        # self.goal_pos = [0.45717, 0.08653-0.2, 0.31973]
        self.goal_pos = [0.58577, -0.50593, 0.36816]  # 门把手的目标位置
        self.door_pos = [0.5, 0.5, 0.5]
        self.cabinet_pos = [0.5, 0.5, 0.5]
        self.current_index = 0

        self.x, self.y, self.z = [], [], []  # 轨迹的坐标

    def callback_pose(self, msg):
        # 处理接收到的消息
        self.init_pose = [msg.position.x, msg.position.y, msg.position.z]
        if not self.first_message_received:
            # 初始化轨迹
            self.x, self.y, self.z = self.trace_trajectory_Astar(
                d0=self.init_pose, goal_pos=self.goal_pos, tf=10.0, freq=100
            )
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
