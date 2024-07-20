# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入必要的模块
import time
import rclpy
from rclpy.node import Node
from tutorial_interfaces.msg import Array3
from tutorial_interfaces.msg import Cloud
from std_msgs.msg import Float32
from PyTac3D import Sensor


class Tac3DPublisher(Node):

    def __init__(self):
        super().__init__('tac3d_publisher')
        self.publisher_array = self.create_publisher(Array3, 'force_r', 10)
        self.publisher_cloud = self.create_publisher(Cloud, 'position_r', 10)
        self.publisher_float = self.create_publisher(Float32, 'index_r', 10)
        self.sensor = Sensor(recvCallback=self.Tac3DRecvCallback, port=9988)


    def Tac3DRecvCallback(self, frame, param):
        # 处理帧数据的逻辑，例如获取各种数据
        SN = 'A1-0040R'  # 假设SN的获取逻辑
        idx = frame['index']
        P = frame.get('3D_Positions')
        D = frame.get('3D_Displacements')
        F = frame.get('3D_Forces')
        Fr = frame.get('3D_ResultantForce')
        Mr = frame.get('3D_ResultantMoment')

        # 发布各种类型的消息
        if Fr is not None and len(Fr[0]) >= 3:
            self.publish_array3_from_frame(Fr[0], 'Fr', '3D_ResultantForce')
        else:
            self.get_logger().warn('未能获取到有效的 Fr 数据或数据不完整')

        # if Mr is not None and len(Mr[0]) >= 3:
        #     self.publish_array3_from_frame(Mr[0], 'Mr', '3D_ResultantMoment')
        # else:
        #     self.get_logger().warn('未能获取到有效的 Mr 数据或数据不完整')

        if P is not None:  # need data integrity control
            self.publish_cloud_from_frame(P, 'P', '3D_Positions')
        else:
            self.get_logger().warn('未能获取到有效的 P 数据或数据不完整')

        if idx is not None:
            self.publish_note_from_frame(idx, 'idx', 'index')
        else:
            self.get_logger().warn('未能获取到有效的 idx 数据或数据不完整')

    def publish_note_from_frame(self, data, frame_name, frame_source):
        msg = Float32()
        # data_l = frame.get(frame_source)]
        if data is not None:
            msg.data = float(data)  # float of ROS2
            self.publisher_float.publish(msg)
            # self.get_logger().info(f'发布 {frame_name} 数据:{msg}')
        else:
            self.get_logger().warn(f'未能获取到有效的 {frame_name} 数据或数据不完整')

    def publish_array3_from_frame(self, data, frame_name, frame_source):
        msg = Array3()
        # data_l = frame.get(frame_source)
        if data is not None and len(data) >= 3:
            msg.x = float(data[0])
            msg.y = float(data[1])
            msg.z = float(data[2])
            self.publisher_array.publish(msg)
            # self.get_logger().info(f'发布 {frame_name} 数据: {msg.x}, {msg.y}, {msg.z}')
        else:
            self.get_logger().warn(f'未能获取到有效的 {frame_name} 数据或数据不完整')

    def publish_cloud_from_frame(self, data, frame_name, frame_source):
        msg = Cloud()
        # data_l = frame.get(frame_source)
        if data is not None:  # need data integrity control
            msg.row1 = [data[i][0] for i in range(400)]
            msg.row2 = [data[i][1] for i in range(400)]
            msg.row3 = [data[i][2] for i in range(400)]
            self.publisher_cloud.publish(msg)
            # self.get_logger().info(f'发布 {frame_name} 数据: row1={msg.row1}, row2={msg.row2}, row3={msg.row3}')
        else:
            self.get_logger().warn(f'未能获取到有效的 {frame_name} 数据或数据不完整')

    def publish(self):
        # 等待 Tac3D-Desktop 启动传感器并建立连接
        self.sensor.waitForFrame()

        time.sleep(5)  # 等待 5 秒钟

        # 发送一次校准信号（确保传感器未与任何物体接触！）
        self.sensor.calibrate('A1-0040R')

        time.sleep(5)  # 等待 5 秒钟

        # 从缓存队列中获取 frame
        while True:
            frame = self.sensor.getFrame()
            if frame is not None:
                self.Tac3DRecvCallback(frame, None)
            # else:
                # self.get_logger().warn('未能获取到有效的 Fr 数据')


def main(args=None):
    rclpy.init(args=args)
    tac3d_publisher = Tac3DPublisher()
    tac3d_publisher.publish()
    rclpy.spin(tac3d_publisher)
    tac3d_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

