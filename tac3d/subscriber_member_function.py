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

import rclpy
from rclpy.node import Node
from tutorial_interfaces.msg import Array3, Cloud
from std_msgs.msg import Float32


class Tac3DSubscriber(Node):

    def __init__(self):
        super().__init__("tac3d_subscriber")
        self.subscription_array = self.create_subscription(
            Array3, "array_topic", self.callback_fr, 10
        )
        self.subscription_cloud = self.create_subscription(
            Cloud, "cloud_topic", self.callback_cloud, 10
        )
        self.subscription_float = self.create_subscription(
            Float32, "float_topic", self.callback_float, 10
        )

    def callback_fr(self, msg):
        self.get_logger().info(f"Received Fr data: {msg.x}, {msg.y}, {msg.z}")

    def callback_cloud(self, msg):
        self.get_logger().info(
            f"Received Cloud data: row1={msg.row1}, row2={msg.row2}, row3={msg.row3}"
        )

    def callback_float(self, msg):
        self.get_logger().info(f"Received Float data: {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    tac3d_subscriber = Tac3DSubscriber()
    rclpy.spin(tac3d_subscriber)
    tac3d_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
