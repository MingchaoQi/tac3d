import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.spatial.transform import Rotation as R

from motion_planning_v3 import Motion_planning

def trace_trajectory_Astar(d0, goal_pos, tf, freq, gri_open=True):
    plan = Motion_planning(dx=0.01, dy=0.01, dz=0.01, gripper_open=gri_open)
    Points_recover = plan.path_searching(start=d0, end=goal_pos)
    if Points_recover is not None:
        X, Y, Z = plan.path_smoothing(
            Path_points=Points_recover, t_final=tf, freq=freq
        )  # 轨迹使用二次B样条曲线进行平滑处理
    else:
        print("the path is not found!!")
        return None
    t0 = env.sim.data.time
    action = np.zeros(action_dim)
    goal_force = np.array([0, 0, 0])
    last_pos = np.append(d0, np.zeros(action_dim - 3))

    observation = []
    Reward = []
    Force = np.zeros(0)

    cp = np.array([2000, 2000, 2000])
    mp = np.array([1, 1, 1]) * 80
    bp = 2 * 0.707 * np.sqrt(cp * mp)

    for i in range(int(tf * freq * 1.2)):
        if i < len(plan.d):
            # 位置控制（这里使用的已经是经过插值平滑轨迹后的轨迹点）
            x_s = np.append(plan.d[i], np.zeros(action_dim - 3))
            v_s = np.append(plan.d_dot[i], np.zeros(action_dim - 3))
            while (env.sim.data.time - t0) < plan.t[i]:
                MyEuler1 = R.from_quat(env._eef_xquat).as_euler("zyx")
                # MyEuler1 = T.mat2euler(T.quat2mat(env._eef_xquat))
                current_pos = np.append(env._eef_xpos, np.zeros(action_dim - 3))
                kp = np.array([20, 20, 20, 0, 0, 0])
                kp = np.append(kp, np.ones(action_dim - 6))
                # action = vel_control(current_pos, x_s, k=kp)
                kd = 0.7 * np.sqrt(kp)

        else:
            goal_pos = x_s
            MyEuler1 = R.from_quat(env._eef_xquat).as_euler("zyx")
            # MyEuler1 = T.mat2euler(T.quat2mat(env._eef_xquat))
            current_pos = np.append(
                env._eef_xpos, np.array([MyEuler1[2], MyEuler1[1], MyEuler1[0], 0])
            )
            kp = np.array([20, 20, 20, 0, 0, 0])
            kp = np.append(kp, np.ones(action_dim - 6))
            action = vel_control(current_pos, goal_pos, k=kp)
            # action = np.array([0, 0, 0, 0, 0, 0, 0])

            obs, reward, done, info = env.step(action)  # take action in the environment
            observation.append(obs)
            Reward.append(reward)
            env.render()  # render on display
        if i % 2 == 0:  # 隔一步显示一次力信息
            ee_force = env.sim.data.sensordata[0:3]
            # print('ee_force=', ee_force)
            # print("ori_current= ", T.quat2axisangle(env._eef_xquat))
            Force = np.append(Force, ee_force)
        last_pos = current_pos
    print("-----------------end the trajectory--------------------")
    MyEuler = T.quat2axisangle(env._eef_xquat)
    print(
        "ee_pos_x=",
        np.append(env._eef_xpos, np.array([MyEuler[0], MyEuler[1], MyEuler[2]])),
    )
    print("goal_pos=", x_s)

    Force = Force.reshape(-1, 3)

    return observation, Reward, Force


# ----------------------------------------------------------------------------------------------------------------------
# create environment instance
robots = "IIWA"
env = robosuite_task_zoo.environments.manipulation.HammerPlaceEnv(
    robots,
    has_renderer=True,
    # gripper_types="RobotiqThreeFingerGripper",
    has_offscreen_renderer=True,
    use_camera_obs=True,
    render_camera="frontview",
    control_freq=20,
    controller_configs=suite.load_controller_config(
        default_controller="OSC_POSE"
    ),  # 操作空间位置控制
)

env.reset()
# env.viewer.set_camera(camera_id=0)
print("-------------------start the trajectory-------------------------")
## set the action dim
action_dim = env.action_dim  # in robot_env.py
neutral = np.zeros(action_dim)



def rotate_mat(axis, radian):
    """
    利用旋转矩阵并结合知识图谱中提取的相关信息，计算门把手的最终运动位置
    旋转矩阵 欧拉角
    """
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix


# 分别是x,y和z轴以及旋转轴
axis_x, axis_y, axis_z = [1, 0, 0], [0, 1, 0], [0, 0, 1]
# 使用列表形式存放数据，可避免数据转换的麻烦
rand_axis = [0, 0, 1]
# 旋转角度为joint_range
# yaw = joint_range["door"][1]
yaw = 0.4
# 返回旋转矩阵
rot_matrix = rotate_mat(rand_axis, yaw)
# 计算点绕着轴运动后的点(变换到门坐标系)
# door_joint_pos = env.door_joint_pos # 获取关节位置坐标有问题，现在没有找到具体原因
# 但是通过另一种方式获得joint的位置，就是通过与door的相对位置关系
door_joint_pos = env.door_pos - np.array([0.225, 0, 0])  # 这里直接手动输入joint和door的相对位置关系，调用知识图谱中信息如下
# door_joint_pos = env.door_pos - np.array(joint_pos["door"])
print("门铰链的初始位置：", env.door_joint_pos)
print("门把手初始位置：", env._handle_xpos)
# x = np.array(env._handle_xpos)
x = np.array(env._handle_xpos) - np.array([door_joint_pos[0], door_joint_pos[1], 0])
# 旋转后的坐标，需要再次变换到世界坐标系
x1 = np.dot(rot_matrix, x)
x1 = x1 + np.array([door_joint_pos[0], door_joint_pos[1], 0])
print("门把手目标位置：", x1)

# 第一段轨迹
MyEuler = R.from_quat(env._eef_xquat).as_euler("xyz")
d_0 = env._eef_xpos
print("d0=", d_0)
# goal_pos = env._slide_handle_xpos
goal_pos = env._handle_xpos
t_f = 10.0
# 检验t
t1 = env.sim.data.time
obs1, _, force1 = trace_trajectory_Astar(d_0, goal_pos, tf=t_f, freq=20)
print("delta_t=", env.sim.data.time - t1)
print("handle_pos_fin=", env._handle_xpos)
print("slide_handle_pos_fin=", env._slide_handle_xpos)

# 第二段轨迹（开关夹爪）
action = neutral.copy()
action[-1] = 1
for i in range(20):
    obs_2, reward, done, info = env.step(action)
    env.render()  # render on display
print("handle_pos_fin=", env._handle_xpos)
print("slide_handle_pos_fin=", env._slide_handle_xpos)

## 第三段轨迹
MyEuler = R.from_quat(env._eef_xquat).as_euler("zyx")
d_0 = env._eef_xpos
print("d0=", d_0)
# goal_pos = env._handle_xpos + np.array([0, 0.2, 0.0])
goal_pos = x1
goal_force = np.array([0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0])
##检验t
t1 = env.sim.data.time
obs_3, _, force2 = trace_trajectory_Astar(
    d_0, goal_pos, tf=t_f, freq=20, gri_open=False
)
print("delta_t=", env.sim.data.time - t1)
print("handle_pos_fin=", env._handle_xpos)
print("slide_handle_pos_fin=", env._slide_handle_xpos)
print("门把手目标位置：", x1)

env.close()

obs = obs1 + obs_3
pos_x, pos_y, pos_z = [], [], []
for j in range(len(obs)):
    pos_x.append(obs[j]["robot0_eef_pos"][0])
    pos_y.append(obs[j]["robot0_eef_pos"][1])
    pos_z.append(obs[j]["robot0_eef_pos"][2])

ax1 = plt.axes(projection='3d')
ax1.plot3D(pos_x, pos_y, pos_z, 'blue')
plt.figure()

force = force1
force = np.append(force, force2)
force = force.reshape(-1, 3)
force_x, force_y, force_z = [], [], []
freq_force = 10
for i in range(len(force)):
    force_x.append(force[i][0])
    force_y.append(force[i][1])
    force_z.append(force[i][2])
