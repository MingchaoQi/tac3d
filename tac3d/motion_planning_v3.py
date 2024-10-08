import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import splrep, splev


class Motion_planning:
    def __init__(
        self, door_pos, cabinet_pos, dx, dy, dz, gripper_open=True
    ):  ##增添了判断夹爪是否张开的判断
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.gripper_open = gripper_open
        self.door_pos = door_pos
        self.cabinet_pos = cabinet_pos

    def range_box_G(
        self,
        start,
        target,
        start_range=np.array([0.02, 0.02, 0.02, 0.01]),
        target_range=np.array([0.03, 0.03, 0.02]),
    ):  # 新的体素划分方式，考虑到夹爪张开时需要避让目标处的障碍物（如果有的话）；夹爪关闭则不用考虑避让目标处的障碍物
        point1 = start
        # 障碍物的中心点
        obstacle1 = self.door_pos  # 门障碍物的中心点
        obstacle2 = self.cabinet_pos  # 抽屉障碍物的中心点
        point2 = target
        obstacle1_box = np.array([0.025, 0.03, 0.06])  ##后面改成自动获取障碍物尺寸信息
        obstacle2_box = np.array([0.013, 0.015, 0.015])

        gripper_mini = np.array(
            [
                point1[0] - start_range[0],
                point1[1] - start_range[1],
                point1[2] - start_range[3],
            ]
        )
        gripper_max = np.array(
            [
                point1[0] + start_range[0],
                point1[1] + start_range[1],
                point1[2] + start_range[2],
            ]
        )

        ## 形成矩形包络空间
        Points = np.array(
            [
                point1,
                point2,
                obstacle1,
                obstacle2,
                obstacle1 - obstacle1_box / 2,
                obstacle1 + obstacle1_box / 2,
                obstacle2 - obstacle2_box / 2,
                obstacle2 + obstacle2_box / 2,
                point2 - target_range / 2,
                point2 + target_range / 2,
                gripper_mini,
                gripper_max,
            ]
        ).reshape(-1, 3)
        self.min_x, self.max_x = np.min(Points[:, 0]), np.max(Points[:, 0])
        self.min_y, self.max_y = np.min(Points[:, 1]), np.max(Points[:, 1])
        self.min_z, self.max_z = np.min(Points[:, 2]), np.max(Points[:, 2])

        self.height = int((self.max_z - self.min_z) / self.dz) + 1
        self.length = int((self.max_x - self.min_x) / self.dx) + 1
        self.width = int((self.max_y - self.min_y) / self.dy) + 1
        self.Map = np.zeros(
            shape=(self.length, self.width, self.height)
        )  ##创建了一个三维地图，为全空
        self.gripper_range = [
            int(start_range[0] / self.dx),
            int(start_range[1] / self.dy),
            int(start_range[2] / self.dz),
            int(start_range[3] / self.dz),
        ]
        self.target_range = [
            int(target_range[0] / self.dx),
            int(target_range[1] / self.dy),
            int(target_range[2] / self.dz),
        ]

        self.start_point = [
            int((point1[0] - self.min_x) / self.dx),
            int((point1[1] - self.min_y) / self.dy),
            int((point1[2] - self.min_z) / self.dz),
        ]
        self.end_point = [
            int((point2[0] - self.min_x) / self.dx),
            int((point2[1] - self.min_y) / self.dy),
            int((point2[2] - self.min_z) / self.dz),
        ]

        ##把障碍点表示在地图中
        obstacle1_point = [
            int((obstacle1[0] - self.min_x) / self.dx),
            int((obstacle1[1] - self.min_y) / self.dy),
            int((obstacle1[2] - self.min_z) / self.dz),
        ]
        obstacle2_point = [
            int((obstacle2[0] - self.min_x) / self.dx),
            int((obstacle2[1] - self.min_y) / self.dy),
            int((obstacle2[2] - self.min_z) / self.dz),
        ]

        for i in range(
            max(int(obstacle1_point[0] - obstacle1_box[0] / (2 * self.dx)), 0),
            min(
                int(obstacle1_point[0] + obstacle1_box[0] / (2 * self.dx)), self.length
            ),
            1,
        ):
            for j in range(
                max(int(obstacle1_point[1] - obstacle1_box[1] / (2 * self.dy)), 0),
                min(
                    int(obstacle1_point[1] + obstacle1_box[1] / (2 * self.dy)),
                    self.width,
                ),
                1,
            ):
                for k in range(
                    max(int(obstacle1_point[2] - obstacle1_box[2] / (2 * self.dz)), 0),
                    min(
                        int(obstacle1_point[2] + obstacle1_box[2] / (2 * self.dz)),
                        self.height,
                    ),
                    1,
                ):
                    self.Map[i][j][k] = 1
        for i in range(
            max(int(obstacle2_point[0] - obstacle2_box[0] / (2 * self.dx)), 0),
            min(
                int(obstacle2_point[0] + obstacle2_box[0] / (2 * self.dx)), self.length
            ),
            1,
        ):
            for j in range(
                max(int(obstacle2_point[1] - obstacle2_box[1] / (2 * self.dy)), 0),
                min(
                    int(obstacle2_point[1] + obstacle2_box[1] / (2 * self.dy)),
                    self.width,
                ),
                1,
            ):
                for k in range(
                    max(int(obstacle2_point[2] - obstacle2_box[2] / (2 * self.dz)), 0),
                    min(
                        int(obstacle2_point[2] + obstacle2_box[2] / (2 * self.dz)),
                        self.height,
                    ),
                    1,
                ):
                    self.Map[i][j][k] = 1

        if (
            self.gripper_open
        ):  ##如果夹爪为打开状态，那么需要检测目标点附近的障碍；如果夹爪为关闭状态，则不需要考虑目标点附近的障碍

            for i in range(
                max(int(self.end_point[0] - target_range[0] / (2 * self.dx)), 0),
                min(
                    int(self.end_point[0] + target_range[0] / (2 * self.dx)),
                    self.length,
                ),
                1,
            ):
                for j in range(
                    max(int(self.end_point[1] - target_range[1] / (2 * self.dy)), 0),
                    min(
                        int(self.end_point[1] + target_range[1] / (2 * self.dy)),
                        self.width,
                    ),
                    1,
                ):
                    for k in range(
                        max(
                            int(self.end_point[2] - target_range[2] / (2 * self.dz)), 0
                        ),
                        min(
                            int(self.end_point[2] + target_range[2] / (2 * self.dz)),
                            self.height,
                        ),
                        1,
                    ):
                        self.Map[i][j][k] = 1  ##增加运动目标点的障碍范围

            self.end_point = [
                int((point2[0] - self.min_x) / self.dx),
                int((point2[1] - self.min_y) / self.dy),
                int(
                    (point2[2] + target_range[2] + start_range[3] + 0.01 - self.min_z)
                    / self.dz
                ),
            ]  ##结束点应为目标点向上平移

    def path_searching(self, start, end):  ##代码的核心，路径规划生成路径点
        self.range_box_G(start=start, target=end)
        self.Astar = A_Search(
            self.start_point, self.end_point, self.Map, self.gripper_range
        )
        # 启发式算法生成路径
        Result = self.Astar.process()
        if len(Result) > 0:
            Points_collection = []
            Recover = []
            for i in Result:
                print("path:(%d,%d,%d)" % (i.x, i.y, i.z))
                Points_collection.append([i.x, i.y, i.z])
                recover_point = [
                    self.dx * i.x + self.min_x,
                    self.dy * i.y + self.min_y,
                    self.dz * i.z + self.min_z,
                ]
                Recover.append(recover_point)
            Recover = np.array(Recover)
            Recover = Recover[::-1]
            Recover = np.append(start, Recover).reshape(-1, 3)

            # Generate points between the last point in Recover and the end point in z direction
            last_point = Recover[-1]
            z_points = []
            z_step = 0.005
            z_diff = end[2] - last_point[2]

            # If the z_diff is large enough, we insert points with 0.01 step in the z direction
            if abs(z_diff) > z_step:
                num_steps = int(abs(z_diff) / z_step)
                z_values = np.linspace(last_point[2], end[2], num_steps + 1)[
                    1:-1
                ]  # Exclude start and end points
                for z in z_values:
                    z_points.append([end[0], end[1], z])

            z_points = (
                np.array(z_points) if z_points else np.empty((0, 3))
            )  # In case no points were added

            Recover = np.vstack(
                [Recover, z_points, end]
            )  # Append vertical points and end point
            return Recover
        else:
            print("the path is not found!")
            return None

    def path_smoothing(self, Path_points, t_final, freq):
        self.n = int(t_final * freq)
        self.t = np.zeros(self.n)
        self.d = np.zeros((self.n, len(self.start_point)))

        Path_points = np.array(Path_points)
        # 分别对每个坐标轴进行插值
        t = np.arange(len(Path_points))
        tck_x = splrep(t, Path_points[:, 0], k=3, s=0)
        tck_y = splrep(t, Path_points[:, 1], k=3, s=0)
        tck_z = splrep(t, Path_points[:, 2], k=3, s=0)
        # 生成插值后的坐标点
        t_new = np.linspace(0, len(Path_points) - 1, self.n)
        x_new = splev(t_new, tck_x)
        y_new = splev(t_new, tck_y)
        z_new = splev(t_new, tck_z)

        return x_new, y_new, z_new


class point:  # 点类（每一个唯一坐标只有对应的一个实例）
    _list = []  # 储存所有的point类实例
    _tag = True  # 标记最新创建的实例是否为_list中的已有的实例，True表示不是已有实例

    def __new__(cls, key):  # 重写new方法实现对于同样的坐标只有唯一的一个实例
        for i in point._list:
            if i.x == key[0] and i.y == key[1] and i.z == key[2]:
                point._tag = False
                return i
        nt = super(point, cls).__new__(cls)
        point._list.append(nt)
        return nt

    def __init__(self, key):
        x = key[0]
        y = key[1]
        z = key[2]
        if point._tag:
            self.x = x
            self.y = y
            self.z = z
            self.father = None
            self.F = 0  # 当前点的评分  F=G+H
            self.G = 0  # 起点到当前节点所花费的消耗
            self.cost = 0  # 父节点到此节点的消耗
        else:
            point._tag = True

    @classmethod
    def clear(
        cls,
    ):  # clear方法，每次搜索结束后，将所有点数据清除，以便进行下一次搜索的时候点数据不会冲突。
        point._list = []

    def __eq__(self, T):  # 重写==运算以便实现point类的in运算
        if type(self) == type(T):
            return (self.x, self.y, self.z) == (T.x, T.y, T.z)
        else:
            return False

    def __str__(self):
        return "(%d,%d, %d)[F=%d,G=%d,cost=%d][father:(%s)]" % (
            self.x,
            self.y,
            self.z,
            self.F,
            self.G,
            self.cost,
            str((self.father.x, self.father.y)) if self.father != None else "null",
        )


class A_Search:  # 核心部分，寻路类
    def __init__(self, arg_start, arg_end, arg_map, gri_range):  # arg_start
        self.start = point(arg_start)  # 储存此次搜索的开始点
        self.end = point(arg_end)  # 储存此次搜索的目的点
        self.Map = arg_map  # 一个三维数组，为此次搜索的地图引用
        self.gri_range = gri_range
        self.Map_scan = []
        self.open = []  # 开放列表：储存即将被搜索的节点
        self.close = []  # 关闭列表：储存已经搜索过的节点
        self.result = []  # 当计算完成后，将最终得到的路径写入到此属性中
        self.count = 0  # 记录此次搜索所搜索过的节点数
        self.useTime = 0  # 记录此次搜索花费的时间--在此演示中无意义，因为process方法变成了一个逐步处理的生成器，统计时间无意义。
        # 开始进行初始数据处理
        self.open.append(self.start)

    def cal_F(self, loc):
        # print('计算值：', loc)
        G = loc.father.G + loc.cost
        H = self.getEstimate(loc)
        F = G + H
        # print("F=%d G=%d H=%d" % (F, G, H))
        return {"G": G, "H": H, "F": F}

    def F_Min(
        self,
    ):  # 搜索open列表中F值最小的点并将其返回，同时判断open列表是否为空，为空则代表搜索失败
        if len(self.open) <= 0:
            return None
        t = self.open[0]
        for i in self.open:
            if i.F < t.F:
                t = i
        return t

    def getAroundPoint(
        self, loc
    ):  # 获取指定点周围所有可通行的点，并将其对应的移动消耗进行赋值。
        nl = []
        print("start to find the aroundPoint of the tar")

        for i in [0, -1, 1]:
            for j in [0, -1, 1]:
                for k in [0, -1, 1]:
                    x = loc.x + i
                    y = loc.y + j
                    z = loc.z + k

                    if (
                        x < 0
                        or x >= self.Map.shape[0]
                        or y < 0
                        or y >= self.Map.shape[1]
                        or z < 0
                        or z >= self.Map.shape[2]
                        or self.Map[x][y][z] == 1
                    ):  ## 排除范围外的点和障碍点
                        print("the (%d,%d,%d) has out of range" % (x, y, z))
                    elif i == j == k == 0:
                        print("reach the father point")
                    else:
                        if (
                            self.Map[x - i][y][z]
                            + self.Map[x][y - j][z]
                            + self.Map[x][y][z - k]
                        ) == 0:  ## 排除存在障碍点的路径
                            nt = point([x, y, z])
                            nt.cost = self.getFcost(i, j, k)
                            nl.append(nt)
                            # print("the (%d,%d,%d) is normal" % (x, y, z))
                        else:
                            print("the (%d,%d,%d) is not reached" % (x, y, z))
        # print('nl:', nl)
        print("the aroundPoint have been gotten")

        return nl

    def getAroundPoint_G(self, loc):  ##考虑夹爪避碰的路径搜索策略
        nl = []
        print("start to find the aroundPoint of the tar")

        for i in [0, -1, 1]:
            for j in [0, -1, 1]:
                for k in [0, -1, 1]:
                    x = loc.x + i
                    y = loc.y + j
                    z = loc.z + k

                    max_x, min_x = min(
                        x + self.gri_range[0], self.Map.shape[0] - 1
                    ), max(x - self.gri_range[0], 0)
                    max_y, min_y = min(
                        y + self.gri_range[1], self.Map.shape[1] - 1
                    ), max(y - self.gri_range[1], 0)
                    max_z, min_z = min(
                        z + self.gri_range[2], self.Map.shape[2] - 1
                    ), max(z - self.gri_range[3], 0)
                    flag = 0

                    if (
                        x < 0
                        or x >= self.Map.shape[0]
                        or y < 0
                        or y >= self.Map.shape[1]
                        or z < 0
                        or z >= self.Map.shape[2]
                        or self.Map[x][y][z] == 1
                    ):  ## 排除范围外的点和障碍点
                        flag = 1
                        print("the (%d,%d,%d) has out of range" % (x, y, z))
                    elif i == j == k == 0:
                        flag = 1
                        print("reach the father point")
                    else:
                        for x1 in range(min_x, max_x + 1, 1):
                            for y1 in range(min_y, max_y + 1, 1):
                                for z1 in range(min_z, max_z + 1, 1):
                                    if (
                                        self.Map[x1][y1][z1] == 1 and flag == 0
                                    ):  ## 排除范围外的点和障碍点
                                        flag = 1
                                        print(
                                            "the (%d,%d,%d) has out of range"
                                            % (x, y, z)
                                        )
                                        continue

                                    if (
                                        flag == 0
                                        and 0 <= (x1 - i) < self.Map.shape[0]
                                        and 0 <= (y1 - i) < self.Map.shape[1]
                                        and 0 <= (z1 - i) < self.Map.shape[2]
                                    ):  ## 排除存在障碍点的路径
                                        if (
                                            self.Map[x1 - i][y1][z1]
                                            + self.Map[x1][y1 - j][z1]
                                            + self.Map[x1][y1][z1 - k]
                                        ) > 0:
                                            flag = 1
                                            print(
                                                "the (%d,%d,%d) is not reached"
                                                % (x, y, z)
                                            )
                                            continue

                    if flag == 0:
                        nt = point([x, y, z])
                        nt.cost = self.getFcost(i, j, k)
                        nl.append(nt)
        # print('nl:', nl)
        print("the aroundPoint have been gotten")

        return nl

    def addToOpen(
        self, l, father
    ):  # 此次判断的点周围的可通行点加入到open列表中，如此点已经在open列表中则对其进行判断，如果此次路径得到的F值较之之前的F值更小，则将其父节点更新为此次判断的点，同时更新F、G值。
        for i in l:
            if i not in self.open:
                if i not in self.close:
                    i.father = father
                    self.open.append(i)
                    r = self.cal_F(i)
                    i.G = r["G"]
                    i.F = r["F"]
            else:
                tf = i.father
                i.father = father
                r = self.cal_F(i)
                if i.F > r["F"]:
                    i.G = r["G"]
                    i.F = r["F"]
                # i.father=father
                else:
                    i.father = tf

        print("the points have been addtoOpen")

    def getEstimate(self, loc):  # H :从点loc移动到终点的预估花费
        return (
            abs(loc.x - self.end.x) + abs(loc.y - self.end.y) + abs(loc.z - self.end.z)
        ) * 10

    def getFcost(self, dx, dy, dz):
        if abs(dx) + abs(dy) + abs(dz) == 1:
            cost = 10
        elif abs(dx) + abs(dy) + abs(dz) == 2:
            cost = 14
        elif abs(dx) + abs(dy) + abs(dz) == 3:
            cost = 17
        else:
            cost = 0

        return cost

    def process(self):
        while True:
            self.count += 1
            tar = self.F_Min()  # 先获取open列表中F值最低的点tar
            if tar == None:
                self.result = None
                self.count = -1
                print("the target is None")
                break
            else:
                print("tar=", tar)
                # aroundP = self.getAroundPoint(tar)  # 获取tar周围的可用点列表aroundP
                aroundP = self.getAroundPoint_G(tar)
                self.addToOpen(
                    aroundP, tar
                )  # 把aroundP加入到open列表中并更新F值以及设定父节点
                self.open.remove(tar)  # 将tar从open列表中移除
                self.close.append(tar)  # 已经迭代过的节点tar放入close列表中
                if self.end in self.open:  # 判断终点是否已经处于open列表中
                    e = self.end
                    self.result.append(e)
                    while True:
                        e = e.father
                        if e == None:
                            break
                        self.result.append(e)
                    point.clear()
                    break

        return self.result
