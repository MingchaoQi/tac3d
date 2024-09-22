import cv2
import numpy as np
from gelsight import gsdevice
from image_boundary import boundary_depth
import time
from image_flow import Flow


# 比较器
def find_stable_region(current_depth_image):
    print("find_stable_region activated")
    global previous_depth_image

    if previous_depth_image is not None:
        # 计算当前深度图与上一帧深度图之间的差别
        diff = cv2.absdiff(previous_depth_image, current_depth_image)
        diff_max = np.max(diff)

        # 设定阈值，找到变化的区域
        threshold = 0.8 * diff_max
        stable_region = np.where(diff > threshold, 255, 0).astype(np.uint8)

        # 更新上一帧深度图
        previous_depth_image = current_depth_image.copy()

        return stable_region
    else:
        # 第一帧深度图直接作为上一帧深度图
        previous_depth_image = current_depth_image.copy()
        return np.zeros_like(current_depth_image)  # 返回空结果


def zoom(real_calibre, edges):
    #  real_calibre标准圆孔已知的直径，确保边缘检测图像清晰（请人工观测，确保算法得到的标准孔边缘合理）
    # edges输入图像边缘，这个矩阵实际上是图像索引，反映原图像的大小和边缘位置（边缘处索引非零）
    # 函数打印并返回图像缩放比：像素距离（个）/真实距离(mm)
    # 获取边缘像素的坐标
    edge_coordinates = []
    height, width = edges.shape
    print('确认图像长宽比：', height, ':', width)
    for y in range(height):
        for x in range(width):
            if edges[y, x] != 0:
                edge_coordinates.append((x, y))

    # 计算 x、y 坐标差距的最大值
    max_x_diff = 0
    max_y_diff = 0
    for coord1 in edge_coordinates:
        for coord2 in edge_coordinates:
            x_diff = abs(coord1[0] - coord2[0])
            y_diff = abs(coord1[1] - coord2[1])
            if x_diff > max_x_diff:
                max_x_diff = x_diff
            if y_diff > max_y_diff:
                max_y_diff = y_diff

    # 比较得到图像缩放比：像素距离（个）/真实距离(mm)
    zoom = (max_x_diff + max_y_diff) / real_calibre
    print('图像缩放比为', zoom)
    return zoom


# 边缘检测
def depth_boundary_detection(img, flag):
    img_raw = cv2.resize(img, (320, 240))  # 320 240
    img_small = img_raw.copy()
    dm = nn.get_depthmap(img_small, mask_markers=True)
    # 保存关键点
    contour_points = []
    # 每隔一秒激活函数
    # 判断是否已经过去 interval 秒
    if flag:
        # 执行需要做的操作
        global stable_region
        stable_region = find_stable_region(dm)

    # 高斯滤波
    ksize = 5  # 高斯核大小，它应该是一个奇数正整数，尺寸越大平滑效果越明显。
    sigma = 2.2  # 高斯核标准差，方向上的扩散程度，越小保留的细节越多
    gray_blur = cv2.GaussianBlur(dm, (ksize, ksize), sigma)
    # 计算x方向的梯度
    gradient_x = cv2.Sobel(dm, cv2.CV_64F, 1, 0, ksize=3)
    # 计算y方向的梯度
    gradient_y = cv2.Sobel(dm, cv2.CV_64F, 0, 1, ksize=3)
    # 计算梯度幅值和方向
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # gradient_direction = np.arctan2(gradient_y, gradient_x)


    # 将梯度幅值图像转换为 CV_8U 类型
    gradient_magnitude_8U = cv2.convertScaleAbs(gradient_magnitude)

    # 根据阈值将图像分为纹理、边界和无梯度三部分
    # texture = int(np.mean(gradient_magnitude_8U * 15))

    # value = np.mean(gradient_magnitude_8U) * (gradient_magnitude_8U.size - np.count_nonzero(gradient_magnitude_8U)) / np.count_nonzero(gradient_magnitude_8U)
    # value = np.mean(gradient_magnitude_8U) * gradient_magnitude_8U.size / np.count_nonzero(gradient_magnitude_8U)
    # if np.isnan(value):
    #     print("Value is NaN")
    # else:
    #     texture = int(value)
    #     print('max:', np.max(gradient_magnitude_8U))
    #     print('texture:', texture)
    #     edges = cv2.Canny(gradient_magnitude_8U, texture, 2 * texture)
    #     # !!!Why must use the max of the texture, or our texture feature is not perfect
    #     img_small[edges != 0] = [255, 0, 0]

    texture = int(np.max(gradient_magnitude_8U))
    # 计算非零值的平均值 不可靠——为什么？
    # non_zero_values = gradient_magnitude_8U[gradient_magnitude_8U > 0]
    # if non_zero_values.size > 0:
    #     texture = int(np.mean(non_zero_values))
    # else:
    #     texture = 255  # 如果没有非零值，可以选择一个默认值或者处理逻辑

    print('max:', np.max(gradient_magnitude_8U))
    print('texture:', texture)
    # 应用 Canny 边缘检测
    edges = cv2.Canny(gradient_magnitude_8U, texture, 2 * texture)

    # 膨胀和腐蚀
    # kernel = np.ones((3, 3), np.uint8)
    # dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    # edges = cv2.erode(dilated_edges, kernel, iterations=1)

    # 双边滤波：蓝色
    img_small[edges != 0] = [255, 0, 0]
    img_edge = img_small.copy()

    # 准备PCA直线分析
    # 获取所有边缘点的坐标
    points = np.column_stack(np.where(edges > 0))

    if points.shape[0] == 0:
        # raise ValueError("没有检测到边缘点，无法进行PCA分析。")
        pass
    else:
        # 使用PCA分析边缘点的分布方向
        mean, eigenvectors = cv2.PCACompute(points.astype(np.float32), mean=np.array([]))

        # 第一主成分向量（最大轴）表示直线的方向
        vx, vy = eigenvectors[1]
        # 旋转90度
        # vx, vy = -vy, vx  # 交换vx, vy并改变符号，得到垂直于第一主成分的方向

        # 拟合直线并计算端点
        x0, y0 = mean[0]  # 中心点
        x1 = int(x0 + vx * 100)
        y1 = int(y0 + vy * 100)
        x2 = int(x0 - vx * 100)
        y2 = int(y0 - vy * 100)

        # 在图像上绘制直线
        # cv2.line(img_small, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 使用Hough变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=25, maxLineGap=10)
        # rho、theta：分辨率
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_small, (x1, y1), (x2, y2), (0, 255, 128), 2)


    # 阈值化边缘图像
    threshold = 2
    edges_dm = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)[1]

    # 将图像转换为8位单通道的二值图像
    # edges_dm = cv2.convertScaleAbs(edges_dm)

    # 轮廓提取
    # contours, _ = cv2.findContours(edges_dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 纹理区
    # threshold_mask = (gradient_magnitude_8U[:, :] > texture/2) | (gradient_magnitude_8U[:, :] < 0.0001)
    # gradient_magnitude[threshold_mask] = 0
    # 深度记忆：绿色
    # img_small[stable_region != 0] = [0, 255, 0]

    # 定义显示窗口的大小
    window_width = 640
    window_height = 480

    # 调整图像的大小
    resized_edges_dm = cv2.resize(edges_dm, (window_width, window_height))
    resized_gradient_magnitude = cv2.resize(gradient_magnitude, (window_width, window_height))
    resized_img_small = cv2.resize(img_small, (window_width, window_height))

    # 显示调整后的图像
    cv2.imshow('Edges', resized_edges_dm)
    cv2.imshow('Gradient Magnitude', resized_gradient_magnitude)
    cv2.imshow('Contours Image', resized_img_small)

    if flag:
        timestamp = str(time.time()).replace(".", "")
        file_name = f"stable_data/image_{timestamp}"
        # 将其归一化到0-255 范围
        gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(file_name + '.jpg', img_raw)
        cv2.imwrite(file_name + 'w.jpg', img_small)
        cv2.imwrite(file_name + 'g.jpg', gradient_magnitude_normalized)
        cv2.imwrite(file_name + 'd.jpg', img_edge)

    return contour_points


# 初始化
previous_depth_image = None
interval = 0.5
flag = 0
stable_region = np.zeros([240, 320], dtype=np.uint8)
# 记录开始时间
# start_time = time.time()

# 读取GelSight图像

# dev=gsdevice.Camera("GelSight Mini R0B 28GH-EJZ8")
# dev.connect()
# f0 = dev.get_raw_image()
# print(f0)


nn, dev = boundary_depth(cam_id="GelSight Mini")  # dev = cam


while True and cv2.waitKey(1) == -1:
    # # 计时器
    # current_time = time.time()
    # if current_time - start_time >= interval:
    #     # 执行需要做的操作
    #     print("One second has passed. Do something here.")
    #     flag = 1
    #     # 更新开始时间
    #     start_time = current_time

    img = dev.get_raw_image()
    if img is None or img.size == 0:
        print("Error: Unable to get image from camera.")
        continue  # 或者退出循环

    # 使用深度信息对边缘进行过滤
    contour_points_depth = depth_boundary_detection(img, flag)
    flag = 1

cv2.destroyAllWindows()
