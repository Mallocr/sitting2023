import torch
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import easygui


# 创建曲线
class Descartes:
    def __init__(self, points:torch.Tensor):
        self.x_points = (points[:, 0]).to('cpu').numpy()  # 提取每个批次的所有点的 x 坐标
        self.y_points = (points[:, 1]).to('cpu').numpy()  # 提取每个批次的所有点的 y 坐标
        self.z_points = (points[:, 2]).to('cpu').numpy()  # 提取每个批次的所有点的 z 坐标

        # 使用三维插值拟合这些点
        interp_func_x = interp1d(np.arange(len(self.x_points)), self.x_points, kind='cubic')
        interp_func_y = interp1d(np.arange(len(self.y_points)), self.y_points, kind='cubic')
        interp_func_z = interp1d(np.arange(len(self.z_points)), self.z_points, kind='cubic')

        # 生成更多点以获得平滑的曲线
        self.t = np.linspace(0, len(self.x_points) - 1, 1000)
        self.x_interp = interp_func_x(self.t)
        self.y_interp = interp_func_y(self.t)
        self.z_interp = interp_func_z(self.t)

        # 绘制原始离散点和插值曲线
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x_points, self.y_points, self.z_points, label='Discrete Points', color='red')
        ax.plot(self.x_interp, self.y_interp, self.z_interp, label='Interpolated Curve', color='blue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Interpolated 3D Curve from Discrete Points')
        ax.legend()

    def judge(self) -> None:
        # 删除首尾指定比例的点，例如删除5%
        percent_to_trim = 0.08
        num_points_to_trim = int(percent_to_trim * len(self.t))

        # 避免删除超过数组长度的点
        num_points_to_trim = min(num_points_to_trim, len(self.x_interp)//2)

        # 修剪插值后的点数
        trimmed_x_interp = self.x_interp[num_points_to_trim:-num_points_to_trim]
        trimmed_y_interp = self.y_interp[num_points_to_trim:-num_points_to_trim]
        trimmed_z_interp = self.z_interp[num_points_to_trim:-num_points_to_trim]
        trimmed_t = self.t[num_points_to_trim:-num_points_to_trim]

        # 计算修剪后曲线的一阶导数和二阶导数
        dx = np.gradient(trimmed_x_interp, trimmed_t)
        dy = np.gradient(trimmed_y_interp, trimmed_t)
        dz = np.gradient(trimmed_z_interp, trimmed_t)

        ddx = np.gradient(dx, trimmed_t)
        ddy = np.gradient(dy, trimmed_t)
        ddz = np.gradient(dz, trimmed_t)

        # 计算曲线在每个点处的切向量和其导数
        tangent_vector = np.array([dx, dy, dz])
        tangent_derivative = np.array([ddx, ddy, ddz])

        # 计算切向量的叉乘和模长
        cross_product = np.cross(tangent_vector.T, tangent_derivative.T)
        tangent_norm = np.linalg.norm(tangent_vector, axis=0)

        # 计算曲率
        curvature = np.linalg.norm(cross_product, axis=1) / tangent_norm**3

        average_curvature = np.mean(curvature)
        print(average_curvature)
        # 设置曲率阈值
        threshold = 18

        # 判断曲率是否大于阈值
        if average_curvature > threshold:
            # easygui.msgbox("Warning: High curvature detected", title="Message From Descartes")
            print("Warning.")
        else:
            print("No warning: Curvature is below the threshold.")


def show() -> None:
    plt.show()
