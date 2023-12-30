import asyncio
from articulate.math import *
import binascii
from bleak import BleakClient
from pygame.time import Clock
import time
import torch
import torch.nn as nn
import threading
import descartes

checkpoint_path = 'E:\\PyCharm\\Sitting_Posture_Detection\\models\\model_checkpoint.pth'

batch_size = 12  # 根据你的数据设置实际的批次大小
num_joints = 5  # 或根据实际关节数量设置
parent_node = [-1,0,1,2,3]
initial_position = torch.tensor([
    [-0.0022, -0.2408, 0.0286],
    [0.0023, -0.1164, -0.0098],
    [0.0068, 0.0216, 0.0170],
    [0.0045, 0.0776, 0.0199],
    [-0.0089, 0.2892, -0.0136]
]).unsqueeze(0).repeat(batch_size, 1, 1).to('cuda:0')  # 重复以匹配批次大小


def rotation_matrix_to_axis_angles(r: torch.Tensor):
    _, num_nodes, _, _ = r.shape  # 获取批次大小和节点数量
    result = []
    for m in range(num_nodes):
        node_matrix = r[:, m, :, :]  # 获取第m个节点所有的旋转矩阵
        node_matrix=torch.squeeze(node_matrix, dim=1)
        axis_angle = rotation_matrix_to_axis_angle(node_matrix)  # 计算轴-角表示法
        result.append(axis_angle)  # 将结果堆叠为张量并添加到结果列表
    result = torch.stack(result, dim=1)  # 沿着第二维度堆叠结果张量
    return result.to('cuda:0')


def axis_angles_to_rotation_matrix(p: torch.Tensor):
    _, num_nodes, _ = p.shape  # 获取批次大小和节点数量
    result = []
    for m in range(num_nodes):
        node_angles = p[:, m, :]  # 获取第m个节点所有的pose
        node_angles=torch.squeeze(node_angles, dim=1)
        node_matrix = axis_angle_to_rotation_matrix(node_angles)  # 计算旋转矩阵表示法
        result.append(node_matrix)  # 将结果堆叠为张量并添加到结果列表
    result = torch.stack(result, dim=1)  # 沿着第二维度堆叠结果张量
    return result.to('cuda:0')


# 在更新joints位置时，考虑batch处理
def update_joint_position(joint_positions, rotation_matrices, joint, parent_node, batch_size):
    # 根节点无需更新
    if parent_node[joint] == -1:
        return

    # 更新父节点位置
    update_joint_position(joint_positions, rotation_matrices, parent_node[joint], parent_node, batch_size)

    # 获取当前关节和父关节的旋转矩阵
    joint_rotation_matrix = rotation_matrices[:, joint, :, :]
    parent_rotation_matrix = rotation_matrices[:, parent_node[joint], :, :]

    # 对batch中每个样本应用旋转和位移
    for b in range(batch_size):
        # 应用父关节变换到当前关节的位移
        joint_positions[b, joint] = (parent_rotation_matrix[b] @ joint_positions[b, joint].unsqueeze(-1)).squeeze(-1) + \
                                    joint_positions[b, parent_node[joint]]


def poses_to_joints(pose_result: torch.Tensor):
    joint_positions = initial_position.clone().to('cuda:0')
    rotation_matrices = axis_angles_to_rotation_matrix(pose_result)
    for j in range(num_joints):
        update_joint_position(joint_positions, rotation_matrices, j, parent_node, batch_size)
    return joint_positions


class LSTMModel(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_lstm_layer, bidirectional=False, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.lstm = nn.LSTM(n_hidden, n_hidden, n_lstm_layer, batch_first=True)
        self.linear2 = nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)

    def forward(self, x):
        x = self.dropout(x)  # Apply dropout if necessary
        x = self.relu(self.linear1(x))  # Linear layer + Activation
        x, (hidden, cell) = self.lstm(x)  # LSTM layer
        # Take the last sequence output
        x = x[:, -1, :] if x.dim() == 3 else x  # Check if x is indeed 3D
        x = self.linear2(x)  # Final linear layer
        return x


checkpoint = torch.load(checkpoint_path)
inertial_pose = LSTMModel(
    n_input=checkpoint['input_size'],
    n_hidden=checkpoint['hidden_size'],
    n_output=checkpoint['output_size'],
    n_lstm_layer=checkpoint['num_layers']
)
inertial_pose.load_state_dict(checkpoint['model_state_dict'])

# 设置推理模式
inertial_pose.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
inertial_pose.to(device)


# 预测函数
def predict(model, in_put):
    with torch.no_grad():
        inputs = in_put.to(device)
        outputs = model(inputs)
    return outputs


def turn_to_decimal(a):
    ints = 0xFFFF
    tto = int(str(a), 16)
    if tto > 32753:
        tto = tto - ints - 1
    return tto


class BLEServer:

    def __init__(self, address, name):
        self.address = address
        self.name = name
        self.data_buffer = []
        self.connected = 0

    async def seton(self):
        self.connected = 1

    async def moveoff(self):
        self.connected = 0

    async def main(self, address, write_char_uuid, notify_char_uuid):

        async def notification_handler(sender, data):
            """Simple notification handler which prints the data received."""
            data = str(binascii.b2a_hex(data), 'utf-8')
            if len(data) != 84:
                # 错误数据会打印出来
                print(f"{self.name}:error data: {data}")
            else:

                self.data_buffer.append(data)
                if len(self.data_buffer) > BUFFER_SIZE:
                    self.data_buffer.pop(0)

        def disconnected_callback(client):
            print("Disconnected callback called!")
            disconnected_event.set()

        disconnected_event = asyncio.Event()

        try:
            client = BleakClient(address, disconnected_callback=disconnected_callback, mtu_size=200)
            await client.connect()
            print(f"Connected: {client.is_connected}")
            d_name = "AB01FFFFFF"
            await self.seton()
            await client.write_gatt_char(write_char_uuid, bytes.fromhex(d_name))
            await client.start_notify(notify_char_uuid, notification_handler)
            await asyncio.sleep(10)
            await disconnected_event.wait()
            await self.moveoff()
        except Exception as e:
            print(e)
            print("Disconnect:", self.name)

    def mean_data_transition(self):
        tensor_list = []
        clock = Clock()
        start_time = time.time()
        duration = 2
        while time.time() - start_time < duration:
            clock.tick(60)
            tensor = self.data_transition()
            tensor_list.append(tensor)
        if len(tensor_list):
            input_tensor = torch.mean(torch.stack(tensor_list, dim=0), dim=0)
            return input_tensor
        return None

    def data_transition(self):
        data = self.get_latest_data()
        if data:
            quat_x = (turn_to_decimal(data[10:12] + data[8:10])) / 10000
            quat_y = (turn_to_decimal(data[14:16] + data[12:14])) / 10000
            quat_z = (turn_to_decimal(data[18:20] + data[16:18])) / 10000
            quat_w = (turn_to_decimal(data[22:24] + data[20:22])) / 10000
            acc_x = (turn_to_decimal(data[38:40] + data[36:38])) / 100
            acc_y = (turn_to_decimal(data[42:44] + data[40:42])) / 100
            acc_z = (turn_to_decimal(data[46:48] + data[44:46])) / 100
            input_tensor = torch.tensor([quat_x, quat_y, quat_z, quat_w, acc_x, acc_y, acc_z])
            return input_tensor
        else:
            return torch.empty((1, 7))

    def get_latest_data(self):
        return self.data_buffer[-1] if self.data_buffer else None


if __name__ == '__main__':

    write_CHARACTERISTIC_UUID = "91680002-1111-6666-8888-0123456789ab"
    notify_CHARACTERISTIC_UUID = "91680003-1111-6666-8888-0123456789ab"
    BUFFER_SIZE = 500
    imu_a = BLEServer("03:85:14:03:1D:1A", "0276")
    imu_b = BLEServer("03:85:14:03:95:CE", "0435")
    t_pool = []

    async def data_read():
        print('\rRecording will finish after 5s...')
        await asyncio.sleep(2)
        tensor1 = imu_a.mean_data_transition()
        tensor2 = imu_b.mean_data_transition()
        oris = tensor1[:4]
        # 将四元数转换为旋转矩阵并转置
        smpl2imu = quaternion_to_rotation_matrix(oris).view(3, 3).t()
        # 获取IMU的平均测量值用于校准
        oris = torch.stack((tensor1[:4], tensor2[:4]))
        accs = torch.stack((tensor1[-3:], tensor2[-3:]))
        # 将四元数转换为旋转矩阵
        oris = quaternion_to_rotation_matrix(oris)
        # 计算设备到骨骼的旋转矩阵
        device2bone = smpl2imu.matmul(oris).transpose(1,2).matmul(torch.eye(3))
        # 根据加速度计算全局惯性参考框架内的加速度偏移量
        acc_offsets = smpl2imu.matmul(accs.unsqueeze(-1))  # [num_imus, 3, 1]
        print('\rFinish recording.\nStart testing.')
        while imu_a.connected == 1 and imu_b.connected == 1:
            acc_list = []
            rot_list = []
            k = 0
            clock = Clock()
            while k < batch_size and imu_a.connected == 1 and imu_b.connected == 1:
                clock.tick(batch_size)
                k = k+1
                tensor1 = imu_a.data_transition()
                tensor2 = imu_b.data_transition()
                ori_raw = torch.stack((tensor1[:4], tensor2[:4]))
                acc_raw = torch.stack((tensor1[-3:], tensor2[-3:]))
                rot_raw = quaternion_to_rotation_matrix(ori_raw).view(1, 2, 3, 3)
                acc_cal = (smpl2imu.matmul(acc_raw.view(-1, 2, 3, 1)) - acc_offsets).view(1, 2, 3)
                acc_list.append(acc_cal)
                rot_cal = smpl2imu.matmul(rot_raw).matmul(device2bone)
                rot_list.append(rot_cal)
            acc = torch.stack(acc_list)
            rot = torch.stack(rot_list)

            in_put = torch.cat((acc.view(-1, 6), rot.view(-1, 18)), dim=1)
            out_put = predict(inertial_pose, in_put)
            rot = torch.squeeze(rot, dim=1)
            in_pose = rotation_matrix_to_axis_angles(rot)
            out_pose = out_put.view(batch_size, 5, 3)[:, :3, :]
            in_pose = in_pose.to('cuda:0')
            pose_result = torch.cat((in_pose, out_pose), dim=1)
            joints = poses_to_joints(pose_result)
            joints = torch.mean(joints, dim=0, keepdim=False)
            """judge(joints)"""
            d = descartes.Descartes(joints)
            d.judge()
            descartes.show()
            pass
        print('Finish.')

    def main():
        async def link():
            task_list = [
                asyncio.create_task(imu_a.main(imu_a.address, write_CHARACTERISTIC_UUID,notify_CHARACTERISTIC_UUID)),
                asyncio.create_task(imu_b.main(imu_b.address, write_CHARACTERISTIC_UUID, notify_CHARACTERISTIC_UUID))
            ]
            await asyncio.sleep(5)
            await asyncio.wait(task_list)
        asyncio.run(link())

    t_pool.append(threading.Thread(target=main))
    for t in t_pool:
        t.start()

    print('Put all imus aligned with your body reference frame (x = Left, y = Up, z = Forward) ')
    print('and turn on the butters in 5s')
    for i in range(10, 0, -1):
        print(i)
        time.sleep(1)

    for i in range(3, 0, -1):
        print(
            '\rSit S-T-A-R-I-G-H-T and be ready to keep the pose for 5 seconds ....'
            '\r The celebration will begin after %d seconds.' % i,
            end='')
        time.sleep(1)

    asyncio.run(data_read())
