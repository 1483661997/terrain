import numpy as np
import rasterio
import glob
import cv2
import random
<<<<<<< HEAD
=======
from einops import rearrange, pack, repeat
>>>>>>> ed712452d0139a8c575a6d677937bade9e1aea21
import torch
import torchvision


class DataLoader():
    """
    定义一个用于管理加载和处理 DEM（数字高程模型）数据的 DataLoader 类。
    """

    def __init__(self,
<<<<<<< HEAD
                 dataset_path='E:/dataset/data',
=======
                 dataset_path='E:/dataset/S1A_IW_GRDH_1SDV_20240928T101343_20240928T101408_055864_06D3E6_AFE3.SAFE/measurement',
>>>>>>> ed712452d0139a8c575a6d677937bade9e1aea21
                 split=[0.8, 0.1, 0.1], device=torch.device('cpu')):
        """
        初始化 DataLoader 对象。

        参数：
        - dataset_path：数据集的路径，默认为指定路径。
        - split：数据集划分比例，默认为训练集、验证集、测试集 8:1:1。
        - device：计算设备，默认为 CPU。
        """
        self.device = device  # 设置计算设备

        # 检查数据集划分比例之和是否为 1.0
<<<<<<< HEAD
        if sum(split) != 1.0:
            raise ValueError('Error: Ratio of split not equal to 1.0!')

        # 获取指定路径下的所有 .tif 文件
        files = glob.glob(dataset_path + '/*.tiff')
        if len(files) == 0:
            raise FileNotFoundError('Error: No .tif files found in the specified directory!')

        N_files = len(files)  # 文件总数
        N_train_files = int(N_files * split[0])  # 训练集文件数
        N_valid_files = int(N_files * split[1])  # 验证集文件数

        # 随机打乱文件列表
        random.shuffle(files)
=======
        if split[0] + split[1] + split[2] != 1.0:
            print('Error: Ratio of split not equal to 1.0!')

        # 获取指定路径下的所有文件
        files = glob.glob(dataset_path + '/*')
        if len(files) == 0:
            print('Error: No files found!')
        elif files[0][-3:] != 'tif':
            print('Error: Non-tif file found in directory!')

        N_files = len(files)  # 文件总数
        N_train_files = round(N_files * split[0])  # 训练集文件数
        N_valid_files = round(N_files * split[1])  # 验证集文件数
>>>>>>> ed712452d0139a8c575a6d677937bade9e1aea21

        # 划分文件列表为训练集、验证集和测试集
        self.train_files = files[:N_train_files]
        self.valid_files = files[N_train_files:N_train_files + N_valid_files]
        self.test_files = files[N_train_files + N_valid_files:]

<<<<<<< HEAD
        self.dem_size = 512  # DEM 数据缩放后的尺寸
        self.gt_size = 128  # 目标（ground truth）块的尺寸
=======
        self.spo2 = 2 ** 13  # 定义裁剪尺寸的幂值，用于裁剪 DEM 数据
        self.dem = None  # 当前加载的 DEM 数据
>>>>>>> ed712452d0139a8c575a6d677937bade9e1aea21

        # 文件索引，用于遍历数据集
        self.train_file_idx = 0
        self.valid_file_idx = 0
        self.test_file_idx = 0

<<<<<<< HEAD
=======
        self.dem_size = 512  # DEM 数据缩放后的尺寸
        self.gt_size = 128  # 目标（ground truth）块的尺寸

        # 用于缓存已加载的 DEM 数据
        self.train_dem_list = []
        self.valid_dem_list = []
        self.test_dem_list = []

>>>>>>> ed712452d0139a8c575a6d677937bade9e1aea21
    def load_dem(self, file_type='train'):
        """
        加载指定类型的数据集（训练、验证、测试）中的一个 DEM 文件。

        参数：
        - file_type：要加载的数据集类型，默认为 'train'。
        """
<<<<<<< HEAD
        # 根据文件类型选择文件列表和索引
        if file_type == 'train':
            if self.train_file_idx >= len(self.train_files):
                self.train_file_idx = 0  # 如果索引超出范围，重置为0
            file = self.train_files[self.train_file_idx]
            self.train_file_idx += 1
        elif file_type == 'valid':
            if self.valid_file_idx >= len(self.valid_files):
                self.valid_file_idx = 0
            file = self.valid_files[self.valid_file_idx]
            self.valid_file_idx += 1
        elif file_type == 'test':
            if self.test_file_idx >= len(self.test_files):
                self.test_file_idx = 0
            file = self.test_files[self.test_file_idx]
            self.test_file_idx += 1
        else:
            raise ValueError('Error: Invalid file type! Valid types are: train/valid/test.')

        # 使用 rasterio 读取 DEM 文件
        # with rasterio.open(file) as src:
        #     data = src.read(1)  # 读取第一波段的数据
        #
        # # 处理无效值（如 NaN、无数据值）
        # data = np.nan_to_num(data, nan=0.0)
        #
        # # 将 DEM 数据缩放到指定尺寸（self.dem_size x self.dem_size）
        # data = cv2.resize(data, (self.dem_size, self.dem_size), interpolation=cv2.INTER_CUBIC)
        #
        # # 数据归一化，可以根据数据的实际情况调整
        # data_mean = np.mean(data)
        # data_std = np.std(data)
        # data = (data - data_mean) / (data_std + 1e-8)  # 加上一个小值，防止除以零
        #
        # self.dem = data  # 保存处理后的 DEM 数据

        with rasterio.open(file) as src:
            data = src.read(1)  # 读取第一波段的数据

        # 处理无效值（如 NaN、无数据值）
        data = np.nan_to_num(data, nan=0.0)

        # **添加数据类型转换**
        data = data.astype(np.float32)  # 将数据类型转换为 float32

        # 将 DEM 数据缩放到指定尺寸（self.dem_size x self.dem_size）
        data = cv2.resize(data, (self.dem_size, self.dem_size), interpolation=cv2.INTER_CUBIC)

        # 数据归一化，可以根据数据的实际情况调整
        data_mean = np.mean(data)
        data_std = np.std(data)
        data = (data - data_mean) / (data_std + 1e-8)  # 加上一个小值，防止除以零

        self.dem = data  # 保存处理后的 DEM 数据

    def sample(self):
        """
        从当前加载的 DEM 数据中采样一个处理过的图像块。

        返回值：
        - sample：包含地形数据、预测掩码、已知点掩码、未知点掩码的张量。
        """
        if self.dem is None:
            raise ValueError('Error: DEM not loaded yet!')

        # 数据增强：随机翻转和旋转
        dem = self.dem.copy()
        if random.randint(0, 1):
            dem = np.flip(dem, axis=1)  # 水平翻转
        if random.randint(0, 1):
            dem = np.flip(dem, axis=0)  # 垂直翻转
        dem = np.rot90(dem, random.randint(0, 3))  # 随机旋转 0、90、180、270 度

        # 随机选择一个起始位置，从 DEM 中裁剪一个 3 x gt_size 的图像块
        max_offset = self.dem_size - 3 * self.gt_size
        if max_offset <= 0:
            raise ValueError('DEM size is too small for the given gt_size.')
        r = random.randint(0, max_offset)
        c = random.randint(0, max_offset)
        patch = dem[r:r + 3 * self.gt_size, c:c + 3 * self.gt_size]

        # 初始化掩码，用于标记已知和未知区域
        mask = np.zeros((3 * self.gt_size, 3 * self.gt_size))
        for i in range(3):
            for j in range(3):
                if random.randint(0, 1):
                    # 随机选择一些子块作为已知区域
                    r_sub = i * self.gt_size
                    c_sub = j * self.gt_size
                    mask[r_sub:r_sub + self.gt_size, c_sub:c_sub + self.gt_size] = 1.0

        # 将中心块标记为未知区域（需要模型进行预测）
        mask[self.gt_size:2 * self.gt_size, self.gt_size:2 * self.gt_size] = 0.0

        # 将已知区域的值保留，未知区域的值设为 0
        input_data = patch * mask

        # 创建预测掩码，仅在中心块位置为 1，其余为 0
        mask_pred = np.zeros((3 * self.gt_size, 3 * self.gt_size))
        mask_pred[self.gt_size:2 * self.gt_size, self.gt_size:2 * self.gt_size] = 1.0

        # 计算未知区域的掩码
        mask_absent = 1.0 - mask

        # 打包为一个包含多个通道的数组
        sample = np.stack([input_data, mask_pred, mask, mask_absent], axis=0)  # 形状为 (4, H, W)

        return sample  # 返回打包后的张量

    def sample_batch(self, batch_size, file_type='train'):
        """
        采样一个包含多个图像块的批次。

        参数：
        - batch_size：批次的大小。
        - file_type：数据集类型，默认为 'train'。

        返回值：
        - batch：包含批次数据的张量，形状为 (batch_size, 4, h, w)。
        """
        # 加载一个 DEM 文件
        self.load_dem(file_type)

        # 采样指定数量的图像块
        batch = [self.sample() for _ in range(batch_size)]

        # 将批次数据打包为张量
        batch = np.stack(batch, axis=0)  # 形状为 (batch_size, 4, H, W)
        batch = torch.from_numpy(batch).float().to(self.device)

        return batch
=======
        # 检查是否已经缓存了 DEM 数据，如果有则直接加载
        if file_type == 'train' and self.train_file_idx < len(self.train_dem_list):
            self.dem = self.train_dem_list[self.train_file_idx]
            self.train_file_idx = (self.train_file_idx + 1) % len(self.train_files)
            return
        elif file_type == 'valid' and self.valid_file_idx < len(self.valid_dem_list):
            self.dem = self.valid_dem_list[self.valid_file_idx]
            self.valid_file_idx = (self.valid_file_idx + 1) % len(self.valid_files)
            return
        elif file_type == 'test' and self.test_file_idx < len(self.test_dem_list):
            self.dem = self.test_dem_list[self.test_file_idx]
            self.test_file_idx = (self.test_file_idx + 1) % len(self.test_files)
            return

        # 根据文件类型选择文件列表和索引
        if file_type == 'train':
            file = self.train_files[self.train_file_idx]
            self.train_file_idx = (self.train_file_idx + 1) % len(self.train_files)
        elif file_type == 'valid':
            file = self.valid_files[self.valid_file_idx]
            self.valid_file_idx = (self.valid_file_idx + 1) % len(self.valid_files)
        elif file_type == 'test':
            file = self.test_files[self.test_file_idx]
            self.test_file_idx = (self.test_file_idx + 1) % len(self.test_files)
        else:
            print('Error: Invalid file type! Valid types are: train/valid/test.')
            return

        # 使用 rasterio 读取 DEM 文件，并裁剪到指定尺寸
        data = rasterio.open(file).read(1)[:self.spo2, :self.spo2]
        # 将 DEM 数据缩放到指定尺寸（self.dem_size x self.dem_size）
        data = cv2.resize(data, (self.dem_size, self.dem_size), interpolation=cv2.INTER_CUBIC)
        # 将数据类型转换为 float64，以便进行减法运算
        data = data.astype('float64')
        # 减去数据的均值，进行归一化
        data -= np.mean(data)
        # 除以标准差（假设为 40.0），进一步归一化
        data /= 40.0  # 40.0 是数据集的标准差

        self.dem = data  # 保存处理后的 DEM 数据

        # 将处理后的 DEM 数据缓存到对应的列表中
        if file_type == 'train':
            self.train_dem_list.append(self.dem)
        elif file_type == 'valid':
            self.valid_dem_list.append(self.dem)
        elif file_type == 'test':
            self.test_dem_list.append(self.dem)
>>>>>>> ed712452d0139a8c575a6d677937bade9e1aea21

    def reset_dem_idx(self, file_type='valid'):
        """
        重置指定类型的数据集（训练、验证、测试）的文件索引。

        参数：
        - file_type：要重置的文件类型，默认为 'valid'。
        """
        if file_type == 'train':
            self.train_file_idx = 0
        elif file_type == 'valid':
            self.valid_file_idx = 0
        elif file_type == 'test':
            self.test_file_idx = 0
        else:
<<<<<<< HEAD
            raise ValueError('Error: Invalid file type in reset! Valid types are: train/valid/test.')
=======
            print('Error: Invalid file type in reset! Valid types are: train/valid/test.')

    def sample(self):
        """
        从当前加载的 DEM 数据中采样一个处理过的图像块。

        返回值：
        - im：包含地形数据、预测掩码、已知点掩码、未知点掩码的张量。
        """
        if self.dem is None:
            print('Error: DEM not loaded yet!')
            return

        # 数据增强：随机翻转和旋转
        if random.randint(0, 1) == 1:
            self.dem = np.flip(self.dem, axis=1)  # 水平翻转
        self.dem = np.rot90(self.dem, random.randint(0, 3))  # 随机旋转 0、90、180、270 度

        # 随机选择一个起始位置，从 DEM 中裁剪一个 3x(gt_size) 的图像块
        r = random.randint(0, self.dem_size - 3 * self.gt_size)
        c = random.randint(0, self.dem_size - 3 * self.gt_size)
        patch = self.dem[r:r + 3 * self.gt_size, c:c + 3 * self.gt_size]

        # 初始化掩码，用于标记已知和未知区域
        mask = np.zeros((3 * self.gt_size, 3 * self.gt_size))
        for i in range(3):
            for j in range(3):
                if random.randint(0, 1) == 1:
                    # 随机选择一些子块作为已知区域
                    r_sub = i * self.gt_size
                    c_sub = j * self.gt_size
                    mask[r_sub:r_sub + self.gt_size, c_sub:c_sub + self.gt_size] = 1.0
        # 将中心块标记为已知区域
        mask[self.gt_size:2 * self.gt_size, self.gt_size:2 * self.gt_size] = 1.0
        # 将已知区域的值保留，未知区域的值设为 0
        patch = patch * mask
        # 计算未知区域的掩码
        mask_absent = 1.0 - mask
        # 将中心块的掩码设为 0，以便模型进行预测
        mask[self.gt_size:2 * self.gt_size, self.gt_size:2 * self.gt_size] = 0.0

        # 创建预测掩码，仅在中心块位置为 1，其余为 0
        mask_pred = np.zeros((3 * self.gt_size, 3 * self.gt_size))
        mask_pred[self.gt_size:2 * self.gt_size, self.gt_size:2 * self.gt_size] = 1.0

        # 使用 einops 将数据打包为指定格式的张量
        im, _ = pack([patch, mask_pred, mask, mask_absent], '* h w')
        return im  # 返回打包后的张量

    def sample_batch(self, batch_size):
        """
        采样一个包含多个图像块的批次。

        参数：
        - batch_size：批次的大小。

        返回值：
        - batch：包含批次数据的张量，形状为 (batch_size, 4, h, w)。
        """
        # 采样指定数量的图像块
        batch = [self.sample() for _ in range(batch_size)]
        # 将批次数据打包为张量
        batch, _ = pack(batch, '* c h w')
        # 转换为 PyTorch 张量，并移动到指定设备
        batch = torch.Tensor(batch).to(self.device)
        return batch
>>>>>>> ed712452d0139a8c575a6d677937bade9e1aea21
