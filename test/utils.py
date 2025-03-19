import os
import numpy as np
import tifffile
import torch
import torch.nn as nn
from scipy.ndimage.filters import median_filter
from skimage.filters import threshold_otsu


def read_multi(path):
    # 读取每张图像，并将其转换为浮动类型并扩展维度
    data_list = []
    # 获取文件夹中所有图像文件的路径，假设文件扩展名为 '.tif'
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif')]
    for image_path in image_paths:
        data = tifffile.imread(image_path)  # 读取图像
        data_tensor = torch.tensor(data, dtype=torch.float32)  # 转换为Tensor
        data_tensor = torch.unsqueeze(torch.unsqueeze(data_tensor, 0), 0)  # 扩展维度为 (1, 1, H, W)
        data_list.append(data_tensor)
    
    # 将多个图像堆叠成一个批次
    data_batch = torch.cat(data_list, dim=0)  # 形状为 (N, 1, 1, H, W)，N 是图像数量
    return data_batch


class Encoder3D(nn.Module):
    def __init__(self, b, input_shape):
        super(Encoder3D, self).__init__()
        self.b = b  # 输出大小的目标尺寸
        self.input_shape = input_shape
        
        # 定义3D卷积层
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool3d(1, 1)  # 保持尺寸不变
        self.conv5 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool3d(1, 1)  # 同样保持尺寸不变

        # 使用假输入计算卷积输出尺寸
        self.dummy_input = torch.randn(1, 1, input_shape, input_shape, input_shape)  # 三维输入
        conv_out = self._get_conv_output(self.dummy_input)

        # Fully connected 层
        self.fc = nn.Linear(conv_out, 512 * b * b * b)

    def _get_conv_output(self, input_tensor):
        x = self.pool1(torch.relu(self.conv1(input_tensor)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.pool4(torch.relu(self.conv4(x)))
        x = self.pool5(torch.relu(self.conv5(x)))
        return x.numel()  # 获取展平后的尺寸

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.pool4(torch.relu(self.conv4(x)))
        x = self.pool5(torch.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        x = x.view(x.size(0), 512, self.b, self.b, self.b)  # 输出形状 [batch, 512, b, b, b]
        return x


class Encoder(nn.Module):
    def __init__(self, b, input_shape):
        super(Encoder, self).__init__()
        self.b = b  # Set the desired output size (b)
        self.input_shape = input_shape

        # Define the layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(1, 1)  # No actual pooling here, just keeping the size
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(1, 1)  # Same as above, no actual pooling here

        self.dummy_input = torch.randn(1, 1, input_shape,
                                       input_shape)  # 1x96x96 input as a dummy for calculating output size
        conv_out = self._get_conv_output(self.dummy_input)

        # Adjusting the fully connected layer for the output size
        self.fc = nn.Linear(conv_out, 512 * b * b * b)  # Output size will be 512 x b^3

    def _get_conv_output(self, input_tensor):
        # Pass through the convolution layers to get the output shape
        x = self.pool1(torch.relu(self.conv1(input_tensor)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.pool4(torch.relu(self.conv4(x)))
        x = self.pool5(torch.relu(self.conv5(x)))
        return x.numel()  # Number of elements in the tensor (flattened size)

    def forward(self, x):
        # Forward pass through convolution layers
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.pool4(torch.relu(self.conv4(x)))
        x = self.pool5(torch.relu(self.conv5(x)))

        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)  # Flatten

        # Apply fully connected layer to produce the output
        x = self.fc(x)

        # Reshape to [n, 512, b, b, b]
        x = x.view(x.size(0), 512, self.b, self.b, self.b)
        return x


def save_tiff(tensor, path):
    # 将输入张量转移到CPU并处理为NumPy数组
    tensor = tensor.cpu()

    # 归一化到[0, 255]范围并转换为字节类型的NumPy数组
    img = tensor.mul(0.5).add(0.5).mul(255).byte().numpy()  # 将[-1, 1]范围的图像映射到[0, 255]

    # 对于3D图像，提取第一个通道，并使用Otsu阈值化
    for i in range(tensor.shape[0]):  # 处理批量图像
        threshold_global_otsu = threshold_otsu(img[i, 0, :, :, :])  # 对三维图像进行阈值化
        segmented_image = (img[i, 0, :, :, :] >= threshold_global_otsu).astype(np.uint8)  # 阈值处理

        # 中值滤波处理，去除噪点
        segmented_image = median_filter(segmented_image, size=(5, 5, 5))  # 对3D图像进行中值滤波，大小可根据需要调整

        # 确保图像为二值图像，像素值为0或255
        segmented_image = np.clip(segmented_image, 0, 1) * 255  # 保证像素值为0或255

        # 保存为TIFF文件
        tifffile.imwrite(f"{path}/batch_{i}.tif", segmented_image.astype(np.uint8))  # 将二值图像保存为TIFF文件
