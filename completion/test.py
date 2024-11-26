import torch

from network import *
from dataloader import DataLoader
import matplotlib

from terrain.completion.network import device

matplotlib.use('TkAgg')


def main():
    # 假设您已经定义了 DataLoader 类，用于加载数据
    # from data_loader import DataLoader

    # 创建数据加载器
    data_loader = DataLoader(
        dataset_path='E:/dataset/data',  # 修改为您的数据集路径
        split=[0.8, 0.1, 0.1],
        device=device
    )

    num_epochs = 100  # 训练轮数
    batch_size = 4  # 批次大小

    for epoch in range(num_epochs):
        # 在每个 epoch 开始时，重置数据加载器的索引
        data_loader.reset_dem_idx('train')

        for iteration in range(len(data_loader.train_files)):
            # 加载一个批次的数据
            batch = data_loader.sample_batch(batch_size, file_type='train')  # 形状为 (batch_size, 4, H, W)

            # 获取输入数据和真实数据
            input_dem = batch[:, 0:1, :, :]  # 输入的 DEM 数据，已知区域的值，未知区域为 0
            mask_pred = batch[:, 1:2, :, :]  # 预测掩码
            mask_known = batch[:, 2:3, :, :]  # 已知区域的掩码
            mask_absent = batch[:, 3:4, :, :]  # 未知区域的掩码

            # 获取真实的 DEM 数据（需要构造真实数据，假设为真实的 DEM）
            real_dem = input_dem.clone()
            real_dem[mask_absent.bool()] = batch[:, 0:1, :, :][mask_absent.bool()]  # 真实的 DEM 数据

            # ---------------------
            # 训练生成器
            # ---------------------
            optimizer_G.zero_grad()

            # 将输入的 DEM 数据和掩码拼接，作为生成器的输入
            gen_input = torch.cat([input_dem, mask_pred], dim=1)  # 输入通道数为 2

            # 生成假图像
            fake_dem = netG(gen_input)

            # 判别器对生成的假图像的预测
            fake_data = torch.cat([input_dem, fake_dem], dim=1)  # 判别器的输入
            pred_fake = netD(fake_data)

            # 生成器的对抗性损失
            valid_label = torch.ones_like(pred_fake, device=device)
            loss_g_gan = criterion_GAN(pred_fake, valid_label)

            # 生成器的 L1 损失（与真实图像的差异）
            loss_g_l1 = criterion_L1(fake_dem, real_dem) * lambda_L1

            # 总的生成器损失
            loss_G = loss_g_gan + loss_g_l1

            # 反向传播和优化
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            # 训练判别器
            # ---------------------
            optimizer_D.zero_grad()

            # 判别器对真实图像的预测
            real_data = torch.cat([input_dem, real_dem], dim=1)
            pred_real = netD(real_data)
            loss_d_real = criterion_GAN(pred_real, valid_label)

            # 判别器对假图像的预测
            fake_data = torch.cat([input_dem, fake_dem.detach()], dim=1)
            pred_fake = netD(fake_data)
            fake_label = torch.zeros_like(pred_fake, device=device)
            loss_d_fake = criterion_GAN(pred_fake, fake_label)

            # 总的判别器损失
            loss_d = (loss_d_real + loss_d_fake) / 2

            # 反向传播和优化
            loss_d.backward()
            optimizer_D.step()

            # 打印损失等信息
            if iteration % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Iteration [{iteration}/{len(data_loader.train_files)}]: "
                      f"Loss_D: {loss_d.item():.4f}, Loss_G: {loss_G.item():.4f}")


def test1():
    # 创建数据加载器
    data_loader = DataLoader(
        dataset_path='E:/dataset/data',  # 修改为您的数据集路径
        split=[0.8, 0.1, 0.1],
        device=device
    )

    # 采样一个批次的数据
    batch_size = 2
    batch = data_loader.sample_batch(batch_size, file_type='train')  # 形状为 (batch_size, 4, H, W)

    # 获取输入数据和掩码
    input_dem = batch[:, 0:1, :, :]  # 输入的 DEM 数据
    mask_pred = batch[:, 1:2, :, :]  # 预测掩码

    # 将输入的 DEM 数据和掩码拼接，作为生成器的输入
    gen_input = torch.cat([input_dem, mask_pred], dim=1)  # 输入通道数为 2

    # 前向传播生成器
    net_g = Generator().to(device)
    netG.apply(weights_init)

    generated_dem = netG(gen_input.to(device))
    print(f"Generated DEM shape: {generated_dem.shape}")  # 应该是 (batch_size, 1, H, W)

    # 判别器前向传播
    net_d = PatchDiscriminator(in_channels=2).to(device)
    net_d.apply(weights_init)

    # 拼接输入数据和生成器的输出
    fake_data = torch.cat([input_dem, generated_dem.detach().cpu()], dim=1)

    pred = net_d(fake_data.to(device))
    print(f"Discriminator output shape: {pred.shape}")  # 应该是 (batch_size, 1, H', W')

    # 验证前向传播是否正常
    assert generated_dem.shape == input_dem.shape, "Generated DEM shape mismatch"
    print("Model forward pass successful.")


def test2():
    # 创建 DataLoader 实例
    data_loader = DataLoader(
        dataset_path='D:/Doc/paper/data/N030E080_N035E085',  # 修改为您的数据集路径
        # dataset_path='D:/Doc/paper/data/N030E080_N035E085',  #
        split=[0.8, 0.1, 0.1],
        device=torch.device('cpu')  # 或者 'cuda' 如果有 GPU
    )

    # 采样一个批次
    batch_size = 4
    batch = data_loader.sample_batch(batch_size, file_type='train')

    # 检查批次的形状
    print(f'Batch shape: {batch.shape}')  # 应该是 (batch_size, 4, H, W)

    # 可视化第一个样本的输入数据和掩码
    import matplotlib.pyplot as plt

    sample = batch[0]  # 取出第一个样本，形状为 (4, H, W)

    input_data = sample[0].cpu().numpy()
    mask_pred = sample[1].cpu().numpy()
    mask_known = sample[2].cpu().numpy()
    mask_absent = sample[3].cpu().numpy()

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.title('Input Data')
    plt.imshow(input_data, cmap='terrain')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title('Prediction Mask')
    plt.imshow(mask_pred, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title('Known Mask')
    plt.imshow(mask_known, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title('Absent Mask')
    plt.imshow(mask_absent, cmap='gray')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test2()
    # main()
