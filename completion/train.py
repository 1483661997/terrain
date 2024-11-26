import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

# 确保 dataloader.py 和 model.py 在同一目录下，或者正确导入
from dataloader import DataLoader
from network import Generator, PatchDiscriminator

def main():
    # 检查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 设置随机种子以确保可重复性
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # 定义训练参数
    num_epochs = 100        # 训练的总轮数
    batch_size = 16         # 批次大小
    learning_rate = 2e-4    # 学习率
    lambda_L1 = 100         # L1 损失的权重

    # 定义损失函数
    criterion_GAN = nn.BCEWithLogitsLoss()  # 对抗性损失
    criterion_L1 = nn.L1Loss()              # L1 损失

    # 初始化数据加载器
    dataset_path = 'D:/Doc/paper/data/N030E080_N035E085'  # 请根据实际数据集路径修改
    data_loader = DataLoader(dataset_path=dataset_path, device=device)

    # 初始化生成器和判别器
    # netG = Generator().to(device)
    netG = Generator(enc_chs=(1, 64, 128, 256, 512, 1024)).to(device)

    netD = PatchDiscriminator(in_channels=2).to(device)

    # 初始化权重
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # 定义优化器
    # optimizer_G = torch.optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    # optimizer_D = torch.optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=1e-3, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # 学习率调度器（可选）
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1)

    # 日志记录器
    writer = SummaryWriter('runs/experiment1')

    # 开始训练
    for epoch in range(num_epochs):
        # 重置数据集索引
        data_loader.reset_dem_idx(file_type='train')
        for i in range(len(data_loader.train_files)):
            # 加载一个批次的数据
            # 获取批次数据，形状为 [batch_size, 4, H, W]
            batch = data_loader.sample_batch(batch_size=batch_size, file_type='train')

            # 添加一个额外的全零通道
            extra_channel = torch.zeros_like(batch[:, 0:1, :, :])

            # 将 4 个通道和额外的通道组合起来，形成 5 个通道
            # input_data = torch.cat(
            #     [batch[:, 0:1, :, :], batch[:, 1:2, :, :], batch[:, 2:3, :, :], batch[:, 3:4, :, :], extra_channel],
            #     dim=1).to(device)
            input_data = batch[:, 0:1, :, :].to(device)  # 形状应为 [16, 1, 384, 384]

            # batch = data_loader.sample_batch(batch_size=batch_size, file_type='train')
            # input_data = batch[:, 0:1, :, :].to(device)    # 输入数据
            mask_pred = batch[:, 1:2, :, :].to(device)     # 预测掩码
            mask_known = batch[:, 2:3, :, :].to(device)    # 已知区域掩码
            mask_absent = batch[:, 3:4, :, :].to(device)   # 未知区域掩码
            # print(f"batch shape: {batch.shape}")
            # print(f"input_data shape: {input_data.shape}")
            # print(f"mask_pred shape: {mask_pred.shape}")
            # print(f"mask_known shape: {mask_known.shape}")
            # print(f"mask_absent shape: {mask_absent.shape}")
            # ---------------------
            #  训练判别器
            # ---------------------
            netD.train()
            optimizer_D.zero_grad()

            # 真实数据
            real_input = input_data * mask_known  # 形状为 [16, 1, 384, 384]
            real_pair = torch.cat((real_input, input_data), dim=1)  # 形状为 [16, 2, 384, 384]

            # print(f"real_pair shape: {real_pair.shape}")  # 添加这行
            real_output = netD(real_pair)





            # print(f"input_data shape: {input_data.shape}")
            # print(f"real_input shape: {real_input.shape}")

            real_output = netD(real_pair)
            # real_label = torch.ones_like(real_output, device=device)
            real_label = torch.full_like(real_output, 0.9, device=device)  # 真实标签从 1 调整为 0.9

            # 生成数据
            fake_output = netG(real_input)
            fake_pair = torch.cat((real_input, fake_output.detach()), dim=1)
            fake_output_D = netD(fake_pair)
            # fake_label = torch.zeros_like(fake_output_D, device=device)
            fake_label = torch.full_like(fake_output_D, 0.1, device=device)  # 虚假标签从 0 调整为 0.1

            # 判别器损失
            loss_D_real = criterion_GAN(real_output, real_label)
            loss_D_fake = criterion_GAN(fake_output_D, fake_label)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  训练生成器
            # -----------------
            netG.train()
            optimizer_G.zero_grad()

            # 重新计算假数据的判别器输出
            fake_pair = torch.cat((real_input, fake_output), dim=1)
            fake_output_D = netD(fake_pair)

            # 生成器损失
            loss_G_GAN = criterion_GAN(fake_output_D, real_label)
            loss_G_L1 = criterion_L1(fake_output * mask_pred, input_data * mask_pred) * lambda_L1
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            # 打印损失
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader.train_files)}], "
                      f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

                # 记录损失到 TensorBoard
                global_step = epoch * len(data_loader.train_files) + i
                writer.add_scalar('Loss/Discriminator', loss_D.item(), global_step)
                writer.add_scalar('Loss/Generator', loss_G.item(), global_step)

        # 更新学习率
        scheduler_G.step()
        scheduler_D.step()

        # 保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(netG.state_dict(), f'generator_epoch_{epoch+1}.pth')
            torch.save(netD.state_dict(), f'discriminator_epoch_{epoch+1}.pth')

        # 在验证集上评估模型
        netG.eval()
        with torch.no_grad():
            data_loader.reset_dem_idx(file_type='valid')
            val_batch = data_loader.sample_batch(batch_size=batch_size, file_type='valid')
            input_data_val = val_batch[:, 0:1, :, :].to(device)
            mask_pred_val = val_batch[:, 1:2, :, :].to(device)
            mask_known_val = val_batch[:, 2:3, :, :].to(device)
            mask_absent_val = val_batch[:, 3:4, :, :].to(device)

            real_input_val = input_data_val * mask_known_val
            fake_output_val = netG(real_input_val)
            fake_pair_val = torch.cat((real_input_val, fake_output_val), dim=1)
            fake_output_D_val = netD(fake_pair_val)

            loss_G_GAN_val = criterion_GAN(fake_output_D_val, torch.ones_like(fake_output_D_val, device=device))
            loss_G_L1_val = criterion_L1(fake_output_val * mask_pred_val, input_data_val * mask_pred_val) * lambda_L1
            loss_G_val = loss_G_GAN_val + loss_G_L1_val

            print(f"Validation Loss G: {loss_G_val.item():.4f}")

            # 记录验证损失
            writer.add_scalar('Loss/Generator_Validation', loss_G_val.item(), epoch + 1)

            # 可视化生成结果
            sample_input = real_input_val[:4]
            sample_output = fake_output_val[:4]
            sample_target = input_data_val[:4]

            # 将输入、生成结果和目标拼接在一起
            img_grid = torchvision.utils.make_grid(torch.cat((sample_input, sample_output, sample_target), dim=0), nrow=4, normalize=True)
            writer.add_image('Validation Results', img_grid, epoch + 1)

        netG.train()

    # 训练结束，保存最终模型
    torch.save(netG.state_dict(), 'generator_final.pth')
    torch.save(netD.state_dict(), 'discriminator_final.pth')
    writer.close()

if __name__ == '__main__':
    main()
