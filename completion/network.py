import torch
import torch.nn as nn
import torchvision


class Block(nn.Module):
    """
    定义 UNet 架构中的基本构建块。
    包含两个卷积层，每个卷积层后跟随批归一化和 ReLU 激活函数。
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class AttentionBlock(nn.Module):
    """
    定义 Attention Gate，用于在解码器中引入注意力机制。
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        参数：
        - g: 解码器的特征（Gating Signal）
        - x: 编码器的特征（Skip Connection）
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class Encoder(nn.Module):
    """
    定义 UNet 架构的编码器部分。
    包含多个编码块和最大池化层，用于下采样输入特征图。
    """

    def __init__(self, chs=(5, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    """
    定义 UNet 架构的解码器部分，包含注意力机制。
    """

    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i + 1], kernel_size=2, stride=2) for i in range(len(chs) - 1)]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.attention_blocks = nn.ModuleList(
            [AttentionBlock(F_g=chs[i + 1], F_l=chs[i + 1], F_int=chs[i + 1] // 2) for i in range(len(chs) - 1)]
        )

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = encoder_features[i]
            enc_ftrs = self.crop(enc_ftrs, x)
            attn_enc_ftrs = self.attention_blocks[i](x, enc_ftrs)
            x = torch.cat([x, attn_enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class Generator(nn.Module):
    """
    定义完整的生成器网络架构，结合编码器、解码器和注意力机制。
    """

    def __init__(self, enc_chs=(5, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[-1], num_class, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        enc_ftrs = enc_ftrs[::-1]  # 反转特征图列表
        x = self.decoder(enc_ftrs[0], enc_ftrs[1:])
        x = self.head(x)
        # x = torchvision.transforms.CenterCrop([128, 128])(x)
        return x


class PatchDiscriminator(nn.Module):
    """
    定义 PatchGAN 判别器，用于判别生成器生成的图像的真伪。
    """

    def __init__(self, in_channels=2, features=[64, 128, 256, 512]):
        super(PatchDiscriminator, self).__init__()
        layers = []
        # 输入层，不使用批归一化
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )

        # 中间层，使用批归一化
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = feature

        # 输出层，输出特征图
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 定义损失函数
criterion_GAN = nn.BCEWithLogitsLoss()  # 对抗性损失
criterion_L1 = nn.L1Loss()  # 重建损失
lambda_L1 = 100  # L1 损失的权重

# 定义生成器和判别器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netG = Generator().to(device)
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
optimizer_G = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
