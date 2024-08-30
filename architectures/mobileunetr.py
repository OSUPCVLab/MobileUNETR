# MobileUNETR Architecture
# Paper: MobileUNETR: A Lightweight End-To-End Hybrid Vision Transformer For Efficient Medical Image Segmentation
# Author: Shehan Perera -- The Ohio State University
# Published in: European Conference on Computer Vision (ECCV) - Bio Image Computing 2024

import math
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from transformers import MobileViTModel


##############################################################################
# Build MobileViT / MobileViT Components
##############################################################################
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU(),
    )


def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU(),
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b p n (h d) -> b p h n d", h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b p h n d -> b p n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out


class MobileViTBlock(nn.Module):
    def __init__(
        self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.0
    ):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(
            x, "b d (h ph) (w pw) -> b (ph pw) (h w) d", ph=self.ph, pw=self.pw
        )
        x = self.transformer(x)
        x = rearrange(
            x,
            "b (ph pw) (h w) d -> b d (h ph) (w pw)",
            h=h // self.ph,
            w=w // self.pw,
            ph=self.ph,
            pw=self.pw,
        )

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(
        self,
        image_size,
        dims,
        channels,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3),
    ):
        super().__init__()
        assert len(dims) == 3, "dims must be a tuple of 3"
        assert len(depths) == 3, "depths must be a tuple of 3"

        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, last_dim = channels

        self.conv1 = conv_nxn_bn(3, init_dim, stride=2)

        self.stem = nn.ModuleList([])
        self.stem.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.stem.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))

        self.trunk = nn.ModuleList([])
        self.trunk.append(
            nn.ModuleList(
                [
                    MV2Block(channels[3], channels[4], 2, expansion),
                    MobileViTBlock(
                        dims[0],
                        depths[0],
                        channels[5],
                        kernel_size,
                        patch_size,
                        int(dims[0] * 2),
                    ),
                ]
            )
        )

        self.trunk.append(
            nn.ModuleList(
                [
                    MV2Block(channels[5], channels[6], 2, expansion),
                    MobileViTBlock(
                        dims[1],
                        depths[1],
                        channels[7],
                        kernel_size,
                        patch_size,
                        int(dims[1] * 4),
                    ),
                ]
            )
        )

        self.trunk.append(
            nn.ModuleList(
                [
                    MV2Block(channels[7], channels[8], 2, expansion),
                    MobileViTBlock(
                        dims[2],
                        depths[2],
                        channels[9],
                        kernel_size,
                        patch_size,
                        int(dims[2] * 4),
                    ),
                ]
            )
        )

        self.trunk.append(
            nn.ModuleList(
                [
                    MV2Block(channels[9], channels[10], 2, expansion),
                    MobileViTBlock(
                        dims[2],
                        depths[2],
                        channels[11],
                        kernel_size,
                        patch_size,
                        int(dims[2] * 4),
                    ),
                ]
            )
        )

        self.to_logits = conv_1x1_bn(channels[-2], last_dim)

    def forward(self, x):
        storage = []

        x = self.conv1(x)
        # print(x.shape)
        # print('\n')
        storage.append(x)

        for conv in self.stem:
            x = conv(x)
            # print(x.shape)
            # storage.append(x)

        # print('\n')
        for idx, (conv, attn) in enumerate(self.trunk):
            x = conv(x)
            x = attn(x)
            # print(x.shape)
            storage.append(x)

        x = self.to_logits(x)

        return x, storage


##############################################################################
# Build Bottleneck
##############################################################################
class MViTBottleneck(nn.Module):
    def __init__(self, dims, channels, depths, kernel_size, patch_size, expansion):
        super().__init__()
        # bottleneck downsample layer
        self.conv = MV2Block(channels[0], channels[1], 2, expansion)
        self.attn = MobileViTBlock(
            dims[0], depths[0], channels[2], kernel_size, patch_size, int(dims[0] * 4)
        )
        self.apply(self.initialize_weights)

    def forward(self, x):
        x = self.conv(x)
        x = self.attn(x)
        return x

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


##############################################################################
# Build MobileUNETR Decoders
##############################################################################
class MV2BlockUP(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # upscale
                nn.ConvTranspose2d(
                    in_channels=inp,
                    out_channels=inp,
                    kernel_size=2,
                    stride=2,
                    bias=False,
                ),
                nn.BatchNorm2d(inp),
                nn.SiLU(),
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out


class MViTDecoderxxs(nn.Module):
    def __init__(
        self,
        dims,
        channels,
        num_classes,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3),
    ):
        super().__init__()

        # reverse dims and channels
        dims = list(reversed(dims))
        channels = list(reversed(channels))

        self.trunk = nn.ModuleList([])
        # upsample layers
        self.trunk.append(
            nn.ModuleList(
                [
                    MV2BlockUP(channels[1], channels[3], 1, expansion),
                    MV2Block(channels[3] * 2, channels[3], 1, expansion=4),
                    MobileViTBlock(
                        dims[0],
                        depths[0],
                        channels[4],
                        kernel_size,
                        patch_size,
                        int(dims[0] * 2),
                    ),
                ]
            )
        )

        self.trunk.append(
            nn.ModuleList(
                [
                    MV2BlockUP(channels[4], channels[6], 1, expansion),
                    MV2Block(channels[6] * 2, channels[6], 1, expansion=4),
                    MobileViTBlock(
                        dims[1],
                        depths[1],
                        channels[6],
                        kernel_size,
                        patch_size,
                        int(dims[1] * 4),
                    ),
                ]
            )
        )

        self.trunk.append(
            nn.ModuleList(
                [
                    MV2BlockUP(channels[6], channels[8], 1, expansion),
                    MV2Block(channels[8] * 2, channels[8], 1, expansion=4),
                    MobileViTBlock(
                        dims[2],
                        depths[2],
                        channels[8],
                        kernel_size,
                        patch_size,
                        int(dims[2] * 4),
                    ),
                ]
            )
        )

        # conv upsample blocks
        self.conv_upsample_1 = nn.ModuleList([])
        self.conv_upsample_1.append(
            nn.ModuleList(
                [
                    MV2BlockUP(channels[8], channels[10], 1, expansion),
                    MV2Block(channels[10] * 2, channels[10], 1, expansion=4),
                ]
            )
        )

        self.mix = nn.Sequential(
            nn.Conv2d(
                channels[10] + channels[-2],
                channels[10],
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1,
            ),
            nn.BatchNorm2d(channels[10]),
            nn.SiLU(),
        )
        self.final = nn.Conv2d(
            channels[10], num_classes, kernel_size=1, stride=1, bias=True
        )

        self.dropout = nn.Dropout(0.1)

        self.apply(self.initialize_weights)

    def forward(self, x, storage):
        # store first feature map
        first = storage[0]

        # reverse input storage list
        storage.reverse()

        # decode
        for index, (upconv, conv, attn) in enumerate(self.trunk):
            upsample_x = upconv(x)
            x = torch.cat((upsample_x, storage[index]), dim=1)
            x = conv(x)
            x = attn(x)

        # conv only decode/upsample
        for _, (upconv, conv) in enumerate(self.conv_upsample_1):
            upsample_x = upconv(x)
            x = torch.cat((upsample_x, storage[-2]), dim=1)
            x = conv(x)

        x = F.interpolate(x, (first.shape[-2], first.shape[-1]), mode="bicubic")
        x = torch.cat((x, first), dim=1)
        x = self.mix(x)
        x = self.dropout(x)
        x = self.final(x)

        return x

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class MViTDecoders(nn.Module):
    def __init__(
        self,
        dims,
        channels,
        num_classes,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3),
    ):
        super().__init__()

        # reverse dims and channels
        dims = list(reversed(dims))
        channels = list(reversed(channels))

        self.trunk = nn.ModuleList([])
        # upsample layers
        self.trunk.append(
            nn.ModuleList(
                [
                    MV2BlockUP(channels[1], channels[3], 1, expansion),
                    MV2Block(channels[3] * 2, channels[3], 1, expansion=4),
                    MobileViTBlock(
                        dims[0],
                        depths[0],
                        channels[4],
                        kernel_size,
                        patch_size,
                        int(dims[0] * 2),
                    ),
                ]
            )
        )

        self.trunk.append(
            nn.ModuleList(
                [
                    MV2BlockUP(channels[4], channels[6], 1, expansion),
                    MV2Block(channels[6] * 2, channels[6], 1, expansion=4),
                    MobileViTBlock(
                        dims[1],
                        depths[1],
                        channels[6],
                        kernel_size,
                        patch_size,
                        int(dims[1] * 4),
                    ),
                ]
            )
        )

        self.trunk.append(
            nn.ModuleList(
                [
                    MV2BlockUP(channels[6], channels[8], 1, expansion),
                    MV2Block(channels[8] * 2, channels[8], 1, expansion=4),
                    MobileViTBlock(
                        dims[2],
                        depths[2],
                        channels[8],
                        kernel_size,
                        patch_size,
                        int(dims[2] * 4),
                    ),
                ]
            )
        )

        # conv upsample blocks
        self.conv_upsample_1 = nn.ModuleList([])
        self.conv_upsample_1.append(
            nn.ModuleList(
                [
                    MV2BlockUP(channels[8], channels[10], 1, expansion),
                    MV2Block(channels[10] * 2, channels[10], 1, expansion=4),
                ]
            )
        )

        # mix layer
        self.mix = nn.Sequential(
            nn.Conv2d(
                channels[10] + channels[-2],
                channels[10],
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1,
            ),
            nn.BatchNorm2d(channels[10]),
            nn.SiLU(),
        )
        self.final = nn.Conv2d(
            channels[10], num_classes, kernel_size=1, stride=1, bias=True
        )

        self.dropout = nn.Dropout(0.1)

        self.apply(self.initialize_weights)

    def forward(self, x, storage):
        # store first feature map
        first = storage[0]

        # reverse input storage list
        storage.reverse()

        # decode
        for index, (upconv, conv, attn) in enumerate(self.trunk):
            upsample_x = upconv(x)
            x = torch.cat((upsample_x, storage[index]), dim=1)
            x = conv(x)
            x = attn(x)

        # conv only decode/upsample
        for _, (upconv, conv) in enumerate(self.conv_upsample_1):
            upsample_x = upconv(x)
            x = torch.cat((upsample_x, storage[-2]), dim=1)
            x = conv(x)

        x = F.interpolate(x, (first.shape[-2], first.shape[-1]), mode="bicubic")
        x = torch.cat((x, first), dim=1)
        x = self.mix(x)
        x = self.dropout(x)
        x = self.final(x)

        return x

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class MViTDecoderxs(nn.Module):
    def __init__(
        self,
        dims,
        channels,
        num_classes,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3),
    ):
        super().__init__()

        # reverse dims and channels
        dims = list(reversed(dims))
        channels = list(reversed(channels))

        self.trunk = nn.ModuleList([])
        # upsample layers
        self.trunk.append(
            nn.ModuleList(
                [
                    MV2BlockUP(channels[1], channels[3], 1, expansion),
                    MV2Block(channels[3] * 2, channels[3], 1, expansion=4),
                    MobileViTBlock(
                        dims[0],
                        depths[0],
                        channels[4],
                        kernel_size,
                        patch_size,
                        int(dims[0] * 2),
                    ),
                ]
            )
        )

        self.trunk.append(
            nn.ModuleList(
                [
                    MV2BlockUP(channels[4], channels[6], 1, expansion),
                    MV2Block(channels[6] * 2, channels[6], 1, expansion=4),
                    MobileViTBlock(
                        dims[1],
                        depths[1],
                        channels[6],
                        kernel_size,
                        patch_size,
                        int(dims[1] * 4),
                    ),
                ]
            )
        )

        self.trunk.append(
            nn.ModuleList(
                [
                    MV2BlockUP(channels[6], channels[8], 1, expansion),
                    MV2Block(channels[8] * 2, channels[8], 1, expansion=4),
                    MobileViTBlock(
                        dims[2],
                        depths[2],
                        channels[8],
                        kernel_size,
                        patch_size,
                        int(dims[2] * 4),
                    ),
                ]
            )
        )

        # conv upsample blocks
        self.conv_upsample_1 = nn.ModuleList([])
        self.conv_upsample_1.append(
            nn.ModuleList(
                [
                    MV2BlockUP(channels[8], channels[10], 1, expansion),
                    MV2Block(channels[10] * 2, channels[10], 1, expansion=4),
                ]
            )
        )

        # mix layer
        self.mix = nn.Sequential(
            nn.Conv2d(
                channels[10] + channels[-2],
                channels[10],
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1,
            ),
            nn.BatchNorm2d(channels[10]),
            nn.SiLU(),
        )
        self.final = nn.Conv2d(
            channels[10], num_classes, kernel_size=1, stride=1, bias=True
        )

        self.dropout = nn.Dropout(0.1)

        self.apply(self.initialize_weights)

    def forward(self, x, storage):
        # store first feature map
        first = storage[0]

        # reverse input storage list
        storage.reverse()

        # decode
        for index, (upconv, conv, attn) in enumerate(self.trunk):
            upsample_x = upconv(x)
            x = torch.cat((upsample_x, storage[index]), dim=1)
            x = conv(x)
            x = attn(x)

        # conv only decode/upsample
        for _, (upconv, conv) in enumerate(self.conv_upsample_1):
            upsample_x = upconv(x)
            x = torch.cat((upsample_x, storage[-2]), dim=1)
            x = conv(x)

        x = F.interpolate(x, (first.shape[-2], first.shape[-1]), mode="bicubic")
        x = torch.cat((x, first), dim=1)
        x = self.mix(x)
        x = self.dropout(x)
        x = self.final(x)

        return x

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


##############################################################################
# Build Pretrained Huggingface MobileViT Implementations
##############################################################################
class MViTsEncoderPretrained(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = MobileViTModel.from_pretrained(
            "apple/mobilevit-small"
        ).base_model

    def forward(self, x):
        raw_input = x.clone()
        hidden_states = self.encoder.forward(x, output_hidden_states=True).hidden_states
        out_dict = {"raw_input": raw_input, "hidden_states": hidden_states}
        return out_dict


class MViTxsEncoderPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MobileViTModel.from_pretrained(
            "apple/mobilevit-x-small"
        ).base_model

    def forward(self, x):
        raw_input = x.clone()
        hidden_states = self.encoder.forward(x, output_hidden_states=True).hidden_states
        out_dict = {"raw_input": raw_input, "hidden_states": hidden_states}
        return out_dict


class MViTxxsEncoderPretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MobileViTModel.from_pretrained(
            "apple/mobilevit-xx-small"
        ).base_model

    def forward(self, x):
        raw_input = x.clone()
        hidden_states = self.encoder.forward(x, output_hidden_states=True).hidden_states
        out_dict = {"raw_input": raw_input, "hidden_states": hidden_states}
        return out_dict


##############################################################################
# Build Pretrained Huggingface MobileViT Implementations
##############################################################################
class MViTsSegPretrained(nn.Module):
    def __init__(
        self,
        encoder_params,
        bottleneck_params,
        decoder_params,
        image_size=512,
    ):
        super().__init__()
        self.encoder = MViTsEncoderPretrained()
        self.bottleneck = MViTBottleneck(**bottleneck_params)
        self.decoder = MViTDecoders(**decoder_params)
        self.image_size = image_size

    def forward(self, x):
        enc_dict = self.encoder(x)
        btlneck = self.bottleneck(enc_dict["hidden_states"][-1])
        dec_out = self.decoder(btlneck, list(enc_dict["hidden_states"]))
        x = F.interpolate(dec_out, self.image_size, mode="bicubic")
        return x


class MViTxsSegPretrained(nn.Module):
    def __init__(
        self,
        encoder_params,
        bottleneck_params,
        decoder_params,
        image_size=512,
    ):
        super().__init__()
        self.encoder = MViTxsEncoderPretrained()
        self.bottleneck = MViTBottleneck(**bottleneck_params)
        self.decoder = MViTDecoderxs(**decoder_params)
        self.image_size = image_size

    def forward(self, x):
        enc_dict = self.encoder(x)
        btlneck = self.bottleneck(enc_dict["hidden_states"][-1])
        dec_out = self.decoder(btlneck, list(enc_dict["hidden_states"]))
        x = F.interpolate(dec_out, self.image_size, mode="bicubic")
        return x


class MViTxxsSegPretrained(nn.Module):
    def __init__(
        self,
        encoder_params,
        bottleneck_params,
        decoder_params,
        image_size=512,
    ):
        super().__init__()
        self.encoder = MViTxxsEncoderPretrained()
        self.bottleneck = MViTBottleneck(**bottleneck_params)
        self.decoder = MViTDecoderxxs(**decoder_params)
        self.image_size = image_size

    def forward(self, x):
        enc_dict = self.encoder(x)
        btlneck = self.bottleneck(enc_dict["hidden_states"][-1])
        dec_out = self.decoder(btlneck, list(enc_dict["hidden_states"]))
        x = F.interpolate(dec_out, self.image_size, mode="bicubic")
        return x


##############################################################################
# Build/Import Model
##############################################################################
def build_mobileunetr_s(config=None, num_classes: int = 1, image_size: int = 512):

    # mobileunetr small config
    if config is None:
        config = {
            "model_parameters": {
                "encoder": None,
                "bottle_neck": {
                    "dims": [240],
                    "depths": [3],
                    "expansion": 4,
                    "kernel_size": 3,
                    "patch_size": [2, 2],
                    "channels": [160, 196, 196],
                },
                "decoder": {
                    "dims": [144, 192, 240],
                    "channels": [
                        16,
                        32,
                        64,
                        64,
                        96,
                        96,
                        128,
                        128,
                        160,
                        160,
                        196,
                        196,
                        640,
                    ],
                    "num_classes": num_classes,
                },
                "image_size": image_size,
            }
        }

    # encoder params
    model = MViTsSegPretrained(
        encoder_params=config["model_parameters"]["encoder"],
        bottleneck_params=config["model_parameters"]["bottle_neck"],
        decoder_params=config["model_parameters"]["decoder"],
        image_size=config["model_parameters"]["image_size"],
    )
    return model


def build_mobileunetr_xs(config=None, num_classes: int = 1, image_size: int = 512):

    # mobileunetr xsmall config
    if config is None:
        config = {
            "model_parameters": {
                "encoder": None,
                "bottle_neck": {
                    "dims": [144],
                    "depths": [3],
                    "expansion": 4,
                    "kernel_size": 3,
                    "patch_size": [2, 2],
                    "channels": [96, 128, 128],
                },
                "decoder": {
                    "dims": [96, 120, 144],
                    "channels": [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 128, 128, 384],
                    "num_classes": num_classes,
                },
                "image_size": image_size,
            }
        }

    # encoder params
    model = MViTxsSegPretrained(
        encoder_params=config["model_parameters"]["encoder"],
        bottleneck_params=config["model_parameters"]["bottle_neck"],
        decoder_params=config["model_parameters"]["decoder"],
        image_size=config["model_parameters"]["image_size"],
    )
    return model


def build_mobileunetr_xxs(config=None, num_classes: int = 1, image_size: int = 512):

    # mobileunetr xxsmall config
    if config is None:
        config = {
            "model_parameters": {
                "encoder": None,
                "bottle_neck": {
                    "dims": [96],
                    "depths": [3],
                    "expansion": 4,
                    "kernel_size": 3,
                    "patch_size": [2, 2],
                    "channels": [80, 96, 96],
                },
                "decoder": {
                    "dims": [64, 80, 96],
                    "channels": [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 96, 96, 320],
                    "num_classes": num_classes,
                },
                "image_size": image_size,
            }
        }

    # encoder params
    model = MViTxxsSegPretrained(
        encoder_params=config["model_parameters"]["encoder"],
        bottleneck_params=config["model_parameters"]["bottle_neck"],
        decoder_params=config["model_parameters"]["decoder"],
        image_size=config["model_parameters"]["image_size"],
    )
    return model
