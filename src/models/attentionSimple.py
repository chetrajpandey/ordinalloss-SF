import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import LinearAttentionBlockSimple, ProjectorBlock
from initialize import *




class Attn_Net(nn.Module):
    def __init__(self, im_size, num_classes, attention=None, normalize_attn=True, init='kaimingNormal'):
        super(Attn_Net, self).__init__()

        self.attention = attention
        self.count = sum(self.attention)
        self.positions = [i for i, val in enumerate(self.attention) if val]
        # print(self.positions)

        # conv blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(num_features=128, affine=True),
            nn.ReLU())
        # print(getattr(self, f"conv_block{self.positions[1]+1}")[0].out_channels)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256, affine=True),
            nn.ReLU())
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256, affine=True),
            nn.ReLU())
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(num_features=512, affine=True),
            nn.ReLU())
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(num_features=512, affine=True),
            nn.ReLU())
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(num_features=512, affine=True),
            nn.ReLU())
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Projectors & Compatibility functions
        if any(self.attention):
            print('Attention Activated: ', self.positions)
            self.projectors = nn.ModuleList([ProjectorBlock(getattr(self, f"conv_block{i+1}")[0].out_channels, 512) for i in self.positions if i < 3])
            self.attns = nn.ModuleList([LinearAttentionBlockSimple(in_features=512, normalize_attn=normalize_attn) for i in self.positions])

            # classification layer
            self.classify = nn.Linear(in_features=512 * self.count, out_features=num_classes, bias=True)
        else:
            self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)

        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")

    def forward(self, x):
        # feed forward
        l1 = self.conv_block1(x)
        x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0)  # /2
        l2 = self.conv_block2(x)
        x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0)  # /4
        l3 = self.conv_block3(x)  # /1
        x = F.max_pool2d(l3, kernel_size=2, stride=2, padding=0)  # /8
        l4 = self.conv_block4(x)  # /2
        x = F.max_pool2d(l4, kernel_size=2, stride=2, padding=0)  # /16
        l5 = self.conv_block5(x)  # /4
        x = F.max_pool2d(l5, kernel_size=2, stride=2, padding=0)  # /32
        l6 = self.conv_block6(x)
        # print(l6.size(0), l6.size(1), l6.size(2), l6.size(3))
        # g = F.adaptive_avg_pool2d(l6, (1,1)).view(l6.size(0), l6.size(1), l6.size(2), l6.size(3))
        g = F.adaptive_avg_pool2d(l6, (1, 1)).view(l6.size(0), 512)



        attn_outputs = []
        g_final = [None] * len(self.positions)
        # pay attention
        if any(self.attention):
            for i, pos in enumerate(self.positions):
                if pos < 3:
                    c = self.projectors[i](eval(f"l{pos+1}"))
                    c, g_final[i] = self.attns[i](c, g)
                else:
                    c, g_final[i] = self.attns[i](eval(f"l{pos+1}"), g)
                attn_outputs.append(c)

            # classification layer
            
            

            x = torch.cat((*g_final,), dim=1)  # batch_sizexC
            x = self.classify(x)  # batch_sizexnum_classes
        else:
            x = self.classify(torch.squeeze(g))

        return x

