import math
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary
import numpy as np


class InResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x, t):
        x = self.conv1(x)
        res = x
        x += t
        x = self.leaky_relu1(x)

        x = self.conv2(x)
        x += res  
        x = self.leaky_relu2(x)
        
        x = self.batch_norm(x)
        return x, res
    

class DownResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownResBlock, self).__init__()

        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x, t):
        x = self.down_conv(x)
        res = x
        x += t
        x = self.leaky_relu1(x)

        x = self.conv(x)
        x += res  
        x = self.leaky_relu2(x)

        x = self.batch_norm(x)
        return x, res
    

class UpResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpResBlock, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, t, skip):
        x = self.up_conv(x)
        res = x
        x += skip
        x += t
        x = self.leaky_relu1(x)

        x = self.conv(x)
        x += res
        x = self.leaky_relu2(x)

        x = self.batch_norm(x)
        return x


class ResAttnBlock(nn.Module):
    def __init__(self, stage, patch_num, dim, ced=64):
        super(ResAttnBlock, self).__init__()

        self.n = patch_num
        self.dim = dim

        down_unit = 2**(stage-1)
        self.transform = nn.Sequential(
            nn.Conv2d(ced, dim, kernel_size=down_unit, stride=down_unit),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(dim))

        self.key = nn.Linear(self.dim, self.dim)
        self.value = nn.Linear(self.dim, self.dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.batch_norm = nn.BatchNorm2d(dim)

    def forward(self, x, c):
        H, W= x.shape[2], x.shape[3]

        res = x
        x = x.permute(0,2,3,1) # [b, h, w, c]
        x = x.view(-1, self.n, self.dim) # [b, n, c], n=patch_num

        c = self.transform(c) # [b, c, h, w]
        c = c.permute(0,2,3,1) # [b, h, w, c]
        c = c.view(-1, self.n, self.dim) # [b, n, c], n=patch_num

        q = c.view(-1, self.n, self.dim) # [b, n, c]
        k = self.key(x)  # [b, n, c]
        v = self.value(x)  # [b, n, c]

        k_t = k.permute(0, 2, 1) # [b, c, n]
        attention = torch.softmax(q @ k_t, dim=-1) # [b, n, n]
        out = attention @ v # [b, n, c]

        out = out.view(-1, H, W, self.dim).permute(0, 3, 1, 2) # [b, c, h, w]
        
        out += res
        out = self.leaky_relu(out)
        out = self.batch_norm(out)
        return out
    
    
class HeadBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HeadBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.leaky_relu3 = nn.LeakyReLU(0.2)
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.head(x)
        return x


class ResUNetv2(nn.Module):
    def __init__(self, resolution=32, in_channels=1, std_cnl=32, ted=256, ced=64):
        super(ResUNetv2, self).__init__()

        self.std_cnl = std_cnl

        self.t_fc1 = nn.Linear(ted, std_cnl)
        self.t_fc2 = nn.Linear(ted, std_cnl * 2)
        self.t_fc3 = nn.Linear(ted, std_cnl * 4)
        self.t_fc4 = nn.Linear(ted, std_cnl * 8)

        self.c_embedding = nn.Conv2d(in_channels, ced, kernel_size=1)

        self.resblock_s1 = InResBlock(in_channels=in_channels, out_channels=std_cnl)
        self.resblock_s2 = DownResBlock(in_channels=std_cnl, out_channels=std_cnl*2)
        self.resblock_s3 = DownResBlock(in_channels=std_cnl*2, out_channels=std_cnl*4)
        self.attn_s3 = ResAttnBlock(stage=3, patch_num=(resolution//4)**2, dim=std_cnl*4, ced=ced)

        self.resblock_bt = DownResBlock(in_channels=std_cnl*4, out_channels=std_cnl*8)
        self.attn_bt = ResAttnBlock(stage=4, patch_num=(resolution//8)**2, dim=std_cnl*8, ced=ced)

        self.resblock_r_s3 = UpResBlock(in_channels=std_cnl*8, out_channels=std_cnl*4)
        self.attn_r_s3 = ResAttnBlock(stage=3, patch_num=(resolution//4)**2, dim=std_cnl*4, ced=ced)
        self.resblock_r_s2 = UpResBlock(in_channels=std_cnl*4, out_channels=std_cnl*2)
        self.resblock_r_s1 = UpResBlock(in_channels=std_cnl*2, out_channels=std_cnl)

        self.head = HeadBlock(in_channels=std_cnl, out_channels=in_channels)

    def forward(self, x, t, c):
        '''
        x: [b, c, h, w] = [b, 1, 32, 32]
        t: [b, time_embed_dim] = [b, 256]
        c: [b, c, h, w] = [b, 1, 32, 32]  
        '''

        # set time embedding per stage
        t1 = self.t_fc1(t).view((-1, self.std_cnl*1, 1, 1)) # [b, 32, 1, 1]
        t2 = self.t_fc2(t).view((-1, self.std_cnl*2, 1, 1)) # [b, 64, 1, 1]
        t3 = self.t_fc3(t).view((-1, self.std_cnl*4, 1, 1)) # [b, 128, 1, 1]
        t4 = self.t_fc4(t).view((-1, self.std_cnl*8, 1, 1)) # [b, 256, 1, 1]

        # condition embedding
        c = self.c_embedding(c) # [b, 64, 32, 32]

        # encoder
        x1, skip1 = self.resblock_s1(x, t1)
        x2, skip2 = self.resblock_s2(x1, t2)
        x3, skip3 = self.resblock_s3(x2, t3)
        x3_a = self.attn_s3(x3, c)

        # bottleneck
        x_bt, _ = self.resblock_bt(x3_a, t4)
        x_bt_a = self.attn_bt(x_bt, c)

        # decoder
        x = self.resblock_r_s3(x_bt_a, t3, skip3)
        x = self.attn_r_s3(x, c)
        x = self.resblock_r_s2(x, t2, skip2)
        x = self.resblock_r_s1(x, t1, skip1)

        # head
        x = self.head(x)

        return x


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    resolution=32
    in_channels=1
    std_cnl=64
    ted=256
    ced=64
    
    x = torch.ones([4, in_channels, resolution, resolution]).cuda()
    t = torch.ones([4, ted]).cuda()
    c = torch.ones([4, in_channels, resolution, resolution]).cuda()

    model = ResUNetv2(resolution, in_channels, std_cnl, ted, ced).to(device)

    print(summary(model,x,t,c))
    print('input:',x.shape,t.shape,c.shape)
    print('output:',model(x,t,c).shape)
    print('===================================')
