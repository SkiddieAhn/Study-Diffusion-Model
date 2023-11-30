import math
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary
import numpy as np


class MLP(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
    
    def forward(self, x):
        return self.mlp(x)
    

class MultiheadCrossAttention(nn.Module):
    def __init__(self, attn_dim: int = 512, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.head_dim = int(attn_dim / num_heads)
        self.query = nn.Linear(attn_dim, attn_dim)
        self.key = nn.Linear(attn_dim, attn_dim)
        self.value = nn.Linear(attn_dim, attn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c):
        batch_size = x.size(0)
        q = self.query(c) # [b, n, d]
        k = self.key(x)
        v = self.value(x)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3) # [b, h, n, d/h]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,3,1) # k.t
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)

        scaling = self.head_dim ** (1/2)
        attention = torch.softmax(q @ k / scaling, dim=-1) # [b, h, n, n]
        x = self.dropout(attention) @ v # [b, h, n, d/h]
        x = x.permute(0,2,1,3).reshape(batch_size, -1, self.attn_dim) # [b, n, d]

        return x
    

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 attn_dim: int =512,
                 drop_p = 0.,
                 num_heads: int = 8,
                 attn_dropout: float = 0,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(attn_dim)
        self.cross_attn = MultiheadCrossAttention(attn_dim, num_heads, attn_dropout)
        self.mlp = MLP(attn_dim, expansion=forward_expansion, drop_p=forward_drop_p)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, c):
        res1 = x
        x = self.layer_norm(x)
        x = self.cross_attn(x, c)
        x += res1

        res2 = x
        x = self.layer_norm(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x += res2

        return x
    

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2),
                                    double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x
    

class down_attn(nn.Module):
    def __init__(self, patch_num, attn_dim, cond_embed_dim, heads, blocks):
        super().__init__()
        self.n = patch_num
        self.attn_dim = attn_dim
        self.cond_embed_dim = cond_embed_dim
        self.blocks = blocks

        self.maxpool = nn.MaxPool2d(2)
        self.mlp = nn.Sequential(
            nn.Linear(self.cond_embed_dim, self.cond_embed_dim//4),
            nn.Linear(self.cond_embed_dim//4, self.n*self.attn_dim)
        )
        self.attn = nn.ModuleList([TransformerEncoderBlock(attn_dim=attn_dim, num_heads=heads) for _ in range(self.blocks)])

    def positional_encoding(self, pos_num, d_model):
        posses = np.arange(pos_num)

        i = np.arange(d_model)//2
        exponent = 2*i/d_model
        pos_emb = posses[:, np.newaxis] / np.power(10000, exponent)

        pos_emb[:, 0::2] = np.sin(pos_emb[:, 0::2])
        pos_emb[:, 1::2] = np.cos(pos_emb[:, 1::2])

        pos_emb = torch.from_numpy(pos_emb).float().to("cuda")

        return pos_emb

    def forward(self, x, c):
        '''
        x: [b, attn_dim, 8, 8]
        c: [b, cond_embed_dim]
        '''

        x = self.maxpool(x) # [b, 512, 4, 4]
        x = x.permute(0,2,3,1) # [b, 4, 4, 512]
        x = x.view(-1, self.n, self.attn_dim) # [b, 4*4, 512]
        x += self.positional_encoding(pos_num=self.n, d_model=self.attn_dim)

        c = self.mlp(c) # [b, 16*512]
        c = c.view(-1, self.n, self.attn_dim) # [b, 16, 512]

        for i in range(self.blocks):
            x = self.attn[i](x, c) # [b, 16, 512]
            
        x = x.view(-1, int(math.sqrt(self.n)), int(math.sqrt(self.n)), self.attn_dim) # [b, 4, 4, 512]
        x = x.permute(0,3,1,2) # [b, 512, 4, 4]

        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

        if in_ch != out_ch:
            self.conv = double_conv(in_ch, out_ch)
        else:
            self.conv = double_conv(in_ch + in_ch // 2, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, resolution, input_channels, output_channel, heads=8, attn_dim=512, time_embed_dim=64, cond_embed_dim=512, blocks=3):
        super(UNet, self).__init__()
        
        self.input_channels = input_channels
        self.ted = time_embed_dim
        self.patch_num = int((resolution // (2**4)) **2) 

        self.fc1 = nn.Linear(self.ted, 64)
        self.fc2 = nn.Linear(self.ted, 128)
        self.fc3 = nn.Linear(self.ted, 256)
        self.fc4 = nn.Linear(self.ted, 512)

        self.inc = inconv(input_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        
        self.down_attn = down_attn(patch_num=self.patch_num, attn_dim=attn_dim, cond_embed_dim=cond_embed_dim, heads=heads, blocks=blocks)
        
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.outc = nn.Conv2d(64, output_channel, kernel_size=3, padding=1)

    def forward(self, x, t, c):
        '''
        x: [b, c, h, w] = [b, 1, 64, 64]
        t: [b, time_embed_dim] = [b, 256]
        c: [c, cond_embed_dim] = [b, 64]  
        '''
        
        t1 = self.fc1(t).view(-1, 64*1, 1, 1) # [b, 64, 1, 1]
        t2 = self.fc2(t).view(-1, 64*2, 1, 1) # [b, 128, 1, 1]
        t3 = self.fc3(t).view(-1, 64*4, 1, 1) # [b, 256, 1, 1]
        t4 = self.fc4(t).view(-1, 64*8, 1, 1) # [b, 512, 1, 1]

        x1 = self.inc(x) # [b, 64, 64, 64]
        x2 = self.down1(t1 + x1) # [b, 128, 32, 32]
        x3 = self.down2(t2 + x2) # [b, 256, 16, 16]
        x4 = self.down3(t3 + x3) # [b, 512, 8, 8]

        # cross attn w/ condition 
        z = self.down_attn(t4 + x4, c)  # [b, 512, 4, 4]

        x = self.up1(z, x4) + t4 # [b, 512, 8, 8]
        x = self.up2(x, x3) + t3 # [b, 256, 16, 16]
        x = self.up3(x, x2) + t2 # [b, 128, 32, 32]
        x = self.up4(x, x1) + t1 # [b, 64, 64, 64]
        x = self.outc(x) # [b, 1, 64, 64]

        return x
    
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    rsl = 64
    in_cnl = 1
    out_cnl = 1
    ted = 64
    ced = 512
    heads = 16
    blocks = 3
    
    x = torch.ones([4, 1, rsl, rsl]).cuda()
    t = torch.ones([4, ted]).cuda()
    c = torch.ones([4, ced]).cuda()

    model = UNet(resolution=rsl, input_channels=in_cnl, heads=heads, output_channel=out_cnl, time_embed_dim=ted, cond_embed_dim=ced, blocks=blocks).to(device)

    print(summary(model,x,t,c))
    print('input:',x.shape,t.shape,c.shape)
    print('output:',model(x,t,c).shape)
    print('===================================')
