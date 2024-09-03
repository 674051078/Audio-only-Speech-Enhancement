import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from dptfsnet import DPTFSNet

class en_SqueezedTCM(nn.Module):
    def __init__(self,input_channel: int,middle_channel: int,dilation: int,F=int):
        super(en_SqueezedTCM, self).__init__()
        self.input_channel = input_channel
        self.middle_channel = middle_channel
        self.dilation = dilation
        self.F = F
        self.pad1 = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.pad2 = nn.ConstantPad2d((2, 2, 1, 0), value=0.)
        self.in_conv = nn.Conv2d(input_channel, middle_channel, kernel_size=(1, 1), stride=(1, 2), bias=False)
        self.in_conv1 =nn.Conv2d(middle_channel, middle_channel, kernel_size=(2,3), bias=False)                               
        self.in_conv2 =nn.Conv2d(middle_channel, middle_channel, kernel_size=(2,5), bias=False)
        self.in_conv12 =nn.Conv2d(middle_channel*2, middle_channel, kernel_size=(1,1), bias=False)

        self.out_norm = nn.LayerNorm(self.F)
        # self.out_norm = nn.GroupNorm(input_channel//4,input_channel,eps=1e-08)
        self.out_prelu = nn.PReLU(input_channel)
        # self.Dconv_gate1 = DilatedConv(middle_channel, twidth=3, fwidth=3, dilation=dilation)
        # self.Dconv_gate2 = DilatedConv(middle_channel, twidth=3, fwidth=3, dilation=dilation)
        self.out_conv = nn.Conv2d(middle_channel, input_channel, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        #x = self.in_prelu(self.in_norm(self.in_conv(x)))
        x = self.in_conv(x)
        x1 = self.in_conv1(self.pad1(x))
        x2 = self.in_conv2(self.pad2(x))
        x12 = torch.cat((x1,x2),dim=1)
        x12_out= torch.sigmoid(self.in_conv12(x12))
        x_out = x12_out*x1 + x12_out*x2
        x_out_last = self.out_prelu(self.out_norm(self.out_conv(x_out)))

        del x1,x2,x12,x12_out,x_out
        return x_out_last

class en_SqueezedTCM(nn.Module):
    def __init__(self,
                 input_channel: int,
                 middle_channel: int,
                 dilation: int,
                 F=int
                 ):
        super(en_SqueezedTCM, self).__init__()
        self.input_channel = input_channel
        self.middle_channel = middle_channel
        self.dilation = dilation
        self.F = F
        self.pad1 = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.pad2 = nn.ConstantPad2d((2, 2, 1, 0), value=0.)
        self.in_conv = nn.Conv2d(input_channel, middle_channel, kernel_size=(1, 1), stride=(1, 2), bias=False)
        self.in_conv1 =nn.Conv2d(middle_channel, middle_channel, kernel_size=(2,3), bias=False)                               
        self.in_conv2 =nn.Conv2d(middle_channel, middle_channel, kernel_size=(2,5), bias=False)
        self.in_conv12 =nn.Conv2d(middle_channel*2, middle_channel, kernel_size=(1,1), bias=False)

        self.out_norm = nn.LayerNorm(self.F)
        # self.out_norm = nn.GroupNorm(input_channel//4,input_channel,eps=1e-08)
        self.out_prelu = nn.PReLU(input_channel)

        self.out_conv = nn.Conv2d(middle_channel, input_channel, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        #x = self.in_prelu(self.in_norm(self.in_conv(x)))
        x = self.in_conv(x)

        x1 = self.in_conv1(self.pad1(x))
        x2 = self.in_conv2(self.pad2(x))
        x12 = torch.cat((x1,x2),dim=1)
        x12_out= torch.sigmoid(self.in_conv12(x12))
        x_out = x12_out*x1 + x12_out*x2
        x_out_last = self.out_prelu(self.out_norm(self.out_conv(x_out)))
        del x1,x2,x12,x12_out,x_out
        return x_out_last

class en_SqueezedTCM(nn.Module):
    def __init__(self,
                 input_channel: int,
                 middle_channel: int,
                 dilation: int,
                 F=int
                 ):
        super(en_SqueezedTCM, self).__init__()
        self.input_channel = input_channel
        self.middle_channel = middle_channel
        self.dilation = dilation
        self.F = F
        self.pad1 = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.pad2 = nn.ConstantPad2d((2, 2, 1, 0), value=0.)
        self.in_conv = nn.Conv2d(input_channel, middle_channel, kernel_size=(1, 1), stride=(1, 2), bias=False)
        self.in_conv1 =nn.Conv2d(middle_channel, middle_channel, kernel_size=(2,3), bias=False)                               
        self.in_conv2 =nn.Conv2d(middle_channel, middle_channel, kernel_size=(2,5), bias=False)
        self.in_conv12 =nn.Conv2d(middle_channel*2, middle_channel, kernel_size=(1,1), bias=False)

        self.out_norm = nn.LayerNorm(self.F)
        # self.out_norm = nn.GroupNorm(input_channel//4,input_channel,eps=1e-08)
        self.out_prelu = nn.PReLU(input_channel)

        self.out_conv = nn.Conv2d(middle_channel, input_channel, kernel_size=(1, 1), bias=False)

        #self.conv1d =nn.Conv1d(input_channel,input_channel,kernel_size=1)
    def forward(self, x):
        #x = self.in_prelu(self.in_norm(self.in_conv(x)))
        x = self.in_conv(x)

        x1 = self.in_conv1(self.pad1(x))
        x2 = self.in_conv2(self.pad2(x))
        x12 = torch.cat((x1,x2),dim=1)
        x12_out= torch.sigmoid(self.in_conv12(x12))
        x_out = x12_out*x1 + x12_out*x2
        x_out_last = self.out_prelu(self.out_norm(self.out_conv(x_out)))
        del x1,x2,x12,x12_out,x_out
        return x_out_last

class de_SqueezedTCM_outlast(nn.Module):
    def __init__(self,
                 input_channel: int,
                 middle_channel: int,
                 out_channels:int,
                 dilation: int,
                 F=int,
                 activation="prelu"
                 ):
        super(de_SqueezedTCM_outlast, self).__init__()
        self.input_channel = input_channel
        self.middle_channel = middle_channel
        self.dilation = dilation
        self.F = F
        if activation:
           self.activation = nn.PReLU(out_channels)
        else:
           self.activation = nn.Tanh(out_channels)
        self.pad1 = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.pad2 = nn.ConstantPad2d((2, 2, 1, 0), value=0.)
        self.in_conv = nn.Conv2d(input_channel, middle_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.in_conv1 =nn.Conv2d(middle_channel, middle_channel, kernel_size=(2,3), bias=False)                               
        self.in_conv2 =nn.Conv2d(middle_channel, middle_channel, kernel_size=(2,5), bias=False)
        self.in_conv12 =nn.Conv2d(middle_channel*2, middle_channel, kernel_size=(1,1), bias=False)

        self.out_norm = nn.LayerNorm(self.F*2)
        self.out_conv =nn.ConvTranspose2d(middle_channel, 2, kernel_size=(1,2),stride=(1, 2))
    def forward(self, x):
        x = self.in_conv(x)
        x1 = self.in_conv1(self.pad1(x))
        x2 = self.in_conv2(self.pad2(x))
        x12 = torch.cat((x1,x2),dim=1)
        x12_out= torch.sigmoid(self.in_conv12(x12))
        x_out = x12_out*x1 + x12_out*x2
        x_out_last = self.activation(self.out_norm(self.out_conv(x_out)))
        del x1,x2,x12,x12_out,x_out
        return x_out_last

class de_SqueezedTCM(nn.Module):
    def __init__(self,
                 input_channel: int,
                 middle_channel: int,
                 dilation: int,
                 F=int,
                 activation="prelu"
                 ):
        super(de_SqueezedTCM, self).__init__()
        self.input_channel = input_channel
        self.middle_channel = middle_channel
        self.dilation = dilation
        self.F = F
        self.pad1 = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.pad2 = nn.ConstantPad2d((2, 2, 1, 0), value=0.)
        self.in_conv = nn.Conv2d(input_channel, middle_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.in_conv1 =nn.Conv2d(middle_channel, middle_channel, kernel_size=(2,3), bias=False)                               
        self.in_conv2 =nn.Conv2d(middle_channel, middle_channel, kernel_size=(2,5), bias=False)
        self.in_conv12 =nn.Conv2d(middle_channel*2, middle_channel, kernel_size=(1,1), bias=False)

        self.out_norm = nn.LayerNorm(self.F*2)
        # self.out_norm = nn.GroupNorm(input_channel//4,input_channel,eps=1e-08)
        self.out_prelu = nn.PReLU(middle_channel*2)
        self.out_conv =nn.ConvTranspose2d(middle_channel, middle_channel*2, kernel_size=(1,2),stride=(1, 2))
    def forward(self, x):
        #x = self.in_prelu(self.in_norm(self.in_conv(x)))
        x = self.in_conv(x)

        x1 = self.in_conv1(self.pad1(x))
        x2 = self.in_conv2(self.pad2(x))
        x12 = torch.cat((x1,x2),dim=1)
        x12_out= torch.sigmoid(self.in_conv12(x12))
        x_out = x12_out*x1 + x12_out*x2
        x_out = self.out_conv(x_out)
        x_out_last = self.out_prelu(self.out_norm(x_out))
        del x1,x2,x12,x12_out,x_out
        return x_out_last

class SPConvTranspose2d1(nn.Module):  # 深度可分离逐点卷积  [b, 64, 32, 64]
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 2)):
        super(SPConvTranspose2d1, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels*2, kernel_size=kernel_size, stride=stride)
    def forward(self, x):  # padding后
         # [b,32,t,32]
        out = self.conv(x)  #[b,64,t,64]
        return out

class SPConvTranspose2d(nn.Module):  # 深度可分离逐点卷积  [b, 64, 32, 64]
    def __init__(self, in_channels, out_channels, kernel_size, r=2, stride=(1, 1)):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.r =r
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size, stride=stride)

    def forward(self, x):  # padding后
         # [2,64,32,64]
        out = self.conv(x)  # [1, 128, 32, 64]
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))  # [b, 2, 64, 250 32]
        out = out.permute(0, 2, 3, 4, 1)  # [b, 64, 250, 32, 2]
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))  # [b, 64, 250, 64]
        return out

class InforComu(nn.Module):
        
        def __init__(self, src_channel, tgt_channel,width=64,F=int):
            super(InforComu, self).__init__()
            self.width=width
            self.F = F
            #self.enc_norm = nn.LayerNorm(128)
            self.enc_prelu = nn.PReLU(self.width)
            self.comu_conv = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 1), stride=(1, 1))
            #self.Layer_norm = nn.LayerNorm(self.F)
            self.groupnorm = nn.GroupNorm(self.width//4,self.width,eps=1e-08)
        def forward(self, src, tgt):
            outputs=self.groupnorm(tgt*self.enc_prelu(self.comu_conv(src)))
            return outputs

class Net(nn.Module):  # 网络总结构
    def __init__(self,in_channels=2,out_channels=2,width=64):
        super(Net, self).__init__()
        self.in_channels =in_channels
        self.out_channels = out_channels # 最终输出
        self.kernel_size = (2, 3)

        self.width = width
        # 输入层
        self.inp_conv = nn.Conv2d(self.in_channels, out_channels=self.width, kernel_size=(1, 1))
        self.inp_norm = nn.LayerNorm(512)  # [b, 16, t 256]
        #self.inp_norm = nn.GroupNorm(self.width//1,self.width,eps=1e-08)
        self.inp_prelu = nn.PReLU()

        self.encoder = Encoder(width=64)
        # 双路径Transformer
        self.DPTFSNet = DPTFSNet(64, 64, num_layers=4)  # nhead=4, num_layers=6  # [2, 64, nframes, 32]   N,F=64
        #self.AHA = AHAM(64)

        self.decoder = Decoder(width=64)
        self.a = nn.Parameter(torch.Tensor([0.5]))
        self.b = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, feat):
        
        out = self.inp_prelu(self.inp_norm(self.inp_conv(feat)))
        #b编码器
        out,enc_list = self.encoder(out)

        #中间层
        out= self.DPTFSNet(out)  # 只定义了输入输出通道数64c
       # out1 = self.AHA(out1)

        #解码器
        Mask_out,out_ri = self.decoder(out,enc_list)

        #Mask_out = self.mask_conv(Mask_out)
        #幅度
        out_ri_mask =Mask_out * feat

        out_wav= out_ri_mask*self.a + out_ri*self.b

        return out_wav

class AHAM(nn.Module):  # aham merge
    def __init__(self,  input_channel=64, kernel_size=(1,1), bias=True, act=nn.ReLU(True)):
        super(AHAM, self).__init__()

        #self.k = Parameter(torch.zeros(1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.max_pool2d = nn.AdaptiveMaxPool2d(1)

        # self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)
        self.conv1=nn.Conv2d(input_channel, 1, kernel_size, (1, 1), bias=bias)

    def merge(self, x, y):
        batch, channel, height, width, blocks = x.size()
        input_x = x  # N*C*H*W*K
        # input_y = y #N*1*1*K*1
        y = self.softmax(y)
        context = torch.matmul(input_x, y)  # N*C*H*W*1
        context = context.view(batch, channel, height, width)  # N*C*H*W
        return context

    def forward(self, input_list): #X:BCTFG Y:B11G1
        batch, channel, frames, frequency= input_list[-1].size()
        x_list = []
        y_list = []
        for i in range(len(input_list)):
            input = self.avg_pool(input_list[i]) #+self.max_pool2d(input_list[i]
            y = self.conv1(input)
            x = input_list[i].unsqueeze(-1)
            y= y.unsqueeze(-2)
            x_list.append(x)
            y_list.append(y)
        
        x_merge = torch.cat((x_list[0],x_list[1], x_list[2]), dim=-1)
        #print(str(x_merge.shape))
        y_merge = torch.cat((y_list[0],y_list[1], y_list[2]), dim=-2)
        #print(str(y_merge.shape))
        #out1 = self.merge(x, y)
        y_softmax = self.softmax(y_merge)
        #print(str(y_softmax.shape))
        aham= torch.matmul(x_merge, y_softmax)
        aham= aham.view(batch, channel, frames, frequency)
        aham_output = input_list[-1] + aham
        #print(str(aham_output.shape))
        del x_list,y_list,x_merge,y_merge, y_softmax,aham

        return aham_output

class Encoder(nn.Module):
    def __init__(self,width=64):
        super(Encoder, self).__init__()

        self.width = width
        # 下采样层1
        self.D_conv1 =en_SqueezedTCM(input_channel=self.width,middle_channel=32,dilation=1,F=256)
        # 下采样层2
        self.D_conv2 =en_SqueezedTCM(input_channel=self.width,middle_channel=32,dilation=2,F=128)
        # 下采样层3
        self.D_conv3 =en_SqueezedTCM(input_channel=self.width,middle_channel=32,dilation=3,F=64)
        # 下采样层4
        self.D_conv4 =en_SqueezedTCM(input_channel=self.width,middle_channel=32,dilation=5,F=32)
    def forward(self, x):
        en_list = []  # 存入跳跃连接列表
        x1 = self.D_conv1(x)  # 补偿+ 输入 # self.con2(left + right)
        en_list.append(x1)  # 存入跳跃连接列表self.con2(left + right)

        x2 = self.D_conv2(x1)
        en_list.append(x2)

        x3 = self.D_conv3(x2)
        en_list.append(x3)

        x_out = self.D_conv4(x3)
        en_list.append(x_out)

        # x = self.D_conv4(x)
        # en_list.append(x)
        del x1,x2 ,x3
        return x_out, en_list

class Encoder1(nn.Module):
    def __init__(self,width=64):
        super(Encoder1, self).__init__()

        self.width = width
        # 下采样层1
        self.D_conv1 =en_SqueezedTCM(input_channel=self.width,middle_channel=32,dilation=1,F=128)
        # 下采样层2
        self.D_conv2 =en_SqueezedTCM(input_channel=self.width,middle_channel=32,dilation=2,F=64)
        # 下采样层3
        self.D_conv3 =en_SqueezedTCM(input_channel=self.width,middle_channel=32,dilation=3,F=32)
        # # 下采样层4
        # self.D_conv4 =en_SqueezedTCM(input_channel=self.width,kd1= 3,middle_channel=32,dilation=3,F=32)
    def forward(self, x):
        en_list = []  # 存入跳跃连接列表
        x1 = self.D_conv1(x)  # 补偿+ 输入 # self.con2(left + right)
        en_list.append(x1)  # 存入跳跃连接列表self.con2(left + right)

        x2 = self.D_conv2(x1)
        en_list.append(x2)

        x_out = self.D_conv3(x2)
        en_list.append(x_out)
        del x1,x2
        return x_out, en_list

class Decoder(nn.Module):
    def __init__(self,width=64):
        super(Decoder, self).__init__()

        self.width=width
        # 上采样层4
        self.de_conv4 =de_SqueezedTCM(input_channel=self.width*2,middle_channel=32,dilation=5,F=32,activation="prelu")
        # 上采样层3
        self.de_conv3 =de_SqueezedTCM(input_channel=self.width*2,middle_channel=32,dilation=3,F=64,activation="prelu")
        # 上采样层2
        self.de_conv2 =de_SqueezedTCM(input_channel=self.width*2,middle_channel=32,dilation=2,F=128,activation="prelu")
        # 上采样层1
        self.de_conv1=de_SqueezedTCM_outlast(input_channel=self.width*2,middle_channel=32,out_channels=2,dilation=1,F=256,activation="prelu")
        
        self.mask_conv1=de_SqueezedTCM_outlast(input_channel=self.width*2,middle_channel=32,out_channels=2,dilation=1,F=256,activation="tanh")

    def forward(self, x, en_list):

        de_x4 = self.de_conv4(torch.cat((x, en_list[-1]), dim=1))
        # #de_list.append(de_x4)
        mask_x4 = self.de_conv4(torch.cat((x, en_list[-1]), dim=1))
        # #mask_list.append(mask_x4)

        de_x3 = self.de_conv3(torch.cat((de_x4, en_list[-2]), dim=1))
        #de_list.append(de_x3)
        mask_x3 = self.de_conv3(torch.cat((mask_x4, en_list[-2]), dim=1))
 
        de_x2 = self.de_conv2(torch.cat((de_x3, en_list[-3]), dim=1))
        #de_list.append(de_x2)
        mask_x2 = self.de_conv2(torch.cat((mask_x3, en_list[-3]), dim=1))
 
        de_out = self.de_conv1(torch.cat((de_x2, en_list[-4]), dim=1))
        #de_list.append(de_x1)
        mask_out  = self.mask_conv1(torch.cat((mask_x2, en_list[-4]), dim=1))
 
        del mask_x2,mask_x3,de_x2,de_x3, de_x4,mask_x4,
        return de_out,mask_out
