import torch
import torch.nn as nn
from metric import get_stoi, get_pesq
from scipy.io import wavfile
import numpy as np
from checkpoints import Checkpoint
from torch.utils.data import DataLoader
from helper_funcs import snr, numParams
from eval_composite import eval_composite
from AudioData import EvalDataset, EvalCollate
from new_model import Net
import h5py
import os
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sr = 16000
n_fft = 1023
hop_length = 256
window = torch.hamming_window(n_fft).cuda()
def stft(x):
    return torch.stft(x, n_fft, hop_length, window=window)#,return_complex=True
def istft(x,length):
    return torch.istft(x,
                        n_fft,
                        hop_length,
                        length=length,
                        window=window)

#file_name = 'test_mix_1'
test_file_list_path = r'/data/ShenXiwen/THCHS30数据集/真实环境噪声/test_file_list_10db'# + '/' + file_name
audio_file_save = r'/data/ShenXiwen/LuoShijie-code/Propose_最终模型_audiosave/真实应用/10db'# + '/' + 'enhanced_' + file_name
if not os.path.isdir(audio_file_save):
    os.makedirs(audio_file_save)

with open(test_file_list_path, 'r') as test_file_list:
    file_list = [line.strip() for line in test_file_list.readlines()]


test_data = EvalDataset(test_file_list_path)
test_loader = DataLoader(test_data,
                        batch_size=1,
                        shuffle=False,
                        num_workers=4,
                        collate_fn=EvalCollate())

ckpt_path = r'LuoShijie-code/Propose_最终模型_modelsave/best.model'

model = Net()
#model = nn.DataParallel(model, device_ids=[0])
#使用DataParallel来实现模型的多GPU训练。DataParallel可以将一个单GPU的模型复制到多个GPU上，每个GPU负责处理一部分数据，然后再将它们的计算结果汇总起来

checkpoint = Checkpoint()
checkpoint.load(ckpt_path)
model.load_state_dict(checkpoint.state_dict)
#从Checkpoint对象中加载的模型权重（状态字典）加载到之前创建的多GPU模型中。这是为了确保model与之前保存的模型权重保持一致。
model.cuda()
print(checkpoint.start_epoch)
print(checkpoint.best_val_loss)
print(numParams(model))

test_loss=[]
# test function
def evaluate(net, eval_loader):
    net.eval()

    print('********Starting metrics evaluation on test dataset**********')
    total_stoi = 0.0
    total_ssnr = 0.0
    total_pesq = 0.0
    total_csig = 0.0
    total_cbak = 0.0
    total_covl = 0.0

    with torch.no_grad():
        count, total_eval_loss = 0, 0.0
        for k, (features, labels) in enumerate(eval_loader):
            features = features.cuda()  # [1, 1, num_frames,frame_size]
            labels = labels.cuda()  # [signal_len, ]

            features_stft = stft(features)
            features_stft=features_stft.permute(0, 3, 1, 2).contiguous().transpose(2, 3)   #[batch_size, F , T, 2C]->[batch_size, 2c , F , T]
            feat_R = features_stft[:,0,:,:].unsqueeze(1)#[B,1,T,256]
            feat_I = features_stft[:,1,:,:].unsqueeze(1)#[B,1,T,256]

            #noisy_mag=(torch.norm(noisy_mag, dim=1)).unsqueeze(1)
            feat = torch.cat((feat_R,feat_I), dim=1)
            output= model(feat)
            output = output.transpose(2, 3).contiguous().permute(0, 2, 3, 1)

            output = istft(output, labels.shape[-1])

            output = output.squeeze()  # (33024，)：[sig_len_recover,]
#将输出张量压缩，以删除大小为1的任何维度。

            eval_loss = torch.mean((output - labels) ** 2)
            total_eval_loss += eval_loss.data.item()

            est_sp = output.cpu().numpy()
            cln_raw = labels.cpu().numpy()

            eval_metric = eval_composite(cln_raw, est_sp, sr)
            total_pesq += eval_metric['pesq']
            total_ssnr += eval_metric['ssnr']
            total_stoi += eval_metric['stoi']
            total_cbak += eval_metric['cbak']
            total_csig += eval_metric['csig']
            total_covl += eval_metric['covl']

#下一行代码的作用是将音频信号 est_sp 以 WAV 格式写入到指定的文件路径。
# 其中，audio_file_save 是保存文件的目录，file_list[k] 是要保存的文件名，sr 是采样率，est_sp 是音频信号。
            #wavfile.write(os.path.join(audio_file_save, os.path.basename(file_list[k])), sr, est_sp.astype(np.float32))
            output_path = os.path.join(audio_file_save, os.path.basename(file_list[k]))

# 如果文件名不以.wav结尾，可以添加
            if not output_path.lower().endswith('.wav'):
                output_path += '.wav'

            wavfile.write(output_path, sr, est_sp.astype(np.float32))
#作用是将经过处理后的音频信号 est_sp 以 WAV 格式写入文件。文件的保存路径由 audio_file_save 和原始音频文件的基本文件名组成。
# 同时，指定了采样率 sr，以及经过处理后的音频信号的数据类型。这样就实现了将处理后的音频保存到指定路径的文件中。
#os.path.basename(file_list[k]) 获取了原始音频文件路径 file_list[k] 的基本文件名部分，
# 将这两部分路径通过 os.path.join 函数连接在一起，就得到了要保存的完整文件路径

            count += 1
        avg_eval_loss = total_eval_loss / count
        test_loss.append(avg_eval_loss)

    return avg_eval_loss, total_stoi / count, total_pesq / count, total_ssnr / count, total_csig / count, total_cbak / count, total_covl / count
print(test_loss)

def eva_noisy(file_path):
    print('********Starting metrics evaluation on raw noisy data**********')
    total_stoi = 0.0
    total_ssnr = 0.0
    total_pesq = 0.0
    total_csig = 0.0
    total_cbak = 0.0
    total_covl = 0.0
    count = 0

    with open(file_path, 'r') as eva_file_list:
        file_list = [line.strip() for line in eva_file_list.readlines()]

    for i in range(len(file_list)):
        filename = file_list[i]
        reader = h5py.File(filename, 'r')

        noisy_raw = reader['noisy_raw'][:]
        cln_raw = reader['clean_raw'][:]

        eval_metric = eval_composite(cln_raw, noisy_raw, sr)

        total_pesq += eval_metric['pesq']
        total_ssnr += eval_metric['ssnr']
        total_stoi += eval_metric['stoi']
        total_cbak += eval_metric['cbak']
        total_csig += eval_metric['csig']
        total_covl += eval_metric['covl']

        count += 1

    return total_stoi / count, total_pesq / count, total_ssnr / count, total_cbak / count, total_csig / count, total_covl / count


avg_eval, avg_stoi, avg_pesq, avg_ssnr, avg_csig, avg_cbak, avg_covl = evaluate(model, test_loader)

print('Avg_loss: {:.4f}'.format(avg_eval))
print('STOI: {:.4f}'.format(avg_stoi))
print('SSNR: {:.4f}'.format(avg_ssnr))
print('PESQ: {:.4f}'.format(avg_pesq))
print('CSIG: {:.4f}'.format(avg_csig))
print('CBAK: {:.4f}'.format(avg_cbak))
print('COVL: {:.4f}'.format(avg_covl))
