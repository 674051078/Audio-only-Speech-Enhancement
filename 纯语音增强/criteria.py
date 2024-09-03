import torch
from preprocess import TorchSignalToFrames
from helper_funcs import ConvSTFT,ConviSTFT
from scipy import linalg
import numpy as np
import scipy
import torch.nn.functional as F

EPS = 1e-8
n_fft = 1023
hop_length = 256
window = torch.hann_window(n_fft).cuda()

class mse_loss(object):
    def __call__(self, outputs, labels, loss_mask):
        masked_outputs = outputs * loss_mask  # 真实的长度
        masked_labels = labels * loss_mask    # 真实的长度
        loss = torch.sum((masked_outputs - masked_labels)**2.0) / torch.sum(loss_mask)
        return loss


# 双分支协作
class TF_loss(object):  #(output_com, label)时域
    def __call__(self, outputs, labels, loss_mask):
        n_fft = 511
        hop_length = 256
        WINDOW = torch.hann_window(n_fft).cuda()
        y_pred = outputs * loss_mask  # 真实的长
        y_true = labels * loss_mask

        pred_stft = torch.stft(y_pred, n_fft, hop_length, win_length=n_fft, window=WINDOW, center=True)
        true_stft = torch.stft(y_true, n_fft, hop_length, win_length=n_fft, window=WINDOW, center=True)
        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        true_stft_real, true_stft_imag = true_stft[:, :, :, 0], true_stft[:, :, :, 1]
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real ** 2 + true_stft_imag ** 2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag ** (2 / 3))
        pred_imag_c = pred_stft_imag / (pred_mag ** (2 / 3))
        true_real_c = true_stft_real / (true_mag ** (2 / 3))
        true_imag_c = true_stft_imag / (true_mag ** (2 / 3))
        real_loss = torch.mean((pred_real_c - true_real_c) ** 2)
        imag_loss = torch.mean((pred_imag_c - true_imag_c) ** 2)
        mag_loss = torch.mean((pred_mag ** (1 / 3) - true_mag ** (1 / 3)) ** 2)
        return (real_loss + imag_loss)*0.1 + mag_loss
    
# Phasen
def l2_norm(s1, s2):
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm 

def si_snr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)

def calc_sdr_torch(estimation, origin, mask=None):
    """
    batch-wise SDR caculation for one audio file on pytorch Variables.
    estimation: (batch, nsample)
    origin: (batch, nsample)
    mask: optional, (batch, nsample), binary
    """

    if mask is not None:
        origin = origin * mask
        estimation = estimation * mask

    origin_power = torch.pow(origin, 2).sum(1, keepdim=True) + EPS  # (batch, 1)

    scale = torch.sum(origin * estimation, 1, keepdim=True) / origin_power  # (batch, 1)

    est_true = scale * origin  # (batch, nsample)
    est_res = estimation - est_true  # (batch, nsample)

    true_power = torch.pow(est_true, 2).sum(1)
    res_power = torch.pow(est_res, 2).sum(1)
    loss = - (10 * torch.log10(true_power) - 10 * torch.log10(res_power))

    return loss.mean()  # (batch, 1)

def stft(x):
    return torch.stft(x, n_fft, hop_length, window=window)
# istft = ISTFT(opt.n_fft, opt.hop_length, window='hanning').to(device)
#def istft(x, length):
    return torch.istft(x,
                        n_fft,
                        hop_length,
                        length=length,
                        window=window)

def Loss_1(y_pred, y_true, loss_mask):  #(output_com, label)时域
        n_fft = 1023
        hop_length = 256
        WINDOW = torch.hann_window(n_fft).cuda()
        # WINDOW = torch.sqrt(torch.hann_window(1200) + 1e-8).cuda()
        # 做法不对 ，带入0去stft算误差
        y_pred = y_pred * loss_mask  # 真实的长
        y_true = y_true * loss_mask
        # WINDOW = torch.sqrt(torch.hann_window(1200,device=0) + 1e-8)
        # 计算snr损失
        # snr = torch.div(torch.mean(torch.square(y_pred - y_true), dim=1, keepdim=True),(torch.mean(torch.square(y_true), dim=1, keepdim=True) + 1e-7))
        # snr_loss = 10 * torch.log10(snr + 1e-7)
        # 计算stft后得复值
        pred_stft = torch.stft(y_pred, n_fft, hop_length, win_length=n_fft, window=WINDOW, center=True)
        true_stft = torch.stft(y_true, n_fft, hop_length, win_length=n_fft, window=WINDOW, center=True)
        pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
        true_stft_real, true_stft_imag = true_stft[:, :, :, 0], true_stft[:, :, :, 1]
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real ** 2 + true_stft_imag ** 2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag ** (2 / 3))
        pred_imag_c = pred_stft_imag / (pred_mag ** (2 / 3))
        true_real_c = true_stft_real / (true_mag ** (2 / 3))
        true_imag_c = true_stft_imag / (true_mag ** (2 / 3))
        real_loss = torch.sum((pred_real_c - true_real_c) ** 2) / torch.sum(loss_mask)
        imag_loss = torch.sum((pred_imag_c - true_imag_c) ** 2) / torch.sum(loss_mask)
        mag_loss = torch.sum((pred_mag ** (1 / 3) - true_mag ** (1 / 3)) ** 2)/ torch.sum(loss_mask)
        
        loss_tf = (real_loss + imag_loss)*0.1 +  mag_loss
        loss_time = torch.sum(torch.abs(y_pred - y_true)) / torch.sum(loss_mask)
        return loss_tf + 0.2*loss_time 
   
class stftm_loss(object):
    def __init__(self, frame_size=512, frame_shift=256, loss_type='mae'):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.loss_type = loss_type
        #self.device = device
        # 分帧式计算频点损失
        self.frame = TorchSignalToFrames(frame_size=self.frame_size,
                                            frame_shift=self.frame_shift)

        D = linalg.dft(frame_size)
        W = np.hamming(self.frame_size)
        DR = np.real(D)
        DI = np.imag(D)
        self.DR = torch.from_numpy(DR).float().cuda()  # to(self.device)
        self.DR = self.DR.contiguous().transpose(0, 1)
        self.DI = torch.from_numpy(DI).float().cuda()  # to(self.device)
        self.DI = self.DI.contiguous().transpose(0, 1)
        self.W = torch.from_numpy(W).float().cuda()  # to(self.device)
    def get_stftm(self, frames):
        frames = frames * self.W
        stft_R = torch.matmul(frames, self.DR)
        stft_I = torch.matmul(frames, self.DI)
        # stftm_r, stftm_i = torch.abs(stft_R), torch.abs(stft_I)
        return stft_R, stft_I

    def __call__(self, outputs, labels, loss_mask):

        outputs = self.frame(outputs)  # 分帧
        labels = self.frame(labels)
        loss_mask = self.frame(loss_mask)

        outputs_r, outputs_i = self.get_stftm(outputs)
        labels_r, labels_i = self.get_stftm(labels)

        masked_outputs_r, masked_outputs_i = outputs_r * loss_mask, outputs_i * loss_mask   # 真实的长度
        masked_labels_r, masked_labels_i = labels_r * loss_mask, labels_i * loss_mask    # 真实的长度
        masked_outputs_mag = torch.sqrt(masked_outputs_r ** 2 + masked_outputs_i ** 2 + 1e-12)
        masked_labels_mag = torch.sqrt(masked_labels_r ** 2 + masked_labels_i ** 2 + 1e-12)
        # _c功率压缩
        pred_real_c = masked_outputs_r / (masked_outputs_mag**(2/3))
        pred_imag_c = masked_outputs_i / (masked_outputs_mag**(2/3))
        true_real_c = masked_labels_r / (masked_labels_mag**(2/3))
        true_imag_c =  masked_labels_i / (masked_labels_mag**(2/3))
        #计算损失
        real_loss = torch.sum((pred_real_c - true_real_c) ** 2) / torch.sum(loss_mask)
        imag_loss = torch.sum((pred_imag_c - true_imag_c) ** 2) / torch.sum(loss_mask)
        mag_loss = torch.sum((masked_outputs_mag ** (1 / 3) - masked_labels_mag ** (1 / 3)) ** 2) / torch.sum(loss_mask)

        # real_loss = torch.sum(torch.abs(masked_outputs_r - masked_labels_r)) / torch.sum(loss_mask) # mae  mean=1/n?????
        # imag_loss = torch.sum(torch.abs(masked_outputs_i - masked_labels_i)) / torch.sum(loss_mask)
        # mag_loss = torch.sum(torch.abs(masked_outputs_mag - masked_labels_mag)) / torch.sum(loss_mask)

        all_loss = 0.5*(real_loss + imag_loss) + mag_loss

        return all_loss
