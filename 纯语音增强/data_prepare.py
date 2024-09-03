import librosa
import soundfile as sf
import random
import os
import numpy as np
from numpy.linalg import norm  # 范数计算


# 数据集不能存放在项目里，一定要存放在服务器下的文件夹
def SNR(x1, x2):
    snr = 10 * np.log10(norm(x1) / norm(x2))  # norm（x):表示x的2范数 --能量的度量
    return snr


# def Resample(input_signal, org_sr, tar_sr):
#     '''
#     :param input_signal: 输入信号
#     :param org_sr: 原始采样率   每秒的采样字节数
#     :param tar_sr: 新采样率
#     :return output_signal:  输出信号
#     '''

#     dtype = input_signal.dtype
#     audio_len = len(input_signal)
#     audio_time_max = 1.0 * (audio_len - 1) / org_sr  # 音频持续时间  =  数据长  /  采样率
#     org_time = 1.0 * np.linspace(0, audio_len, audio_len) / org_sr  # 横坐标 （各个长度对应的时间）
#     tar_time = 1.0 * np.linspace(0, int(audio_time_max * tar_sr),
#                                  int(audio_time_max * tar_sr)) / tar_sr  # （令总时长不变)新采样率下的信号长度会不同
#     output_signal = np.interp(tar_time, org_time, input_signal).astype(dtype)  # 得到对应新采样值的得到对应音频信号
#     return output_signal


if __name__ == '__main__':
    # 服务器上这两个目录得改   输入预处理的 纯净语音 和 噪声
    # clean_file_path = os.getcwd() + '/dataset_big/clean_train_val'  # 添加r避免字符转义
    # print(clean_file_path)
    # noise_file_path = os.getcwd() + '/dataset_big/noise_long_train'  # getcwd()返回当前工作目录
    # print(noise_file_path)
    # clean_files = os.listdir(clean_file_path)  # 列出纯净语音文件夹中的目录列表
    # # clean_file1 = os.getcwd() + os.path.join(r'\dataset\origin_dataset\clean' ,clean_files[0])
    # # print(clean_file1)
    # noise_files = os.listdir(noise_file_path)  # 列出噪声文件夹中的目录clean_file_path = os.getcwd() + '/dataset2/origin_dataset/clean'  # 添加r避免字符转义
    # print(len(noise_files))
    # noise_file_path = os.getcwd() + '/dataset2/origin_dataset/noise1'  # getcwd()返回当前工作目录
    # print(noise_file_path)
    # clean_files = os.listdir(clean_file_path)  # 列出纯净语音文件夹中的目录列表
    # # clean_file1 = os.getcwd() + os.path.join(r'\dataset\origin_dataset\clean' ,clean_files[0])
    # # print(clean_file1)
    # noise_files = os.listdir(noise_file_path)  # 列出噪声文件夹中的目录
    # snr = [-1, -2, -3, -4, -5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # -10到15 随机抽
    # snr = [15] #[-6, -3, 0, 3, 6, 9]  # 信噪比 6种
    # resample
    # resample = True#False
    # # pad_and_cut   是否裁剪
    # pad_and_cut =True  #False
    # mix_specch
    mix_specch = True

    # 先进行重采样 纯净语音 和 噪声
    # if resample:
    #     tar_sr = 8000  # 新采样率文中为8K 但常用为16k
    #     for i in range(len(clean_files)):  # 每个音频
    #         s, org_sr1 = librosa.load(os.path.join(clean_file_path, clean_files[i]), sr=None)  # 路径 和文件名
    #         s_resample = librosa.resample(s, orig_sr=org_sr1, target_sr=tar_sr)  # 更新采样率
    #         sf.write(os.path.join(os.getcwd() + '/dataset/resample_dataset/re_clean', clean_files[i].split('.')[0])
    #                   + '.wav', s_resample, tar_sr)  # 存入重采样采样后音频文件
    #
    #     for j in range(len(noise_files)):
    #         # load读取WAV、MP3等文件 sr 默认22050 ，sr=None以原始采样率读取文件，高于该采样率的音频文件会被下采样，低于该采样率的文件会被上采样
    #         # 返回 音频信号值(类型ndarray)和采样率
    #         n, org_sr2 = librosa.load(os.path.join(noise_file_path, noise_files[j]), sr=None)
    #         n_resample = librosa.resample(n, orig_sr=org_sr2, target_sr=tar_sr)
    #         # 保存音频（（连接路径名主件）文件名，音频内容，采样率）
    #         sf.write(os.path.join(os.getcwd() + '/dataset/resample_dataset/re_noise',noise_files[j].split('.')[0])  # 保存重采样后的文件
    #                  + '_8K' + '.wav', n_resample, tar_sr)

    # 将混合好的数据集裁剪到合适的尺寸用于训练
    # if pad_and_cut:
    #     # 重采样后的纯净语音修剪尺度
    #     m_clean_file_path = os.getcwd() + '/dataset/resample_dataset/re_clean'
    #     m_clean_files = os.listdir(m_clean_file_path)  # 列出重采样后的纯净语音目录
    #     for i in range(len(m_clean_files)):
    #         s, sr = librosa.load(os.path.join(m_clean_file_path, m_clean_files[i]), sr=None)
    #         audio_len = len(s)
    #         if (audio_len >16128):  # 在s上新截取24192的长度（3秒）      2秒：
    #             residual = audio_len - 16128
    #             num = random.randint(0, residual)
    #             s_new = s[0:16128]  #   纯净语音固定的前2秒
    #         if (audio_len < 16128):  # 补0
    #             s_new = np.pad(s, (0, 16128 - audio_len), 'constant')  # 连续值填充  前面0个后面补不够长度到s 填充值默认为0
    #         sf.write(os.path.join(os.getcwd() + '/dataset/pad_and_cut_dataset/pc_clean', m_clean_files[i]), s_new,
    #                  sr)

    # # # 噪声修剪尺度
    # m_noise_file_path = os.getcwd() + '/dataset/resample_dataset/re_noise'
    # m_noise_files = os.listdir(m_noise_file_path)
    # for j in range(len(m_noise_files)):
    #     n, sr = librosa.load(os.path.join(m_noise_file_path, m_noise_files[j]), sr=None)
    #     audio_len = len(n)
    #     if (audio_len > 16128):   #2帧长度  3帧24192
    #         residual = audio_len - 16128
    #         num = random.randint(0, residual)
    #         n_new = n[num:num + 16128]    #噪声随机切割2秒
    #     if (audio_len < 16128):
    #         n_new = np.pad(n, (0, 16128 - audio_len), 'constant')
    #     sf.write(os.path.join(os.getcwd() + '/dataset/pad_and_cut_dataset/pc_noise', m_noise_files[j]), n_new,
    #              sr)

    # 然后直接混合语音和噪声
    if mix_specch == True:
        re_clean_file_path = os.getcwd() + '/dataset_big/test/clean_8_9'
        re_noise_file_path = os.getcwd() + '/dataset_big/test/noise_long_test'
        clean_files = os.listdir(re_clean_file_path)
        noise_files = os.listdir(re_noise_file_path)
        print(len(noise_files))
        print(len(clean_files))
        for j in range(len(clean_files)):
            s, sr_clean = librosa.load(os.path.join(re_clean_file_path, clean_files[j]), sr=16000)
            for k in range(len(noise_files)):
                n, sr_noise = librosa.load(os.path.join(re_noise_file_path, noise_files[k]), sr=16000)# 按不同信噪比混合  多种组合
                if len(s) < len(n):
                    a=len(s)//16000
                    # start = random.randint(0, n.shape[0] - s.shape[0]) #n.shape[0] - s.shape[0]
                    # n1 = n[int(start):int(start) + s.shape[0]]  # 在n中抽取纯净语音等长音频
                    start = random.randint(0, 11-a-1)
                    # print(start)
                    b = start*16000
                    n1 = n[b : b + s.shape[0]]
                    print(len(n1)==len(s))# 在n中抽取纯净语音等长音频
                if len(s) > len(n):  # 扩长n的长度来等于S
                    n_list = []
                    for ax in range(0, (len(s) // len(n)) * 2):
                        n_list.append(n)
                    n_list = np.hstack(n_list)  # 列向量
                    n1 = n_list[0:s.shape[0]]
                if len(s) == len(n):
                    n1 = n
                    # 指定信噪比，噪声衰减因子（混合方式！）
                snr1 = np.random.choice([10] , replace=True, size=1)
                # print(snr1[0])
                # snr = np.random.randint(-5, 11, size=1)
                # s_sum = np.sum(s ** 2)
                # n_sum = np.sum(n ** 2)
                # alpha = np.sqrt(s_sum / (n_sum * pow(10, snr[k] / 10.0)))
                alpha = np.sqrt(norm(s) / (norm(n1) * pow(10, snr1[0] * 0.1)))  # 噪声衰减因子
                # n = n / norm(n) * norm(s) / (10.0 ** (0.05 * snr[k]))
                noise = alpha * n1
                y = s + noise
                sf.write(os.path.join(os.getcwd() + '/dataset_big/test/clean_test',
                                    str(j) + '_' + str(k) + '_' + str(clean_files[j].split('.')[0]) +
                                    '.wav'), s, 16000)
                sf.write(os.path.join(os.getcwd() + '/dataset_big/test/noisy_test',
                                str(j) + '_' + str(k) + '_' + str(clean_files[j].split('.')[0]) +
                                '.wav'), y, 16000)
                # sf.write(os.path.join(os.getcwd() + '/dataset_big/test/noise_test', 
                #                  str(j) + '_' + str(snr[k]) + 'db_' + str(clean_files[j].split('.')[0]) +
                #                     '.wav'), noise, 16000)

    # 测试集对语音尺寸没有特定要求，可以采用zero_padding的方式来与网络输入对其，增强后的语音再进行cut_zeros就可以了
