import torch
from torch.utils.data import Dataset
from preprocess import SignalToFrames, ToTensor
import soundfile as sf
import numpy as np
import random
import h5py
import glob
import os


class TrainingDataset(Dataset):  # 生成训练数据 和 标签
    r"""Training dataset."""

    def __init__(self, file_path, frame_size=512, frame_shift=256, nsamples=64000):  # 样本点数（每一条语音的）
        # file_path is the path of training dataset
        # option1: '/data/KaiWang/pytorch_learn/pytorch_for_speech/dataset/timit_mix/trainset/two_data'  #直接数据输入模式
        # option2 : .txt file format  file_path='/data/KaiWang/pytorch_learn/pytorch_for_speech/DDAEC/train_file_list'
        # self.file_list = glob.glob(os.path.join(file_path, '*'))

        with open(file_path, 'r') as train_file_list:  # 在训练数据文件中打开文件
            self.file_list = [line.strip() for line in train_file_list.readlines()]  # 每条语音对的位置

        self.nsamples = nsamples
        self.to_tensor = ToTensor()

    def __len__(self):
        # print(len(self.file_list))
        return len(self.file_list)  # 训练语音对总条数

    def __getitem__(self, index):
        filename = self.file_list[index]
        reader = h5py.File(filename, 'r')
        feature = reader['noisy_raw'][:]  # 输入含噪语音训练
        label = reader['clean_raw'][:]  # 输入的纯净语音标签
        reader.close()

        size = feature.shape[0] # batch_size
        start = random.randint(0, max(0, size + 1 - self.nsamples))
        feature = feature[start:start + self.nsamples]  # 超过4秒的截取
        label = label[start:start + self.nsamples]

        # 累加batch_size后输入
        # audio_len = feature.shape[0]
        # # 裁剪长度
        # if audio_len > self.nsamples:
        #     start = random.randint(0, (audio_len + 1 - self.nsamples))
        #     feature = feature[start:start + self.nsamples]  # 不够4秒补0吗
        #     label = label[start:start + self.nsamples]
        # if audio_len < self.nsamples:
        #     feature = np.pad(feature, (0, self.nsamples - audio_len), 'constant')
        feature = self.to_tensor(feature)  # [sig_len]  三维张量形式
        label = self.to_tensor(label)  # [sig_len]  二维张量形式

        return feature, label  # 每一对的


class EvalDataset(Dataset):
    r"""Evaluation dataset."""

    def __init__(self, file_path, frame_size=512, frame_shift=256, nsamples=64000):
        # self.filename = filename
        # print(file_path)
        with open(file_path, 'r') as validation_file_list:
            self.file_list = [line.strip() for line in validation_file_list.readlines()]
        self.nsamples = nsamples

        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        reader = h5py.File(filename, 'r')

        feature = reader['noisy_raw'][:]
        label = reader['clean_raw'][:]

        # 4维是单条送入模型直接吗？ 标签
        # audio_len = feature.shape[0]
        # 裁剪长度
        # if audio_len > self.nsamples:
        #     start = random.randint(0, (audio_len + 1 - self.nsamples))
        #     feature = feature[start:start + self.nsamples]  # 不够4秒补0吗
        #     label = label[start:start + self.nsamples]
        # if audio_len < self.nsamples:
        #     feature = np.pad(feature, (0, self.nsamples - audio_len), 'constant')

        feature = np.reshape(feature, [1, -1])  # [b=1, sig_len]
        feature = self.to_tensor(feature)  # [b=1, sig_len]
        label = self.to_tensor(label)  # [sig_len,]

        return feature, label


# testing 中 clean 和 noisy分不同的noisy和dB
class TestDataset(Dataset):
    def __init__(self, clean_file_path, noisy_file_path, frame_size, frame_shift, nsamples=32768):
        self.clean_test_name = os.listdir(clean_file_path)
        self.noisy_test_name = os.listdir(noisy_file_path)
        self.noisy_file_path = noisy_file_path
        self.clean_file_path = clean_file_path
        self.nsamples = nsamples

        # self.get_frames = SignalToFrames(frame_size=frame_size, frame_shift=frame_shift)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.clean_test_name)

    def __getitem__(self, index):
        noisy_name = '%s_%s.wav' % (self.clean_test_name[index].split('.')[0], os.path.basename(self.noisy_file_path))
        if noisy_name in self.noisy_test_name:
            noisy_audio, sr = sf.read(os.path.join(self.noisy_file_path, noisy_name))
            clean_audio, sr1 = sf.read(os.path.join(self.clean_file_path, self.clean_test_name[index]))
            if sr != 16000 and sr1 != 16000:
                raise ValueError('Invalid sample rate')

            # audio_len = noisy_audio.shape[0]
            # # 裁剪长度
            # if audio_len > self.nsamples:
            #     start = random.randint(0, (audio_len + 1 - self.nsamples))
            #     noisy_audio = noisy_audio[start:start + self.nsamples]  # 不够4秒补0吗
            #     clean_audio = clean_audio[start:start + self.nsamples]
            # if audio_len < self.nsamples:
            #     noisy_audio = np.pad(noisy_audio, (0, self.nsamples - audio_len), 'constant')

            feature = np.reshape(noisy_audio, [1, -1]) # [b=1, sig_len]
            feature = self.to_tensor(feature)  # [b=1, sig_len]         [1, 1, num_frames, frame_size]
            label = self.to_tensor(clean_audio)  # [sig_len, ]

        else:
            raise TypeError('Invalid noisy audio file')

        return feature, label


class TrainCollate(object):  # feature .label 是补零0的，sig_len是真实长度

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):  # [[feature_len],[label_len]]
            # 一个batch是三维 feature + label:[[[sig_len], [sig_len]]，[],[],[]] batch_size = 4

            # sorted by sig_len for label
            sorted_batch = sorted(batch, key=lambda x: x[1].shape[0], reverse=True)  # 对所有可迭代对象进行降序排序操作，不改变列表本身
            # [(feature_len，label_len),(),()...]  label_len长度从大到小 值矩阵
            lengths = list(map(lambda x: (x[0].shape[0], x[1].shape[0]), sorted_batch))
            # [(len(feature_len)，len(label_len)),(),()]
            # 创建特征-标签张量矩阵
            padded_feature_batch = torch.zeros((len(lengths), lengths[0][0]))  # [batch,len(feature_len)]
            padded_label_batch = torch.zeros((len(lengths), lengths[0][0]))  # [batch,len(label_len)]
            lengths1 = torch.zeros((len(lengths),), dtype=torch.int32)  # [batch,]

            for i in range(len(lengths)):
                # padded_feature_batch[0,0:len(feature_len)]= feature_len
                padded_feature_batch[i, 0:lengths[i][0]] = sorted_batch[i][0]  # 放入张量中，不够补O, 长度看起最大的
                # padded_label_batch[0, 0:len(label_len)]= label_len
                # 此时padded_feature_batch = padded_label_batch
                padded_label_batch[i, 0:lengths[i][1]] = sorted_batch[i][1]
                # lengths1[0] = len(label_len) 真实长度
                lengths1[i] = lengths[i][1]  # 只放label_sig_len

            return padded_feature_batch, padded_label_batch, lengths1
        else:
            raise TypeError('`batch` should be a list.')

class EvalCollate1(object):

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):#isinstance()函数用于判断一个对象是否是指定类型的实例
            feat_nchannels = batch[0][0].shape[0]  # 1
            feat_dim = batch[0][0].shape[-1]#含噪语音总长度
            label_dim = batch[0][1].shape[-1]#纯净语音总长度
            feature_batch = torch.zeros((len(batch), feat_nchannels,feat_dim ))
            label_batch = torch.zeros((len(batch), label_dim ))
            for i in range(len(batch)):
                feature_batch[i, :,0:batch[i][0].shape[-1] ] = batch[i][0]
                label_batch[i ,0:batch[i][1].shape[-1] ] = batch[i][1]
            #return batch[0][0], batch[0][1]#,batch[1][0], batch[1][1]
            return feature_batch,label_batch
        #batch是列表类型，这行代码将返回列表中第一个样本中的第一个和第二个元素
        else:
            raise TypeError('`batch` should be a list.')

class EvalCollate(object):

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):
            return batch[0][0], batch[0][1]
        else:
            raise TypeError('`batch` should be a list.')


class TestCollate(object):

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):
            # testdataloder 中的batch_size = 1; 因此就返回仅有的一个(feature, label)
            return batch[0][0], batch[0][1]
        else:
            raise TypeError('`batch` should be a list.')
