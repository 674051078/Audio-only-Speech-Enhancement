import numpy as np
import random
import math
import librosa
import argparse
from scipy.io import wavfile
import glob
import os
import h5py
import time
import torch
import soundfile as sf

def gen_train_pair(tra_clean_index):

    train_clean_path = r'THCHS30数据集/trainset-40000条语音/clean'
    train_noisy_path = r'THCHS30数据集/trainset-40000条语音/noisy'
    train_mix_path = r'THCHS30数据集/trainset-40000条语音/mix/train_mix'

    train_clean_name = sorted(os.listdir(train_clean_path))
    train_noisy_name = sorted(os.listdir(train_noisy_path))

    print('训练集数量：',len(tra_clean_index))

    for count in range(len(tra_clean_index)):

        clean_name = train_clean_name[tra_clean_index[count]]
        noisy_name = train_noisy_name[tra_clean_index[count]]
        #print(clean_name, noisy_name)
        if clean_name == noisy_name:   #
            file_name = '%s_%d' % ('train_mix', count+1)
            train_writer = h5py.File(train_mix_path + '/' + file_name, 'w')

            clean_audio, sr = librosa.load(os.path.join(train_clean_path, clean_name), sr=16000)  #已包含采样
            noisy_audio, sr1 = librosa.load(os.path.join(train_noisy_path, noisy_name), sr=16000)

            train_writer.create_dataset('noisy_raw', data=noisy_audio.astype(np.float32), chunks=True)
            train_writer.create_dataset('clean_raw', data=clean_audio.astype(np.float32), chunks=True)
            train_writer.close()
        else:
            raise TypeError('clean file and noisy file do not match')
    # save .txt file
    print('sleep for 3 secs...')
    time.sleep(3)
    print('begin save .txt file...')
    train_file_list_path = 'THCHS30数据集/trainset-40000条语音/train_file_list'

    train_file_list = sorted(glob.glob(os.path.join(train_mix_path, '*')))
    read_train = open(train_file_list_path, "w+")

    for i in range(len(train_file_list)):
        read_train.write("%s\n" % (train_file_list[i]))

    read_train.close()
    print('making training data finished!')


def gen_val_pair(val_clean_index):

    val_clean_path = r'THCHS30数据集/trainset-40000条语音/clean'
    val_noisy_path = r'THCHS30数据集/trainset-40000条语音/noisy'
    val_mix_path = r'THCHS30数据集/trainset-40000条语音/mix/val_mix'

    val_clean_name = sorted(os.listdir(val_clean_path))
    val_noisy_name = sorted(os.listdir(val_noisy_path))
    print( '验证集数量：',len(val_clean_index))

    for count in range(len(val_clean_index)):
        clean_name = val_clean_name[val_clean_index[count]]
        noisy_name = val_noisy_name[val_clean_index[count]]

        # print(clean_name, noisy_name)
        if clean_name == noisy_name:  #
            file_name1 = '%s_%d' % ('val_mix', count + 1)
            val_writer = h5py.File(val_mix_path + '/' + file_name1, 'w')

            clean_audio, sr = librosa.load(os.path.join(val_clean_path, clean_name), sr=16000)  # 已包含采样
            noisy_audio, sr1 = librosa.load(os.path.join(val_noisy_path, noisy_name), sr=16000)

            val_writer.create_dataset('noisy_raw', data=noisy_audio.astype(np.float32), chunks=True)
            val_writer.create_dataset('clean_raw', data=clean_audio.astype(np.float32), chunks=True)
            val_writer.close()

    # save .txt file
    print('sleep for 3 secs...')
    time.sleep(3)
    print('begin save .txt file...')
    validiation_file_list_path = 'THCHS30数据集/trainset-40000条语音/val_file_list'
    val_file_list = sorted(glob.glob(os.path.join(val_mix_path, '*')))
    read_val = open(validiation_file_list_path, "w+")

    for i in range(len(val_file_list)):
        read_val.write("%s\n" % (val_file_list[i]))

    read_val.close()
    print('making validation data finished!')


def gen_test_pair():

    test_clean_path = r'/data/ShenXiwen/LuoShijie-code/dataset/voice_bank/testset/clean_test'
    test_noisy_path = r'/data/ShenXiwen/LuoShijie-code/dataset/voice_bank/testset/noisy_test'
    test_mix_path = r'/data/ShenXiwen/LuoShijie-code/dataset/voice_bank_mix/test_mix'

    test_clean_name = sorted(os.listdir(test_clean_path))
    test_noisy_name = sorted(os.listdir(test_noisy_path))

    print('测试集数量：',len(test_clean_name))

    for count in range(len(test_clean_name)):

        clean_name = test_clean_name[count]
        noisy_name = test_noisy_name[count]
        # print(clean_name, noisy_name)
        if clean_name == noisy_name:
            file_name = '%s_%d' % ('test_mix', count + 1)
            train_writer = h5py.File(test_mix_path + '/' + file_name, 'w')

            clean_audio, sr = librosa.load(os.path.join(test_clean_path, clean_name), sr=16000)
            noisy_audio, sr1 = librosa.load(os.path.join(test_noisy_path, noisy_name), sr=16000)

            train_writer.create_dataset('noisy_raw', data=noisy_audio.astype(np.float32), chunks=True)
            train_writer.create_dataset('clean_raw', data=clean_audio.astype(np.float32), chunks=True)
            train_writer.close()
        else:
            raise TypeError('clean file and noisy file do not match')

    # save .txt file
    print('sleep for 3 secs...')
    time.sleep(3)
    print('begin save .txt file...')
    test_file_list_path = '/data/ShenXiwen/LuoShijie-code/test_file_list'
    test_file_list = sorted(glob.glob(os.path.join(test_mix_path, '*')))
    read_test = open(test_file_list_path, "w+")

    for i in range(len(test_file_list)):
        read_test.write("%s\n" % (test_file_list[i]))

    read_test.close()
    print('making testing data finished!')

def gen_test_pair1():

    test_clean_path = r'/data/ShenXiwen/THCHS30数据集/真实环境噪声/clean/10db'
    test_noisy_path = r'/data/ShenXiwen/THCHS30数据集/真实环境噪声/noisy/10db'
    test_mix_path = r'/data/ShenXiwen/THCHS30数据集/真实环境噪声/test_mix/10db_mix'

    test_clean_name = sorted(os.listdir(test_clean_path))
    test_noisy_name = sorted(os.listdir(test_noisy_path))

    print('测试集数量：',len(test_clean_name))

    for count in range(len(test_clean_name)):

        clean_name = test_clean_name[count]
        noisy_name = test_noisy_name[count]
        # print(clean_name, noisy_name)
        if clean_name == noisy_name:
            file_name = '%s_%d' % ('test_mix', count + 1)
            train_writer = h5py.File(test_mix_path + '/' + file_name, 'w')

            clean_audio, sr = librosa.load(os.path.join(test_clean_path, clean_name), sr=16000)
            noisy_audio, sr1 = librosa.load(os.path.join(test_noisy_path, noisy_name), sr=16000)

            train_writer.create_dataset('noisy_raw', data=noisy_audio.astype(np.float32), chunks=True)
            train_writer.create_dataset('clean_raw', data=clean_audio.astype(np.float32), chunks=True)
            train_writer.close()
        else:
            raise TypeError('clean file and noisy file do not match')

    # save .txt file
    print('sleep for 3 secs...')
    time.sleep(3)
    print('begin save .txt file...')

    test_file_list_path = '/data/ShenXiwen/THCHS30数据集/真实环境噪声/test_file_list_10db'
    test_file_list = sorted(glob.glob(os.path.join(test_mix_path, '*')))
    read_test = open(test_file_list_path, "w+")
    # test_file_list = sorted(glob.glob(os.path.join(test_mix_path, '*')))
    # read_test = open("test_file_list", "w+")

    for i in range(len(test_file_list)):
        read_test.write("%s\n" % (test_file_list[i]))

    read_test.close()
    print('making testing data finished!')

if __name__ == "__main__":
#但如果脚本被导入为模块到其他脚本中，那么 if __name__ == '__main__': 下缩进的代码将不会执行。这是因为此时脚本被用作一个可重用的模块，
#其他脚本可以导入它并访问它的函数、变量等，而不必执行脚本的主要功能。
#当脚本被直接运行时（作为主程序），if __name__ == '__main__': 下缩进的代码将会执行。这是因为此时脚本被视为主程序，
#你希望执行一些特定的操作，如启动应用程序或执行脚本的主要功能。
    # val_clean_index = []
    # # train_clean_path = r'/data/ShenXiwen/LuoShijie-code/dataset/test1111/train111/train-clean'#测试
    # # train_noisy_path = r'/data/ShenXiwen/LuoShijie-code/dataset/test1111/train111/train-noisy'#测试
    # train_clean_path = r'THCHS30数据集/trainset-40000条语音/clean'
    # train_noisy_path = r'THCHS30数据集/trainset-40000条语音/noisy'
    # train_clean_name = sorted(os.listdir(train_clean_path))
    # train_noisy_name = sorted(os.listdir(train_noisy_path))

    # print(len(train_clean_name))
    # val_num = math.ceil(len(train_clean_name) * 0.08)
    # #val_index = np.random.randint(0, len(train_clean_name), val_num)
    # all_indices = np.arange(len(train_clean_name))  # 创建包含所有索引的数组
    # val_index = np.random.choice(all_indices, val_num, replace=False)
    # for i in range(len(val_index)):
    #     if val_index[i] not in val_clean_index:
    #         val_clean_index.append(val_index[i])  # 取出连续一段纯净语音作验证语音
    # print(len(val_clean_index))
    # tra_clean_index = [x for x in range(0, len(train_clean_name)) if x not in val_clean_index]  # 取出除去验证的训练语
    # print(len(tra_clean_index))


    #gen_train_pair(tra_clean_index)  # noisy 与 clean同名
    #gen_val_pair(val_clean_index)
    #gen_test_pair()
    gen_test_pair1()

