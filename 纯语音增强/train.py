import torch
#from preprocess import TorchOLA
from AudioData import TrainingDataset, TrainCollate, EvalCollate, EvalDataset
from Datasets import AudioDataset
from torch.utils.data import DataLoader
from new_model import Net
from metric import get_pesq, get_stoi
from helper_funcs import numParams, compLossMask, snr
from criteria import mse_loss, TF_loss
from checkpoints import Checkpoint
import os
from tqdm import tqdm
import numpy as np
import json
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 训练参数
max_epochs =100
batch_size =2
eval_steps = 10000
# lr scheduling
lr= 0.0005# 初始学习率为 0.0005
#weight_delay = 1e-7
step_num = 0
warm_ups = 4000
# STFT参数
sr = 16000
train_length = 64000  #最长4秒截断，t可变f=256
n_fft =1023
hop_length = 256
window = torch.hann_window(n_fft).cuda()

# 保存模型
resume_model = None  # 不是None的话 就是 相应的存model的路径
#resume_model = 'LuoShijie-code/RCED_modelsave/latest.model-34.model'
model_save_path = 'LuoShijie-code/Propose_最终模型_modelsave'
if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)    
early_stop = True

train_file_list_path = 'THCHS30数据集/trainset-40000条语音/train_file_list'
validation_file_list_path = 'THCHS30数据集/trainset-40000条语音/val_file_list'

# data and data_loader
train_data = TrainingDataset(train_file_list_path, frame_size=511, frame_shift=256)
train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=TrainCollate(),
                          drop_last=True
                          )

validation_data = EvalDataset(validation_file_list_path, frame_size=511, frame_shift=256)
validation_loader = DataLoader(validation_data,
                               batch_size=1,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=EvalCollate(),
                               drop_last=True
                               )

# define model
model = Net()
#model = torch.nn.DataParallel(model)
model = model.cuda()
print('Number of learnable parameters: %d' % numParams(model))

optimizer = torch.optim.Adam(model.parameters(), lr)

time_loss = mse_loss()
tf_loss = TF_loss()

def stft(x):
    return torch.stft(x, n_fft, hop_length, window=window)
# istft = ISTFT(opt.n_fft, opt.hop_length, window='hanning').to(device)
def istft(x, length):
    return torch.istft(x,
                        n_fft,
                        hop_length,
                        length=length,
                        window=window)

def validate(net, eval_loader, test_metric=False):
    net.eval()
    if test_metric:
        print('********Starting metrics evaluation on val dataset**********')
        total_stoi = 0.0
        total_snr = 0.0
        total_pesq = 0.0

    with torch.no_grad():
        count, total_eval_loss = 0, 0.0
        for k, (features, labels) in enumerate(eval_loader):
            features = features.cuda()  # [1,signal_len]
            labels = labels.cuda()  # [signal_len]

            features_stft = stft(features).permute(0, 3, 1, 2).contiguous().transpose(2, 3)   #[batch_size, F , T, 2C]->[batch_size, 2c , F , T]
            feat_R = features_stft[:,0,:,:].unsqueeze(1)#[B,1,T,256]
            feat_I = features_stft[:,1,:,:].unsqueeze(1)#[B,1,T,256]

            feat = torch.cat((feat_R,feat_I), dim=1)
            output= model(feat)
            output = output.transpose(2, 3).contiguous().permute(0, 2, 3, 1)

            output = istft(output, features.shape[1])  # [1,signal_len]
            # output = output[:, :, :labels.shape[-1]]
            output = torch.squeeze(output, dim=0)

            output = output[:labels.shape[-1]]  # keep length same (output label)

            eval_loss = torch.mean((output - labels) ** 2)
            total_eval_loss += eval_loss.data.item()

            est_sp = output.cpu().numpy()
            cln_raw = labels.cpu().numpy()
            if test_metric:
                st = get_stoi(cln_raw, est_sp, sr)
                pe = get_pesq(cln_raw, est_sp, sr)
                sn = snr(cln_raw, est_sp)
                total_pesq += pe
                total_snr += sn
                total_stoi += st

            count += 1
        avg_eval_loss = total_eval_loss / count
    net.train()
    if test_metric:
        return avg_eval_loss, total_stoi / count, total_pesq / count, total_snr / count
    else:
        return avg_eval_loss


# train model
if resume_model:
    print('Resume model from "%s"' % resume_model)
    checkpoint = Checkpoint()
    checkpoint.load(resume_model)

    start_epoch = checkpoint.start_epoch + 1
    best_val_loss = checkpoint.best_val_loss
    prev_val_loss = checkpoint.prev_val_loss
    num_no_improv = checkpoint.num_no_improv
    half_lr = checkpoint.half_lr
    model.load_state_dict(checkpoint.state_dict)
    optimizer.load_state_dict(checkpoint.optimizer)

else:
    print('Training from scratch.')
    start_epoch = 0
    best_val_loss = float("inf")
    prev_val_loss = float("inf")
    num_no_improv = 0
    half_lr = False

train_losses=[]
valid_losses=[]

val_STOI=[]
val_SNR=[]
val_PESQ=[]
epochs = []
time1 = []
# learn_rate=[]

for epoch in range(start_epoch, max_epochs):
    model.train()
    total_train_loss, count, ave_train_loss = 0.0, 0, 0.0
    start_time = time.time()

    for index, (features, labels, sig_len) in enumerate(train_loader):

        features = features.cuda()
        # label -- [batch_size, 1, signal_length]
        labels = labels.cuda()
        loss_mask = compLossMask(labels, nframes=sig_len)
        optimizer.zero_grad()

        features_stft = stft(features).permute(0, 3, 1, 2).contiguous().transpose(2, 3)   #[batch_size, F , T, 2C]->[batch_size, 2c , F , T]
        feat_R = features_stft[:,0,:,:].unsqueeze(1)#[B,1,T,256]
        feat_I = features_stft[:,1,:,:].unsqueeze(1)#[B,1,T,256]

        feat = torch.cat((feat_R,feat_I), dim=1)
        output= model(feat)

        #output -- [batch_size, 1, sig_len_recover]
        output = output.transpose(2, 3).contiguous().permute(0, 2, 3, 1)
       # print('output:', output.shape)
        output = istft(output, features.shape[1])
        output = torch.squeeze(output, dim=1)
        output = output[:, :labels.shape[-1]]  # [batch_size, sig_len]
  

        # 还原为时域后，计算时间、频域损失
        loss_time = time_loss(output, labels, loss_mask)
        loss_tf = tf_loss(output, labels, loss_mask)

        loss = 0.2 * loss_time + loss_tf   # 0.4/ 0.6
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()

        train_loss = loss.data.item()
        total_train_loss += train_loss

        count += 1

        del loss, loss_time, loss_tf, output, loss_mask, features, labels
        print('iter = {}/{}, epoch = {}/{}, train_loss = {:.5f}'.format(index + 1, len(train_loader), epoch + 1, max_epochs, train_loss))

        if (index + 1) % eval_steps == 0:
            ave_train_loss = total_train_loss / count

            # validation
            avg_eval_loss = validate(model, validation_loader)
            model.train()

            print('Epoch [%d/%d], Iter [%d/%d],  ( TrainLoss: %.5f | EvalLoss: %.5f )' % (
            epoch + 1, max_epochs, index + 1, len(train_loader), ave_train_loss, avg_eval_loss))

            count = 0
            total_train_loss = 0.0


        if (index + 1) % len(train_loader) == 0:
            break

    # validate metric
    avg_eval, avg_stoi, avg_pesq, avg_snr = validate(model, validation_loader, test_metric=True)
    model.train()
    print('#' * 50)
    print('')
    print('After {} epoch the performance on validation score is a s follows:'.format(epoch + 1))
    print('')

    print('train_Avg_loss: {:.5f}'.format(ave_train_loss))
    train_losses.append(ave_train_loss)

    print('Avg_loss: {:.5f}'.format(avg_eval))
    valid_losses.append(avg_eval)

    print('STOI: {:.5f}'.format(avg_stoi))
    val_STOI.append(avg_stoi)

    print('SNR: {:.5f}'.format(avg_snr))
    val_SNR.append(avg_snr)

    print('PESQ: {:.5f}'.format(avg_pesq))
    val_PESQ.append(avg_pesq)

    epochs.append(epoch)
    if epoch<30:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    else: #epoch >= 30:
    # adjust learning rate and early stop
        if avg_eval >= prev_val_loss:
            num_no_improv += 1
            if num_no_improv == 2:
                half_lr = True
            if num_no_improv >= 5 and early_stop is True:   #超过10个 停止训练
                print("No improvement and apply early stop")
                break
        else:
            num_no_improv = 0

    if half_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 2
                half_lr = False

    #print('Learning rate adjusted to  %6f' % (param_group['lr']))
    prev_val_loss = avg_eval

    if avg_eval < best_val_loss:
        best_val_loss = avg_eval
        is_best_model = True
    else:
        is_best_model = False

    # save model
    latest_model = 'latest.model'
    best_model = 'best.model'

    checkpoint = Checkpoint(start_epoch=epoch,
                            best_val_loss=best_val_loss,
                            prev_val_loss=prev_val_loss,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            num_no_improv=num_no_improv,
                            half_lr=half_lr)
    checkpoint.save(is_best=is_best_model,
                    filename=os.path.join(model_save_path, latest_model + '-{}.model'.format(epoch + 1)),
                    best_model=os.path.join(model_save_path, best_model))

    loss_data = {'train_losses': train_losses, 'valid_losses': valid_losses}
    with open('losses.json', 'w') as losses:
        try:
            json.dump(loss_data, losses)
            print("成功写入losses.json 文件")
        except Exception as e:
            print("写入losses.json 文件时出现错误:", str(e))


    mertic_data = {'valid_STOI': val_STOI, 'valid_SNR': val_SNR,'valid_PESQ':val_PESQ}
    with open('valid-mertic.json', 'w') as mertic:
        try:
            json.dump(mertic_data, mertic)
            print("成功写入 valid-mertic.json 文件")
        except Exception as e:
            print("写入 valid-mertic.json 文件时出现错误:", str(e))

    end_time = time.time()
    execution_time = end_time - start_time
    time1.append(execution_time)
    # 打印执行时间
    print(f"程序执行时间: {execution_time} 秒")
    time_data = {'times': time1}
    with open('time.json', 'w') as times:
        try:
            json.dump(time_data, times)
            print("成功写入losses.json 文件")
        except Exception as e:
            print("写入losses.json 文件时出现错误:", str(e))