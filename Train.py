# import copy
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np

from torch.utils.data import DataLoader
#from SubbandCRN import TSCNet
# from CRNModule import  CRN
from GDConformer import TSCNet
from SISNRLoss import  si_snr
import librosa
import sys
from tqdm import tqdm,trange
from pesq import pesq
from pystoi import stoi
import pysepm
from NetConfig import get_net_params
import timeit
import torchaudio

from DataFromH5 import DatasetFromHdf5

from ptflops import get_model_complexity_info

global_max = None
global_min = None

def magnitude_feature_1(file):
    wav, fs = librosa.load(file,sr=16000,mono=True)
    spec = librosa.stft(wav, n_fft=512, win_length=512, hop_length=128, window='hann').T
    mag = np.abs(spec)
    magnitude = np.zeros((257, 257), dtype='float32')
    magnitude[0: len(mag), 0: len(mag[0])] = mag
    magnitude[np.isneginf(magnitude)] = 0
    mag=np.array(magnitude)
    phase = np.angle(spec)

    return mag,phase

def Complex(waveform):
    n_fft = 512  # FFT窗口大小
    hop_length = 128  # 帧移
    win_length = 512  # 窗口长度

    # 计算复数谱
    spec,sr=librosa.load(waveform,sr=16000)
    spec = librosa.stft(spec, n_fft=n_fft, hop_length=hop_length, win_length=win_length).T

    real=np.real(spec)
    imaginary=np.imag(spec)

    mags,phase=librosa.magphase(spec)

    complex_spec=np.stack((real,imaginary),axis=-1)

    complex_spec=complex_spec.transpose((2,0,1))
    # print("complex_spec.shape___", complex_spec.shape)
    return complex_spec,mags,phase
def compute_spectrum(complex_data):

    real_part=complex_data[:, 0, :, :]
    imaginary_part=complex_data[:, 1, :, :]
    magnitude=torch.sqrt(real_part**2 + imaginary_part**2).unsqueeze(1)

    return real_part,imaginary_part,magnitude

def triple_loss(output,label):
    est_real, est_imag, est_mag = compute_spectrum(output)

    clean_real, clean_imag, clean_mag = compute_spectrum(label)
    loss_mag = torch.nn.MSELoss()(est_mag, clean_mag)
    loss_ri = torch.nn.MSELoss()(est_real, clean_real) + torch.nn.MSELoss()(est_imag, clean_imag)

    est_spec = torch.stack([est_real, est_imag], dim=-1).squeeze(1).permute(0, 2, 1, 3)
    clean_spec = torch.stack([clean_real, est_imag], dim=-1).squeeze(1).permute(0, 2, 1, 3)

    est_spec = torch.view_as_complex(est_spec )
    est_audio= torch.istft(est_spec, 512, 128, window=torch.hamming_window(512).cuda(device), onesided=True)
    clean_spec = torch.view_as_complex(clean_spec)
    clean_audio = torch.istft(clean_spec, 512, 128, window=torch.hamming_window(512).cuda(device), onesided=True)

    time_loss = torch.mean(torch.abs(est_audio - clean_audio))
    loss = 0.1*loss_ri + 0.9 * loss_mag + 1.0 * time_loss
    return loss


'复数谱进行逆傅里叶变换,尚未做正规化，并且无误的。'
def test(noisy_file,clean_file,model):
    # print(noisy_file)
    clean_wav, fs = librosa.load(clean_file,sr=16000,mono=True)
    noisy_complex,noisy_magnitude, noisy_phase = Complex(noisy_file)
    noisy_complex = torch.from_numpy(noisy_complex).to(torch.float32).to(device)

    noisy_complex=noisy_complex.unsqueeze(0)


    noisy_complex_output = model(noisy_complex)

    noisy_complex=noisy_complex_output.squeeze(0)

    noisy_complex_output = noisy_complex[0,:, :] + 1j * noisy_complex[1,:, :]
    # print("de",noisy_complex_output.shape)
    noisy_complex_output = noisy_complex_output.cpu().detach().numpy()
    noisy_complex_output=noisy_complex_output.T
    enh_wav = librosa.istft(noisy_complex_output, hop_length=128, window="hann")


    return enh_wav, clean_wav


def val(cleandir, noisyloc,model):
    noisydir = noisyloc
    files = os.listdir(noisydir)

    clean_stoi_sum=0.0
    noisy_stoi_sum=0.0
    enh_stoi_sum=0.0
    clean_pesq_sum=0.0
    noisy_pesq_sum=0.0
    enh_pesq_sum = 0.0
    enh_csig_sum=0.0
    enh_cbak_sum=0.0
    enh_covl_sum=0.0
    noisy_csig_sum=0.0
    noisy_cbak_sum=0.0
    noisy_covl_sum=0.0
    step=0
    for fname in tqdm(files,desc="Test"):
        clean_file = os.path.join(cleandir, fname)
        noisy_file = os.path.join(noisyloc, fname)
        # print(noisy_file)
        enh_data, clean_data = test(noisy_file, clean_file, model)
        if len(clean_data) > len(enh_data):
            clean_data = clean_data[:len(enh_data)]
        else:
            enh_data = enh_data[:len(clean_data)]
        # noisy_origin_data = noisy_origin_data[:len(clean_data)]
        noisy_data,sr=librosa.load(noisy_file,sr=16000,mono=True)
        noisy_data = noisy_data[:len(clean_data)]

        clean_stoi = stoi(clean_data, clean_data, fs_sig=16000)
        noisy_stoi = stoi(clean_data, noisy_data, fs_sig=16000)
        enh_stoi = stoi(clean_data, enh_data, fs_sig=16000)

        clean_pesq=pesq(16000,clean_data, clean_data,'wb')
        noisy_pesq = pesq( 16000,clean_data, noisy_data,'wb')
        enh_pesq =pesq(16000,clean_data, enh_data,'wb')

        noisy_csig, noisy_cbak, noisy_covl = pysepm.composite(clean_data, noisy_data, fs=16000)
        enh_csig,enh_cbak,enh_covl=pysepm.composite(clean_data,enh_data,fs=16000)

        step = step + 1
        clean_stoi_sum=clean_stoi_sum+clean_stoi
        noisy_stoi_sum=noisy_stoi_sum+noisy_stoi
        enh_stoi_sum=enh_stoi_sum+enh_stoi
        clean_pesq_sum = clean_pesq_sum + clean_pesq
        noisy_pesq_sum=noisy_pesq_sum+noisy_pesq
        enh_pesq_sum=enh_pesq_sum+enh_pesq
        noisy_csig_sum=noisy_csig_sum+noisy_csig
        enh_csig_sum=enh_csig_sum+enh_csig
        noisy_cbak_sum = noisy_cbak_sum + noisy_cbak
        enh_cbak_sum=enh_cbak_sum+enh_cbak
        noisy_covl_sum=noisy_covl_sum+noisy_covl
        enh_covl_sum=enh_covl_sum+enh_covl

    Noisy_PESQ_mean =noisy_pesq_sum/step
    Enhanced_PESQ_mean =enh_pesq_sum/step
    Noisy_STOI_mean=noisy_stoi_sum/step
    Enhanced_STOI_mean=enh_stoi_sum/step
    Noisy_CSIG_mean=noisy_csig_sum/step
    Enhanced_CSIG_mean = enh_csig_sum / step
    Noisy_CBAK_mean=noisy_cbak_sum/step
    Enhanced_CBAK_mean = enh_cbak_sum / step
    Noisy_COVL_mean=noisy_covl_sum/step
    Enhanced_COVL_mean = enh_covl_sum / step


    result = (
        f"Validation Results --> "
        f"noisy pesq: {Noisy_PESQ_mean:.4f} --> enhanced pesq: {Enhanced_PESQ_mean:.4f} --> "
        f"noisy stoi: {Noisy_STOI_mean:.4f} --> enhanced stoi: {Enhanced_STOI_mean:.4f} --> "
        f"noisy csig: {Noisy_CSIG_mean:.4f} --> enhanced csig: {Enhanced_CSIG_mean:.4f} --> "
        f"noisy cbak: {Noisy_CBAK_mean:.4f} --> enhanced cbak: {Enhanced_CBAK_mean:.4f} --> "
        f"noisy covl: {Noisy_COVL_mean:.4f} --> enhanced covl: {Enhanced_COVL_mean:.4f}"
)


    log_path = "G:\\gzcm\\MultiSEModel\\save\\log\\log.txt"
    with open(log_path, "a") as fo:
        fo.write(result + "\n")


    print(result)

if __name__ == "__main__":
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ##------------------定义模型-----------------##

    model=TSCNet()
    model=model.to(device)
    model.train()

    optimizer=torch.optim.Adam(params=model.parameters(), lr=0.0001,eps=1e-05)
    # scheduler = (optimizer, step_size=5, gamma=0.1)

    train_data = DatasetFromHdf5("G:\\gzcm\\Dataset\\TrainData\\vb_train_new_complex.h5")
    train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True, drop_last=True)


    train_epoch =200

    for epoch in trange(train_epoch, desc='Epochs', unit='epoch'):
        epoch_loss = 0.0
        total_batches=len(train_loader)


        model.train()
        macs, params = get_model_complexity_info(model, (2, 257, 257), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)

        print(f"MACs: {macs}", f"Params: {params}")
        start_time = time.time()
        batch_bar = tqdm(train_loader, desc='Batches', unit='batch', leave=False)
        for i_batch, (feature,label) in enumerate(batch_bar):

            feature = feature.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output=model(feature)

            # 计算损失函数
            loss =triple_loss(output,label)
            loss=loss.to(device)

            #误差反向传播
            loss.backward()
            #进行参数更新
            optimizer.step()
            epoch_loss+=loss.item()

            batch_bar.set_postfix({'loss': f'{loss.item():.4f}'})


        loss_mean = epoch_loss /total_batches
        print(f"\nepoch = {epoch+1:02d} mean_loss = {loss_mean:.6f}")
        duration = time.time() - start_time
        print(f"\nEpoch {epoch+1:02d}  mean_loss = {loss_mean:.6f}  time = {duration:.1f}s")
        # 进行模型保存
        save_name = os.path.join('G:\\gzcm\\MultiSEModel\\save\\model\\', 'model_%d_%.4f.pth' % (epoch, loss_mean))
        torch.save(model, save_name)

        macs, params = get_model_complexity_info(model, (2, 257, 257), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)

        print(f"MACs: {macs}",f"Params: {params}")

        print("--------------test-------------------")

        clean_file = "G:\\gzcm\\Dataset\\voicebank\\wavesplit\\test\\clean\\"
        noisy_fil = "G:\\gzcm\\Dataset\\voicebank\\wavesplit\\test\\noisy\\"

        noisy = os.listdir(noisy_fil)
        model.eval()
        with torch.no_grad():
            val(clean_file, noisy_fil, model)



