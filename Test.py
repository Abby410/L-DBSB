# import copytrain
import time
import librosa
import librosa.display
# import pypesq
import torch
import numpy as np
import os
# from GatedConformerSEModel import TSCNet
from CRNModule import  CRN
import matplotlib.pyplot as plt
# from SACRN import SACRN
# from SeparableCRN import DSCRN
from GatedConvNet import GCRN
from CRNModule import  CRN
from pesq import pesq
from pystoi import stoi
import pysepm
KMP_DUPLICATE_LIB_OK=True
# import csv
import datetime
from tqdm import trange
# from LSTM import net
import soundfile as sf


# from basic_function import feature_stft


def Magnitude(file):
    # wav, fs = librosa.load(file, sr=16000, mono=True)
    spec = librosa.stft(file, n_fft=512, win_length=512, hop_length=128, window='hann').T
    mag = np.abs(spec)
    LPS = np.log(mag ** 2+1e-5)
    phase = np.angle(spec)
    magnitude = np.zeros((1, 257, 257), dtype=float)
    magnitude[0, 0: len(LPS), 0: len(LPS[0])] = LPS
    magnitude[np.isneginf(magnitude)] = 0
    LPS = np.array(magnitude)
    # print("lps",LPS.shape)
    return LPS, phase


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
    # mags=np.sqrt(real **2+image**2)
    # print("mags",mags.shape)
    # phase=np.angle(spec)
    # print("complex_spec.shape", complex_spec.shape)
    # print(type(complex_spec))
    # real_part=complex_spec[..., 0]
    # print("____",real_part.shape)
    # imaginary_part = complex_spec[..., 1]
    # print("***", imaginary_part.shape)
    complex_spec=np.stack((real,imaginary),axis=-1)
    # print("complex_spec.shape___",complex_spec.shape)
    # print(complex_spec.shape)
    # print(type(complex_spec))
    complex_spec=complex_spec.transpose((2,0,1))
    # print("complex_spec.shape___", complex_spec.shape)
    return complex_spec,mags,phase

def draw_magnitude_wav(noisy_file, clean_file, enh_file, fs):
    plt.subplot(3, 1, 1)
    plt.title("Noisy Spectrum", fontsize=12)
    # plt.colorbar(format='%+2.0f dB')
    # plt.figure(figsize=(10, 6))
    plt.specgram(noisy_file, Fs=16000, scale_by_freq=True, sides='default', cmap="jet")
    plt.xlabel('Time/s', fontsize=10)
    plt.ylabel('Frequency/Hz', fontsize=10)

    plt.subplot(3, 1, 2)
    plt.title("Clean Spectrum", fontsize=12)
    # plt.colorbar(format='%+2.0f dB')
    # plt.figure(figsize=(10, 6))
    plt.specgram(clean_file, Fs=16000, scale_by_freq=True, sides='default', cmap="jet")
    plt.xlabel('Time/s', fontsize=10)
    plt.ylabel('Frequency/Hz', fontsize=10)

    plt.subplot(3, 1, 3)
    plt.title("Enhanced Spectrum", fontsize=12)
    # plt.colorbar(format='%+2.0f dB')
    # plt.figure(figsize=(10, 6))
    plt.specgram(enh_file, Fs=16000, scale_by_freq=True, sides='default', cmap="jet")
    plt.xlabel('Time/s', fontsize=10)
    plt.ylabel('Frequency/Hz', fontsize=10)

    plt.tight_layout()
    # plt.savefig('/content/drive/My Drive/spec.png', dpi=600)
    plt.show()


# def test(noisy_file, clean_file, model):
#     clean_wav, fs = librosa.load(clean_file, sr=16000, mono=True)
#     # clean_wav=torch.from_numpy(clean_wav).to(device)
#     noisy_magnitude, noisy_phase = Magnitude(noisy_file)
#     noisy_magnitude = torch.from_numpy(noisy_magnitude).to(torch.float32).to(device)
#
#     with torch.no_grad():
#         magnitude_output = model(x=noisy_magnitude)
#         # magnitude_output=magnitude_output.to(device)
#         # magnitude_output, target = model(x=magnitude_input, y=magnitude_clean)
#         magnitude_output = magnitude_output.cpu().detach().numpy()
#         # target =target.cpu().detach().numpy()
#         magnitude_output = np.reshape(magnitude_output, (-1, 257))
#         # print("magnitude_output", magnitude_output.shape)
#         magnitude = np.delete(magnitude_output, range(len(noisy_phase), 513), axis=0)
#         # print("magnitude", magnitude.shape)
#         magnitude = np.sqrt(np.exp(magnitude))
#         # print("___", magnitude.shape)
#         stft_reconstructed_clean = magnitude.T * noisy_phase.T
#         enh_wav = librosa.istft(stft_reconstructed_clean, hop_length=128, window="hann")
#
#     return enh_wav, clean_wav
def split_band(file):
    lps_features,phase=Magnitude(file)
    low_freq_cutoff = 300  # 低频分割频率
    high_freq_cutoff = 513  # 高频分割频率

    # 切分低频部分
    low_freq_features = lps_features[:, :low_freq_cutoff, :]
    # 切分高频部分
    high_freq_features = lps_features[:, low_freq_cutoff:high_freq_cutoff, :]

    return low_freq_features, high_freq_features

# def test(noisy_file,clean_file,model):
#
#     clean_wav, fs = librosa.load(clean_file,sr=16000,mono=True)
#     noisy_wav,fs= librosa.load(noisy_file,sr=16000,mono=True)
#     # clean_wav=torch.from_numpy(clean_wav).to(torch.float32).to(device)
#     # clean_wav=wav_normlize(clean_wav)
#     # noisy_wav= wav_normalize(noisy_wav)
#     noisy_magnitude,noisy_phase=Magnitude(noisy_wav)
#     # print("noisy_phase",noisy_phase.shape)
#     noisy_magnitude = torch.from_numpy(noisy_magnitude).to(torch.float32).to(device)
#     noisy_high_fre, noisy_low_fre=split_band(noisy_wav)
#     noisy_high_fre = torch.from_numpy(noisy_high_fre).to(torch.float32).to(device)
#     noisy_low_fre = torch.from_numpy(noisy_low_fre).to(torch.float32).to(device)
#     # noisy_magnitude_norm=torch.from_numpy(noisy_magnitude_norm).to(torch.float32).to(device)
#     # model.eval()
#     # with torch.no_grad():
#     magnitude_output = model(x1=noisy_low_fre,x2=noisy_magnitude,x3=noisy_high_fre)
# # magnitude_output=magnitude_output.to(device)
# # magnitude_output, target = model(x=magnitude_input, y=magnitude_clean)
#     magnitude_output = magnitude_output.cpu().detach().numpy()
#     # target =target.cpu().detach().numpy()
#     magnitude_output = np.reshape(magnitude_output, (-1, 257))
#     # print("magnitude_output", magnitude_output.shape)
#     magnitude = np.delete(magnitude_output, range(len(noisy_phase), 513), axis=0)
#     # print("magnitude", magnitude.shape)
#     # 求取最大值
#     # max_value = np.max(magnitude)
#     #
#     # # 稳定的指数计算方法
#     # magnitude = np.exp(magnitude - max_value)
#     # magnitude = np.sqrt(magnitude)
#     magnitude = np.sqrt(np.exp(magnitude))
#     # print("___", magnitude.shape)
#
#     stft_reconstructed_clean = magnitude.T * noisy_phase.T
#     enh_wav = librosa.istft(stft_reconstructed_clean, hop_length=128, window="hann")
#     # enh_wav=inverse_normalize(enh_wav)
#
#     return enh_wav, clean_wav

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

def val(cleandir, noisyloc,noisytype, model):
# def val(cleandir, noisyloc,  model):
    clean_stoi,clean_pesq,noisy_stoi, enh_stoi, noisy_pesq, enh_pesq = 0.0,0.0,0.0, 0.0, 0.0, 0.0
    print(noisyloc,noisytype)
    noisydir = noisyloc + noisytype + "//"
    # noisydir = noisyloc
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
    for i in range(len(files)):
        clean_file = cleandir + files[i]
        noisy_file = noisydir + files[i]
        print(noisy_file)
        enh_data, clean_data = test(noisy_file, clean_file, model)
        if len(clean_data) > len(enh_data):
            clean_data = clean_data[:len(enh_data)]
        else:
            enh_data = enh_data[:len(clean_data)]
        # noisy_origin_data = noisy_origin_data[:len(clean_data)]
        noisy_data,sr=librosa.load(noisy_file,sr=16000,mono=True)
        noisy_data = noisy_data[:len(clean_data)]
        draw = draw_magnitude_wav(noisy_data, clean_data, enh_data, 16000)
        print(draw)
        # noisy_data=noisy_data.to(device)
        # print("clean_data",clean_data.shape)
        # print("enh_data",enh_data.shape)
        clean_stoi = stoi(clean_data, clean_data, fs_sig=16000)
        noisy_stoi = stoi(clean_data, noisy_data, fs_sig=16000)
        enh_stoi = stoi(clean_data, enh_data, fs_sig=16000)
        silent_ref = np.zeros(16000)
        clean_pesq=pesq(16000,clean_data, clean_data,'wb')
        noisy_pesq = pesq( 16000,clean_data, noisy_data,'wb')
        enh_pesq = pesq(16000,clean_data, enh_data,'wb')

        noisy_csig, noisy_cbak, noisy_covl = pysepm.composite(clean_data, noisy_data, fs=16000)
        enh_csig,enh_cbak,enh_covl=pysepm.composite(clean_data,enh_data,fs=16000)
        # print(enh_csig,'\n',enh_cbak,'\n',enh_covl)

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

            noisytype
            + "  -->  "
            + " noisy pesq: "
            + str(Noisy_PESQ_mean)
            + "  --> "
            " enhanced pesq: "
            + str(Enhanced_PESQ_mean)
            + "  -->  "
            + "  noisy stoi:  "
            + str(Noisy_STOI_mean)
            + "  -->  "
            + " enhanced stoi "
            + "  -->  "
            + str(Enhanced_STOI_mean)
            + "  -->  "
            + " noisy csig "
            + "  -->  "
            + str(Noisy_CSIG_mean)
            + " enhanced csig "
            + "  -->  "
            + str(Enhanced_CSIG_mean)
            + "  -->  "
            + " noisy cbak "
            + str(Noisy_CBAK_mean)
            + "  -->  "
            + " enhanced cbak "
            + "  -->  "
            + str(Enhanced_CBAK_mean)
            + "  -->  "
            + " noisy covl "
            + str(Noisy_COVL_mean)
            + "  -->  "
            + " enhanced covl "
            + "  -->  "
            + str(Enhanced_COVL_mean)

    )

    fo = open("/root/autodl-tmp/MultiSEModel/save/log/test.txt", "a")
    fo.write(result + "\n")
    print(result)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # cleandir = "G:\\gzcm\\Dataset\\xbmuz_amdo\\wavesplit\\test\\clean\\"
    # noisydir = "G:\\gzcm\\Dataset\\xbmuz_amdo\\wavesplit\\test\\noisy\\"
    cleandir = "/root/autodl-tmp/gzcm/BodSpeBD/TEST/CLEAN/"
    noisydir = "/root/autodl-tmp/gzcm/BodSpeBD/TEST/NOISY/"
    noisy = os.listdir(noisydir)
    model =TSCNet()
    # model=CRN()
    # model = model.to(device)
    # print(model)
    # model.eval()
    model=torch.load("/root/autodl-tmp/MultiSEModel/save/model/model_153_0.1836.pth",map_location='cuda:0')
    # model.eval()
    for file in noisy:
        # print(123)
     val(cleandir, noisydir, file,model)


