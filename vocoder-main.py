from model.models import ForwardTransformer
from utils.training_config_manager import TrainingConfigManager
from data.audio import Audio
from scipy.io import wavfile
import numpy as np
from generator import Generator
import argparse
import time as t
import torch

# Control GPU Memory : gpus[number],  memory_limit=
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
     tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=16000)])

parser = argparse.ArgumentParser()
parser.add_argument('-sp', type = float, default = 1, help = 'Input speed')
parser.add_argument('-t', type = str, default = '請輸入中文句子', help = 'Input text')
parser.add_argument('-ar', type = bool, default = False, help = 'Input True or False to use AR model')
args = parser.parse_args()

##################### text conversion #####################
temp = []
for i in args.t:
    temp.append(i)
args.t = ""
for i in range(len(temp)):
    if temp[i] == "奇":
        if temp[i+1] == "數":
            temp[i] = "雞"
        elif temp[i+1] == "偶":
            if temp[i+2] == "數":
                temp[i] = "雞"
    if temp[i] == "澄":
        temp[i]=  "成"
    if temp[i] == "放":
        if temp[i+1] == "假":
            temp[i+1] = "價"
    if temp[i] == "幾":
        if temp[i+1] == "乎":
            temp[i] = "雞"
    if temp[i] == "暖":
        if temp[i+1] == "和":
            temp[i+1] = "活"
    if temp[i] == "著":
        if temp[i+1] == "急":
            temp[i] = "昭"
        elif temp[i+1] == "名" or temp[i+1] == "作":
            temp[i] = "住"

###########################
for i in range(len(temp)):
    args.t = args.t + temp[i]
    print(args.t)
print(args.t)

## Load MelGAN
vocoder = torch.hub.load('../melgan-train2', 'melgan', source='local')
vocoder.eval()

# Autoregressive Transformer TTS predict
if args.ar == True:
    config_loader = TrainingConfigManager(config_path='./config/training_config.yaml', aligner=True)
    AR_model = config_loader.load_model()
    audio = Audio.from_config(config_loader.config)
    print("----------------------start----------------------")
    start = t.time()
    AR_out = AR_model.predict(args.t)
    end = t.time()
    print(end - start, "s for AR predict mel", AR_out["mel"].shape[0], "length")
    AR_wav = audio.reconstruct_waveform(AR_out['mel'].numpy().T, n_iter=60)
    wavfile.write("./AR_grif.wav", 22050, AR_wav)

# FFT predict
if args.ar == False:
    FFT_model = ForwardTransformer.load_model('./logdir/bznsyp/tts_swap_conv_dims.alinger_extralayer_layernorm/weights/step_85000/')
    audio = Audio.from_config(FFT_model.config)
    print("----------------------start----------------------") 
    start = t.time()
    FFT_out = FFT_model.predict(args.t, speed_regulator = args.sp)
    end = t.time()
    print(end - start, "s for FFT predict mel", FFT_out["mel"].shape[0], "length")
    FFT_wav = audio.reconstruct_waveform(FFT_out['mel'].numpy().T, n_iter=60)          
    wavfile.write("./FFT_grif.wav", 22050, FFT_wav)

####################################################### MelGAN
## AR model
if args.ar == True:
    AR_mel = torch.tensor(AR_out['mel'].numpy().T[np.newaxis,:,:])
    if torch.cuda.is_available():
        vocoder = vocoder.cuda()
        AR_mel = AR_mel.cuda()
    with torch.no_grad():
        AR_speech = vocoder.inference(AR_mel)

    wavfile.write("./AR_melgan.wav", 22050, AR_speech.cpu().numpy())
    ete = t.time()
    print(ete - start, "s for AR End-to-End synthesis")

## FFT model
if args.ar == False:
    FFT_mel = torch.tensor(FFT_out['mel'].numpy().T[np.newaxis,:,:])
    if torch.cuda.is_available():
        vocoder = vocoder.cuda()
        FFT_mel = FFT_mel.cuda()
    with torch.no_grad():
        FFT_speech = vocoder.inference(FFT_mel)

    wavfile.write("./FFT_melgan.wav", 22050, FFT_speech.cpu().numpy())
    ete = t.time()
    print((ete - start), "s for FFT End-to-End synthesis")
