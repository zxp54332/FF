import socket
import os
import sys
import numpy as np
import time
from model.models import ForwardTransformer
from utils.training_config_manager import TrainingConfigManager
from data.audio import Audio
from scipy.io import wavfile
from generator import Generator
import argparse
import time as t
import torch
import tensorflow as tf

# Control GPU Memory : gpus[number],  memory_limit=
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
         tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=16000)])


def socket_service(FFT_model, AR_model, vocoder):
    try:
        socket.setdefaulttimeout(20)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('***.***.**.***', 4000))    # Enter your IP
    except ConnectionResetError:
            print("==> ConnectionResetError")
            pass
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print ('Socket bind complete')
    s.listen(10)
    print ('Socket now listening')
    print("Wait for Connection.....................")
    while True:
        sock, addr = s.accept()  #addr是一个元组(ip,port)
        print("Accept connection from {}".format(addr))  #查看发送端的ip和端口
        text = str(sock.recv(2000), encoding="utf-8")
        if not text:
            pass
        else:
            sp = float(str(sock.recv(4),encoding="utf-8"))
            model = str(sock.recv(100), encoding="utf-8")
            text = text_normalization(text)
            if model == "FFT":
                FFT_synthesis(text, FFT_model, vocoder, sp)
                with open("./FFT_melgan.wav", 'rb') as f:
                    for data in f:
                        sock.send(data)
            elif model == "TransformerTTS":
                AR_synthesis(text, AR_model, vocoder)
                with open("./AR_melgan.wav", 'rb') as f:
                    for data in f:
                        sock.send(data)
        sock.close()


def text_normalization(text):
    temp = []
    for i in text:
            temp.append(i)
    text = ""
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
        if temp[i] == "音":
            if temp[i+1] == "樂":
                temp[i+1] = "月"
        if temp[i] == "樂":
            if temp[i+1] == "器":
                temp[i] = "月"
#---------------------------------------------------
        if temp[i] == "0":
            temp[i] = "零"
        if temp[i] == "1":
            temp[i] = "一"
        if temp[i] == "2":
            temp[i] = "二"
        if temp[i] == "3":
            temp[i] = "三"
        if temp[i] == "4":
            temp[i] = "四"
        if temp[i] == "5":
            temp[i] = "五"
        if temp[i] == "6":
            temp[i] = "六"
        if temp[i] == "7":
            temp[i] = "七"
        if temp[i] == "8":
            temp[i] = "八"
        if temp[i] == "9":
            temp[i] = "九"

    for i in range(len(temp)):
        text = text + temp[i]
    return text

def FFT_synthesis(text, FFT_model, vocoder, sp):
    print(text, sp, FFT_model)
    print("----------------------start----------------------")
    start = t.time()
    FFT_out = FFT_model.predict(text, speed_regulator = sp)
    FFT_mel = torch.tensor(FFT_out['mel'].numpy().T[np.newaxis,:,:])
    if torch.cuda.is_available():
        vocoder = vocoder.cuda()
        FFT_mel = FFT_mel.cuda()
    with torch.no_grad():
        FFT_speech = vocoder.inference(FFT_mel)

    wavfile.write("./FFT_melgan.wav", 22050, FFT_speech.cpu().numpy())
    ete = t.time()
    print((ete - start), "s for FFT End-to-End synthesis", FFT_out["mel"].shape[0], "length")


def AR_synthesis(text, AR_model, vocoder):
    print(text, AR_model)
    print("----------------------start----------------------")
    start = t.time()
    AR_out = AR_model.predict(text)
    AR_mel = torch.tensor(AR_out['mel'].numpy().T[np.newaxis,:,:])
    if torch.cuda.is_available():
        vocoder = vocoder.cuda()
        AR_mel = AR_mel.cuda()
    with torch.no_grad():
        AR_speech = vocoder.inference(AR_mel)

    wavfile.write("./AR_melgan.wav", 22050, AR_speech.cpu().numpy())
    ete = t.time()
    print(ete - start, "s for AR End-to-End synthesis", AR_out["mel"].shape[0], "length")


if __name__=="__main__":
    #load model
    FFT_model = ForwardTransformer.load_model('./logdir/bznsyp/tts_swap_conv_dims.alinger_extralayer_layernorm/weights/step_85000/')
    config_loader = TrainingConfigManager(config_path='./config/training_config.yaml', aligner=True)
    AR_model = config_loader.load_model()
    vocoder = torch.hub.load('../melgan-train2', 'melgan', source='local')
    vocoder.eval()
    # create socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print ('Socket created')
    socket_service(FFT_model, AR_model, vocoder)
