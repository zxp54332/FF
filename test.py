from model.models import ForwardTransformer
from data.audio import Audio
from scipy.io import wavfile
import numpy as np
from generator import Generator
import argparse
import librosa
import getch
import sys
import numpy as np

# Control GPU Memory : gpus[number],  memory_limit=
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
     tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])

parser = argparse.ArgumentParser()
parser.add_argument('-sp', type = float, default = 1, help = 'Input speed')
parser.add_argument('-t', type = str, default = '請輸入中文句子', help = 'Input text')
args = parser.parse_args()

model = ForwardTransformer.load_model('./logdir/bznsyp/tts_swap_conv_dims.alinger_extralayer_layernorm/weights/step_85000/')
audio = Audio.from_config(model.config)
out = model.predict(args.t, speed_regulator = args.sp)
print(out['mel'].numpy().T.shape)
# Convert spectrogram to wav (with griffin lim)
wav = audio.reconstruct_waveform(out['mel'].numpy().T)
wavfile.write("./griffim.wav", 22050, wav)

import torch
melsp = torch.load("../melgan-train2//BZNSYP/testtt/p258.mel")
melsp = melsp.squeeze().numpy()  # (1, 80, t) -> (80, t))
print(melsp.shape)

wav = audio.reconstruct_waveform(melsp, n_iter=100)
wavfile.write("./griffim.wav", 22050, wav)

####################################################### MelGAN

# Set up the paths
from pathlib import Path
import sys
import torch
import numpy as np

#MelGAN_path = '../melgan/'

#sys.modules.pop('model')
#sys.path.append(MelGAN_path)

vocoder = torch.hub.load('../melgan-train2', 'melgan', source='local')
vocoder.eval()
mel = torch.tensor(melsp[np.newaxis, :, :])

if torch.cuda.is_available():
    vocoder = vocoder.cuda()
    mel = mel.cuda()

with torch.no_grad():
    speech = vocoder.inference(mel)

wavfile.write("./melgan.wav", 22050, speech.cpu().numpy())

