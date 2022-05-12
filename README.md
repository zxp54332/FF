<h2 align="center">
<p>Transformer-Based End-to-End Feed-Forward Neural Chinese Speech Synthesis</p>
</h2>

The text input by the client is sent to the server to synthesize speech and return, and the server and client are connected through socket to transmit data.
(https://www.youtube.com/watch?v=kqExk3rd_wQ)

At present, the problem of synthesizing polyphonic words is encountered, and the International Phonetic Alphabet is used as data, which can only be judged and modified manually during synthesis.

★ Because there is a lack of part of the MelGAN vocoder code, it can only be synthesized in the Griffin Lim method. The program code is for reference only.

MelGAN vocoder is trained on the BZNSYP corpus
- [MelGAN](https://github.com/seungwonpark/melgan)

## Installation

Make sure you have:

* Python >= 3.6

Install espeak as phonemizer backend:
```
sudo apt-get install espeak-ng
```

Then install the rest with pip:
```
pip install -r requirements.txt
```
## Custom dataset
Prepare a folder containing your metadata and wav files
```
|- dataset_folder/
|   |- metadata.csv
|   |- wavs/
|       |- file1.wav
|       |- ...
```
if `metadata.csv` has the following format you can use the bznsyp preprocessor in `data/metadata_readers.py`, otherwise add your own under the same file.
``` wav_file_name|transcription ```

Make sure that:
 -  the metadata reader function name is the same as ```data_name``` field in ```training_config.yaml```.
 -  the metadata file (can be anything) is specified under ```metadata_path``` in ```training_config.yaml``` 

## Training
Change the ```--config``` argument based on the configuration of your choice.
### Train Aligner Model
#### Create training dataset
```bash
python create_training_data.py --config config/training_config.yaml
```
This will populate the training data directory (default `transformer_tts_data.bznsyp`).
#### Training
```bash
python train_aligner.py --config config/training_config.yaml
```
### Train TTS Model
#### Compute alignment dataset
Use the aligner model to create the durations dataset
```bash
python extract_durations.py --config config/training_config.yaml
```
this will add the `durations.<session name>` as well as the char-wise pitch folders to the training data directory.
#### Training
```bash
python train_tts.py --config config/training_config.yaml
```
#### Training & Model configuration
- Training and model settings can be configured in `training_config.yaml`

#### Resume or restart training
- To resume training simply use the same configuration files
- To restart training, delete the weights and/or the logs from the logs folder with the training flag `--reset_dir` (both) or `--reset_logs`, `--reset_weights`

#### Monitor training
```bash
tensorboard --logdir /logs/directory/
```

## Prediction

In a python script
```python
from data.audio import Audio
from model.models import ForwardTransformer
from utils.training_config_manager import TrainingConfigManager

audio = Audio.from_config(model.config)

# Feed Forward
FF_model = ForwardTransformer.load_model('/path/to/weights/')
FF_out = FF_model.predict('Please, say something.')

# Autoregressive
AR_model = config_loader.load_model()
AR_out = AR_model.predict('Please, say something.')

# Convert spectrogram to wav (with griffin lim)
FF_wav = audio.reconstruct_waveform(FF_out['mel'].numpy().T)
AR_wav = audio.reconstruct_waveform(AR_out['mel'].numpy().T)
```
