from pathlib import Path
import librosa
import scipy
import random
import json
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import yaml

####################################################
CONF_PATH = "config/config.yml"
####################################################

def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def process_wav(wav_path, out_path, sr=16000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=160,
                win_length=640, fmin=50, fmax = 4000, top_db=80, bits=8, offset=0.0, duration=None):
    wav, _ = librosa.load(wav_path, sr=sr,
                          offset=offset, duration=duration)
    #wav = wav / np.abs(wav).max() * 0.999

    mel = librosa.feature.melspectrogram(y=preemphasis(wav, preemph),
                                         sr=sr,
                                         n_fft=n_fft,
                                         n_mels=n_mels,
                                         hop_length=hop_length,
                                         win_length=win_length,
                                         fmin=fmin,
                                         #fmax=fmax,
                                         power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=top_db)   # calculate log mel
    logmel = logmel / top_db + 1

    wav = mulaw_encode(wav, mu=2**bits)

    np.save(out_path / (wav_path.name.replace(".wav", "_wav.npy")), wav)
    np.save(out_path / (wav_path.name.replace(".wav", "_mel.npy")), logmel)
    #np.save(out_path.with_suffix(".wav.npy"), wav)
    #np.save(out_path.with_suffix(".mel.npy"), logmel)
    #return out_path, logmel.shape[-1]


def preprocess_dataset(conf_path):
    with Path(conf_path).open(mode="r") as conf_file:
        conf = yaml.safe_load(conf_file)
    FS = conf["data"]["sample_rate"]
    BITS = conf["model"]["decoder"]["bits"]
    FFTL = conf["data"]["fftl"]
    NUM_MEL = conf["data"]["mel_dim"]
    HOP_LEN = conf["data"]["hop_len"]
    WIN_LEN = conf["data"]["win_len"]
    FMIN = conf["data"]["fmin"]
    FMAX = conf["data"]["fmax"]

    dataset_dir = Path(conf["path"]["dataset_dir"])
    tr_data_name = conf["path"]["train_data_name"]
    ev_data_name = conf["path"]["eval_data_name"]


    tr_data_dir = dataset_dir / tr_data_name
    ev_data_dir = dataset_dir / ev_data_name


    vcc_spks = []
    for spk in Path(tr_data_dir).iterdir():
        vcc_spks.append(spk.name)

    out_dir = Path(conf["path"]["norm_data_dir"])

    #executor = ProcessPoolExecutor(max_workers=cpu_count())

    # Generate yml for speakers--ID pairs:
    print("Generating speakers--ID pairs in YML")
    spks_id_yml = conf["path"]["spks_id_yml"]
    spks_id_dict = {}
    for spk in vcc_spks:
        spks_id_dict[spk] = vcc_spks.index(spk)

    with open(spks_id_yml, "w") as f:
        yaml.dump(spks_id_dict, f)

    print("Speakers--ID pairs YML completes")
    # Pre-process the wav/mel data :
    for dataset_name in [tr_data_name, ev_data_name]:
        idx = 1
        print("Extracting features for {} set".format(dataset_name))
        for spk in vcc_spks:
            if dataset_name == tr_data_name:
                spk_wav_dir = tr_data_dir / spk
                save_dir = out_dir/"train"
            elif dataset_name == ev_data_name:
                spk_wav_dir = ev_data_dir / spk
                save_dir = out_dir / "eval"
            else:
                print("Error datset reading. Please check the name of the dataset.")

            spk_save_dir = save_dir / spk
            spk_save_dir.mkdir(parents=True, exist_ok=True)

            spk_wav_list = list(Path(spk_wav_dir).glob("*.wav"))
            for wav in spk_wav_list:
                process_wav(
                    wav_path=wav,
                    out_path=spk_save_dir,
                    sr=FS,
                    n_fft=FFTL,
                    n_mels=NUM_MEL,
                    hop_length=HOP_LEN,
                    win_length=WIN_LEN,
                    fmin=FMIN,
                    fmax=FMAX,
                    bits=BITS
                )
                idx+=1
        print("Processed {} utterances in total.".format(idx))

if __name__ == "__main__":
    preprocess_dataset(CONF_PATH)
