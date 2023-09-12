from pathlib import Path
import librosa
import scipy
import json
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import random
import yaml

####################################################
CONF_PATH = "config/config.yml"
RANDOM_SEED = 1277
####################################################

def read_from_scp(scp_path):
    item_list = []
    with open(str(scp_path), "r") as scpfile:
        for line in scpfile:
            item_list.append(line.strip('\n'))   # delete '\n' in the end of line
    return item_list


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


def process_mel(wav_path, out_path, sr=16000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=160,
                win_length=400, fmin=50, fmax = 4000, top_db=80, bits=9, offset=0.0, duration=None):
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

    #wav = mulaw_encode(wav, mu=2**bits)

    #np.save(out_path / (wav_path.name.replace(".wav", "_wav.npy")), wav)
    np.save(out_path / (wav_path.name.replace(".wav", "_mel.npy")), logmel)
    #np.save(out_path.with_suffix(".wav.npy"), wav)
    #np.save(out_path.with_suffix(".mel.npy"), logmel)

def preprocess_noise(wav_path, out_path, sr=8000, bits=8, offset=0.0, duration=None):
    wav, _ = librosa.load(wav_path, sr=sr,
                          offset=offset, duration=duration)
    wav = mulaw_encode(wav, mu=2 ** bits)
    np.save(out_path / (wav_path.name.replace(".wav", "_noise.npy")), wav)


def preprocess_wav(wav_path, out_path, sr=8000, bits=8, offset=0.0, duration=None):
    wav, _ = librosa.load(wav_path, sr=sr,
                          offset=offset, duration=duration)
    wav = mulaw_encode(wav, mu=2 ** bits)
    np.save(out_path / (wav_path.name.replace(".wav", "_wav.npy")), wav)


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
    tr_noisy_data_name = conf["path"]["tr_noisy_data_name"]
    tr_clean_data_name = conf["path"]["tr_clean_data_name"]
    tr_noise_data_name = conf["path"]["tr_noise_data_name"]

    ev_noisy_data_name = conf["path"]["ev_noisy_data_name"]
    ev_clean_data_name = conf["path"]["ev_clean_data_name"]
    ev_noise_data_name = conf["path"]["ev_noise_data_name"]

    # Generate yml for speakers--ID pairs:
    print("Generating speakers--ID pairs in YML")
    vcc_spks = []
    for spk in Path(dataset_dir / tr_noisy_data_name).iterdir():
        vcc_spks.append(spk.name)

    spks_id_yml = conf["path"]["spks_id_yml"]
    spks_id_dict = {}
    for spk in vcc_spks:
        spks_id_dict[spk] = vcc_spks.index(spk)

    with open(spks_id_yml, "w") as f:
        yaml.dump(spks_id_dict, f)

    print("Speakers--ID pairs YML completes")

    #filter_num = conf["data"]["filter_num"]
    num_speakers = conf["data"]["n_speakers"]

    out_dir = Path(conf["path"]["norm_data_dir"])


    '''
    all_speakers = []
    for spk in dataset_dir.iterdir():
        all_speakers.append(spk.name)

    filtered_speakers = []
    for speaker in all_speakers:
        wav_list = read_from_scp(dataset_dir / speaker / "spk_wavs.scp")
        if len(wav_list) >= filter_num:
            filtered_speakers.append(speaker)

    if len(filtered_speakers) < num_speakers:
        print("Number of fitered speakers should be larger than the num_speakers. Change num_speakers or filter_num.")
        exit()

    sampled_speakers = random.sample(filtered_speakers, num_speakers)
    # Generate yml for speakers--ID pairs:
    
    print("Generating speakers--ID pairs in YML")
    spks_id_yml = conf["path"]["spks_id_yml"]
    spks_id_dict = {}
    for spk in sampled_speakers:
        spks_id_dict[spk] = sampled_speakers.index(spk)
    with open(spks_id_yml, "w") as f:
        yaml.dump(spks_id_dict, f)

    print("Speakers--ID pairs YML completes")
    '''
    #executor = ProcessPoolExecutor(max_workers=cpu_count())

    # Pre-process the wav/mel data :

    if len(vcc_spks) != num_speakers:
        print("Speakers number can not match the one in config.")
        exit()

    idx_files = 1
    for dataset_name in [tr_noisy_data_name, tr_clean_data_name, tr_noise_data_name,
                         ev_noisy_data_name, ev_clean_data_name, ev_noise_data_name]:
        print("Extracting features for {} set".format(dataset_name))
        for spk in tqdm(vcc_spks):
            wav_list = list(Path(dataset_dir / dataset_name / spk).glob("*.wav"))
            save_dir = out_dir / dataset_name / spk
            save_dir.mkdir(exist_ok=True, parents=True)
            for wav_path in wav_list:
                if dataset_name == tr_noisy_data_name or dataset_name == ev_noisy_data_name:
                    preprocess_wav(wav_path, save_dir, sr=FS, bits=BITS)
                elif dataset_name == tr_noise_data_name or dataset_name == ev_noise_data_name:
                    preprocess_noise(wav_path, save_dir, sr=FS, bits=BITS)
                elif dataset_name == tr_clean_data_name or dataset_name == ev_clean_data_name:
                    process_mel(wav_path, save_dir,
                                sr=FS,
                                preemph=0.97,
                                n_fft=FFTL,
                                n_mels=NUM_MEL,
                                hop_length=HOP_LEN,
                                win_length=WIN_LEN,
                                fmin=FMIN,
                                fmax=FMAX,
                                top_db=80,
                                bits=BITS)
                else:
                    print("Wrong dataset which is not mentioned in config: ", dataset_name)
                    exit()

                idx_files +=1

    print("Processed {} utterances in total.".format(idx_files))


if __name__ == "__main__":
    preprocess_dataset(CONF_PATH)
