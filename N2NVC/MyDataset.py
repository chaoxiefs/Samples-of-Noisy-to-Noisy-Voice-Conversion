import numpy as np
import torch
from torch.utils.data import Dataset
from random import randint
from pathlib import Path
import yaml

def is_file_exist(file_path):
    if Path(file_path).exists():
        return True
    else:
        return False

def read_from_scp(scp_path):
    item_list = []
    with open(str(scp_path), "r") as scpfile:
        for line in scpfile:
            item_list.append(line.strip('\n'))   # delete '\n' in the end of line
    return item_list

class MyDataset(Dataset):
    # Use SCP of noisy dataset (_wav.npy)
    def __init__(self, conf, type = "train"):

        self.sample_frame = conf["data"]["sample_frames"]
        self.hop_len = conf["data"]["hop_len"]

        self.tr_noisy_data_name = conf["path"]["tr_noisy_data_name"]
        self.tr_clean_data_name = conf["path"]["tr_clean_data_name"]
        self.tr_noise_data_name = conf["path"]["tr_noise_data_name"]

        self.ev_noisy_data_name = conf["path"]["ev_noisy_data_name"]
        self.ev_clean_data_name = conf["path"]["ev_clean_data_name"]
        self.ev_noise_data_name = conf["path"]["ev_noise_data_name"]

        # Read speaker-id YML:
        spks_id_path = conf["path"]["spks_id_yml"]
        with Path(spks_id_path).open(mode="r") as conf_file:
            self.spks_id_yml = yaml.safe_load(conf_file)
        # Read wav scp list:
        norm_scp_dir = Path(conf["path"]["norm_scp_dir"])
        scp_file_path = norm_scp_dir / type / "all_wavs.scp"

        if is_file_exist(scp_file_path):
            self.wav_scp_list = read_from_scp(scp_file_path)
        else:
            print("Data scp file doesn't exit. Please check the scp file ")
            exit()
        self.scp_list_len = len(self.wav_scp_list)

    def __len__(self):
        return self.scp_list_len

    def __getitem__(self, index):
        wav_path = self.wav_scp_list[index]
        #print("wav_path: ", wav_path)
        main_dir = Path(wav_path).parents[2]
        dataset_name = Path(wav_path).parents[1].name
        speaker = Path(wav_path).parents[0].name

        file_main, _ = Path(wav_path).stem.rsplit(sep="_", maxsplit=1)
        #print("file main: ", file_main)

        mel_file_name = "de_{}_mel.npy".format(file_main)
        noise_file_name = Path(wav_path).name.replace("_wav.npy", "_noise.npy")

        if dataset_name == self.tr_noisy_data_name:
            mel_path = main_dir / self.tr_clean_data_name / speaker / mel_file_name
            noise_path = main_dir / self.tr_noise_data_name / speaker / noise_file_name

        elif dataset_name == self.ev_noisy_data_name:
            mel_path = main_dir / self.ev_clean_data_name / speaker / mel_file_name
            noise_path = main_dir / self.ev_noise_data_name / speaker / noise_file_name

        mel = np.load(mel_path)

        while mel.shape[-1] < self.sample_frame + 3:
            # print("Mel len shorter than sample frame, re-sampling.")
            # this is from random: range is [0,scp_list_len]
            new_index = randint(0, self.scp_list_len - 1)
            wav_path = self.wav_scp_list[new_index]
            main_dir = Path(wav_path).parents[2]
            dataset_name = Path(wav_path).parents[1].name
            speaker = Path(wav_path).parents[0].name

            file_main, _ = Path(wav_path).stem.rsplit(sep="_", maxsplit=1)

            mel_file_name = "{}_mel.npy".format(file_main)
            noise_file_name = Path(wav_path).name.replace("_wav.npy", "_noise.npy")

            if dataset_name == self.tr_noisy_data_name:
                mel_path = main_dir / self.tr_clean_data_name / speaker / mel_file_name
                noise_path = main_dir / self.tr_noise_data_name / speaker / noise_file_name

            elif dataset_name == self.ev_noisy_data_name:
                mel_path = main_dir / self.ev_clean_data_name / speaker / mel_file_name
                noise_path = main_dir / self.ev_noise_data_name / speaker / noise_file_name

            mel = np.load(mel_path)

        label = np.load(wav_path)
        noise = np.load(noise_path)

        #print("------------------------------------------------")
        #print("clean(mel): ", mel_path)
        #print("noisy(label): ", wav_path)
        #print("noise(wav): ", noise_path)

        pos = randint(1, mel.shape[-1] - self.sample_frame - 2)
        mel_cut = mel[:, pos - 1:pos + self.sample_frame + 1]
        #wav_cut = wav[pos * self.hop_len:(pos + self.sample_frame) * self.hop_len + 1]
        noise_cut = noise[pos * self.hop_len + 1 : (pos + self.sample_frame) * self.hop_len + 1]
        label_cut = label[pos * self.hop_len:(pos + self.sample_frame) * self.hop_len + 1]

        speaker_id = self.spks_id_yml[speaker]

        return torch.LongTensor(noise_cut), torch.LongTensor(label_cut),torch.FloatTensor(mel_cut), speaker_id
