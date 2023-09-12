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
    def __init__(self, conf, type = "train"):
        self.sample_frame = conf["data"]["sample_frames"]
        self.hop_len = conf["data"]["hop_len"]
        # Read speaker-id YML:
        spks_id_path = conf["path"]["spks_id_yml"]
        with Path(spks_id_path).open(mode="r") as conf_file:
            self.spks_id_yml = yaml.safe_load(conf_file)
        # Read mel scp list:
        norm_scp_dir = Path(conf["path"]["norm_scp_dir"])
        if type == "train":
            dataset_name = "train"
        elif type == "eval":
            dataset_name = "eval"
        else:
            print("Key error. Now type is {} but required [train, eval]".format(type))
            dataset_name = None
            exit()
        scp_file_path = norm_scp_dir / dataset_name / "all_mels.scp"
        if is_file_exist(scp_file_path):
            self.mel_scp_list = read_from_scp(scp_file_path)
        else:
            print("Data scp file doesn't exit. Please check the scp file ")
            exit()
        self.scp_list_len = len(self.mel_scp_list)

    def __len__(self):
        return self.scp_list_len

    def __getitem__(self, index):
        mel_path = self.mel_scp_list[index]
        mel = np.load(mel_path)
        wav_path = self.mel_scp_list[index].replace("_mel.npy", "_wav.npy")
        while mel.shape[-1] < self.sample_frame+ 3:
            #print("Mel len shorter than sample frame, re-sampling.")
            # this is from random: range is [0,scp_list_len]
            new_index = randint(0, self.scp_list_len-1)
            mel_path = self.mel_scp_list[new_index]
            mel = np.load(mel_path)
            wav_path = self.mel_scp_list[new_index].replace("_mel.npy", "_wav.npy")

        wav = np.load(wav_path)
        pos = randint(1, mel.shape[-1] - self.sample_frame - 2)
        mel_cut = mel[:, pos - 1:pos + self.sample_frame + 1]
        wav_cut = wav[pos * self.hop_len:(pos + self.sample_frame) * self.hop_len + 1]

        speaker_name = Path(wav_path).parent.name
        speaker_id = self.spks_id_yml[speaker_name]

        return torch.LongTensor(wav_cut), torch.FloatTensor(mel_cut), speaker_id
