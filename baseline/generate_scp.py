# Copyright (c) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Generate kaldi-like scp related files for crank

"""

import sys
import argparse
import random
import yaml
import logging
from pathlib import Path

####################################################
CONF_PATH = "config/config.yml"
####################################################

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) " "%(levelname)s: %(message)s",
)
def is_scp_exist(scpdir, name="wav"):
    scpdir = Path(scpdir)/"{}.scp".format(name)
    if scpdir.exists():
        return True
    else:
        return False

def generate_scp(save_dir, file_list, name="wav"):
    def write_lines(path, lines):
        with open(str(path), "a") as fp:
            for line in lines:
                fp.write("{}\n".format(line))

    wavscp = []
    for f in file_list:
        wavscp.append(f)

    save_dir.mkdir(parents=True, exist_ok=True)
    write_lines(save_dir / "{}.scp".format(name), wavscp)

def main(conf):

    norm_scpdir = Path(conf["path"]["norm_scp_dir"])

    norm_data_dir = Path(conf["path"]["norm_data_dir"])
    if is_scp_exist(norm_scpdir/"train", name="all_mels"):
        print("scp files already exists.")
        exit()
    '''
        if dataset_name == tr_data_name:
            spk_wav_dir = tr_data_dir / spk
        elif dataset_name == ev_data_name:
            spk_wav_dir = tst_data_dir / spk
        else:
            print("Error datset reading. Please check the name of the dataset.")
    '''
    for state in ["train", "eval"]:
        print("Generate scp for {} set.".format(state))
        spks = []
        for spk in (norm_data_dir / state).iterdir():
            spks.append(spk.name)
        for spk in spks:
            spk_scpdir = norm_scpdir / state / spk
            spk_norm_data_dir = norm_data_dir / state

            spk_mel_list = list((spk_norm_data_dir / spk).glob("*mel.npy"))

            generate_scp(spk_scpdir, spk_mel_list, name="mel")
            generate_scp(norm_scpdir / state, spk_mel_list, name="all_mels")


if __name__ == "__main__":
    with Path(CONF_PATH).open(mode="r") as conf_file:
        conf = yaml.safe_load(conf_file)
    main(conf)
