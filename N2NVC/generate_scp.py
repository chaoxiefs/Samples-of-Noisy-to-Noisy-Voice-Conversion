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
# Generate scp for noisy wav npy.
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
    #print("file list: ", file_list)
    wavscp = []
    for f in file_list:
        wavscp.append(f)
    #print("len of wavscp: ", len(wavscp))
    save_dir.mkdir(parents=True, exist_ok=True)
    write_lines(save_dir / "{}.scp".format(name), wavscp)

def main(conf):
    # generate SCP for noisy dataset (_wav.npy)
    norm_scpdir = Path(conf["path"]["norm_scp_dir"])
    norm_data_dir = Path(conf["path"]["norm_data_dir"])

    tr_noisy_data_name = conf["path"]["tr_noisy_data_name"]
    ev_noisy_data_name = conf["path"]["ev_noisy_data_name"]

    #clean_data_name = conf["path"]["clean_data_name"]
    #noise_data_name = conf["path"]["noise_data_name"]

    spks_id_path = conf["path"]["spks_id_yml"]
    with Path(spks_id_path).open(mode="r") as conf_file:
        sampled_speakers = yaml.safe_load(conf_file)

    if is_scp_exist(norm_scpdir/ tr_noisy_data_name, name="all_wavs") is False:

        spk_id =1
        for dataset in [tr_noisy_data_name, ev_noisy_data_name]:
            if dataset == tr_noisy_data_name:
                scp_save = norm_scpdir / "train"
            elif dataset == ev_noisy_data_name:
                scp_save = norm_scpdir / "eval"
            else:
                print("Wrong dataset name {}.".format(dataset))
                exit()

            for spk in sampled_speakers:
                print("No.{} Speaker: {}".format(spk_id, spk))
                spk_id+=1

                all_wav_list = list((norm_data_dir/ dataset / spk).glob("*wav.npy"))

                generate_scp(scp_save / spk, all_wav_list, name="wav")

                generate_scp(scp_save, all_wav_list, name="all_wavs")

    else:
        print("wav.scp already exists. Please delete orig scp file if want to re-generate")


if __name__ == "__main__":
    with Path(CONF_PATH).open(mode="r") as conf_file:
        conf = yaml.safe_load(conf_file)
    main(conf)
