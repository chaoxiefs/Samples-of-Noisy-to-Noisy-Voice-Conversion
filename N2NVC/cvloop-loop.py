
from pathlib import Path
import torch
import os
import numpy as np
import librosa
import pyloudnorm
import yaml
import soundfile as sf
from preprocess import preemphasis, mulaw_encode
from model import Encoder, Decoder

####################################################
CONF_PATH = "config/config.yml"
SOURCE_SPEAKERS = ["VCC2SF3","VCC2SF4","VCC2SM3","VCC2SM4"]
TARGET_SPEAKERS = ["VCC2TM2","VCC2TF2"]
#manual_gpu_id = "5"
start_checkpoint_steps = 200000
end_checkpoint_steps = 400000
per_step = 25000 # 5000 for minimum unit
num_uttr = None # "None" for all
use_noise = False
####################################################
#os.environ["CUDA_VISIBLE_DEVICES"] = manual_gpu_id

def convert(conf_path):
    with Path(conf_path).open(mode="r") as conf_file:
        conf = yaml.safe_load(conf_file)

    print("Which config: ", conf_path)
    print("Which dataset: ", conf["path"]["tst_clean_data_name"])

    dataset_path = Path(conf["path"]["tst_dataset_dir"]) / conf["path"]["tst_clean_data_name"]
    #noise_dataset_name = Path(conf["path"]["dataset_dir"]) / conf["path"]["ev_noise_data_name"]
    noise_dataset_name = conf["path"]["tst_noise_data_name"]

    with open(conf["path"]["spks_id_yml"]) as spks_id_file:
        spks_id_dict = yaml.safe_load(spks_id_file)
    if use_noise:
        convert_dir = Path(
            "{}_{}".format(conf["path"]["convert_save_dir"], Path(conf["path"]["tst_dataset_dir"]).name)) / "use-noise"
    else:
        convert_dir = Path("{}_{}".format(conf["path"]["convert_save_dir"],
                                          Path(conf["path"]["tst_dataset_dir"]).name)) / "clean"  # / "loop-for-MCD"

    checkpoints_dir = Path(conf["path"]["exp_dir"]) / "checkpoints"
    '''
    gpu_id = str(conf["training"]["gpu_id"])
    
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # for mannual set:
    os.environ["CUDA_VISIBLE_DEVICES"] = manual_gpu_id
    '''
    # "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device("cuda") # if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**conf["model"]["encoder"])
    decoder = Decoder(**conf["model"]["decoder"])
    encoder.to(device)
    decoder.to(device)

    times = int((end_checkpoint_steps - start_checkpoint_steps) / per_step)
    for time in range(times + 1):
        checkpoint_steps = int(start_checkpoint_steps + time * per_step)
        chosen_checkpoint = "model.ckpt-{}.pt".format(checkpoint_steps)
        #print("Load checkpoint from: {}:".format(checkpoints_dir))

        checkpoint = torch.load(checkpoints_dir / chosen_checkpoint, map_location=lambda storage, loc: storage)
        #print("Use manual checkpoint: ", manual_checkpoint)

        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])

        encoder.eval()
        decoder.eval()

        #meter = pyloudnorm.Meter(conf["data"]["sample_rate"])

        for s_spk in SOURCE_SPEAKERS:
            for t_spk in TARGET_SPEAKERS:
                print("Speaker: ", s_spk, " ID: ", spks_id_dict[s_spk])

                spk_wav_dir = dataset_path / s_spk
                if num_uttr is None:
                    wav_list = list(spk_wav_dir.glob("*.wav"))
                    print("len of wav_list: ", len(wav_list))
                elif type(num_uttr) == int:
                    wav_list = list(spk_wav_dir.glob("*.wav"))[:num_uttr]
                    print("len of wav_list: ", len(wav_list))
                else:
                    print("Wrong uttr numbers: ", num_uttr)
                    exit()
                #print(wav_list)
                save_convert_dir = convert_dir / "steps-{}".format(checkpoint_steps) / "T_{}".format(t_spk) / s_spk
                save_convert_dir.mkdir(parents=True, exist_ok=True)
                for wav_path in wav_list:
                    wav, _ = librosa.load(wav_path, sr=conf["data"]["sample_rate"])

                    #ref_loudness = meter.integrated_loudness(wav)
                    #wav = wav / np.abs(wav).max() * 0.999
                    #peak = np.abs(wav).max()
                    #if peak > 1.0:
                        #wav /= peak
                    top_db = 80
                    mel = librosa.feature.melspectrogram(
                        preemphasis(wav, preemph=0.97),
                        sr=conf["data"]["sample_rate"],
                        n_fft=conf["data"]["fftl"],
                        n_mels=conf["data"]["mel_dim"],
                        hop_length=conf["data"]["hop_len"],
                        win_length=conf["data"]["win_len"],
                        fmin=conf["data"]["fmin"],
                        fmax=conf["data"]["fmax"],
                        power=1)
                    logmel = librosa.amplitude_to_db(mel, top_db=top_db)
                    logmel = logmel / top_db + 1

                    mel = torch.FloatTensor(logmel).unsqueeze(0).to(device)
                    #print("len of spks id dict: ", len(spks_id_dict))
                    print("target spk: ", t_spk)
                    target_spk_id = spks_id_dict[t_spk]
                    input_spk_id = torch.LongTensor([target_spk_id]).to(device)

                    if use_noise:
                        # print(wav_path)
                        # print(wav_path.parents)
                        # print(wav_path.parents[0])
                        _, file_name = str(wav_path.name).split("_", 1)

                        noise_path = wav_path.parents[2] / noise_dataset_name / wav_path.parent.name / file_name

                        noise_wav, _ = librosa.load(noise_path, sr=conf["data"]["sample_rate"])
                        if mel.shape[-1] % 2 == 0:
                            gen_wav_len = mel.shape[-1] * conf["data"]["hop_len"]
                        else:
                            gen_wav_len = (mel.shape[-1] - 1) * conf["data"]["hop_len"]

                        gen_wav_len = mel.shape[-1] * conf["data"]["hop_len"]

                        if len(noise_wav) > gen_wav_len:
                            gap = len(noise_wav) - gen_wav_len
                            pad = int(gap / 2)

                            left_pad = pad
                            right_pad = pad + gap - (int(gap / 2) * 2)

                            noise_wav = noise_wav[left_pad:-right_pad-1]
                            noise_wav = np.concatenate([[0], noise_wav])

                        elif len(noise_wav) < gen_wav_len:
                            gap = gen_wav_len - len(noise_wav)
                            pad = int(gap / 2)

                            left_pad = pad
                            right_pad = pad + gap - (int(gap / 2) * 2)
                            noise_wav = np.pad(noise_wav, (left_pad, right_pad), 'constant')

                        noise_wav = mulaw_encode(noise_wav, mu=2 ** int(conf["model"]["decoder"]["bits"]))
                        noise_wav = torch.from_numpy(noise_wav)
                        noise_wav = noise_wav.long()
                        noise_wav = noise_wav.to(device)

                    else:
                        if mel.shape[-1] % 2 == 0:
                            gen_wav_len = mel.shape[-1] * conf["data"]["hop_len"]
                        else:
                            gen_wav_len = (mel.shape[-1] - 1) * conf["data"]["hop_len"]

                        gen_wav_len = mel.shape[-1] * conf["data"]["hop_len"]

                        noise_wav = torch.ones((gen_wav_len,), dtype=torch.int64) * int((2 ** int(conf["model"]["decoder"]["bits"]))/2)
                        noise_wav = noise_wav.to(device)

                    with torch.no_grad():
                        #print("mel shape: ", mel.shape)
                        #print("supposed wav len: ", mel.shape[-1]*conf["data"]["hop_len"])
                        #print("noise wav shape: ",noise_wav.shape)
                        z, _ = encoder.encode(mel)
                        output = decoder.generate(z, noise_wav, input_spk_id)
                    #output_loudness = meter.integrated_loudness(output)
                    #output = pyloudnorm.normalize.loudness(output, output_loudness, ref_loudness)
                    #output.astype(np.float32)
                    converted_wav_path = save_convert_dir / wav_path.name
                    sf.write(converted_wav_path, output.astype(np.float32), samplerate=conf["data"]["sample_rate"])
                    #sf.write(converted_wav_path, output, samplerate=conf["data"]["sample_rate"])

if __name__ == "__main__":
    convert(CONF_PATH)
