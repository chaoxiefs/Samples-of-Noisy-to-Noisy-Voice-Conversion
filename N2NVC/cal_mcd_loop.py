from pathlib import Path
import soundfile as sf
import pyworld as pw
import numpy as np
import matplotlib.pyplot as plt
#New:
import pysptk
import yaml
from fastdtw import fastdtw
from scipy import spatial


#########################################################################
VC_DATASET_DIR = Path("convert_esc50-all")

REF_DIR = Path("/esc50-all/norm_vcc2018_eval_16k_snr5")

MCD_SAVE_DIR = Path("MCD_all")

VC_SPEAKERS = ["VCC2SF3", "VCC2SF4", "VCC2SM3", "VCC2SM4"]
TARGET_SPEAKERS = ["VCC2TF2", "VCC2TM2"]
FS = 16000
ALPHA = 0.41
##########################################################################
F0_CONF_PATH = "config/f0-clean.yml"


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

def mcep(wav_path, fs=FS, fftl=1024, shiftms=5.0, minf0=40.0, maxf0=500.0, dim=24, alpha=ALPHA):
        """Return mel-cepstrum sequence parameterized from spectral envelope
        -------
        mcep : array, shape (`T`, `dim + 1`)
            Mel-cepstrum sequence
        """
        x, _ = sf.read(wav_path)
        f0, time_axis = pw.harvest(x, fs, f0_floor=minf0,
                                   f0_ceil=maxf0, frame_period=shiftms)
        spc = pw.cheaptrick(x, f0, time_axis, fs,
                            fft_size=fftl)

        return pysptk.sp2mc(spc, dim, alpha), f0

def calculate_mcd(vc_wav_path, target_wav_path, minf0, maxf0):
    print("vc path: ", vc_wav_path)
    print("target path: ", target_wav_path)
    vc_mcep,vc_f0 = mcep(vc_wav_path, minf0=minf0, maxf0=maxf0)
    t_mcep,t_f0 = mcep(target_wav_path, minf0=minf0, maxf0=maxf0)
    vc_mcep = vc_mcep[:, 1:]
    t_mcep = t_mcep[:, 1:]

    # non-silence parts
    vc_idx = np.where(vc_f0 > 0)[0]
    vc_mcep = vc_mcep[vc_idx]
    t_idx = np.where(t_f0 > 0)[0]
    t_mcep = t_mcep[t_idx]

    # DTW
    _, path = fastdtw(vc_mcep, t_mcep, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    vc_mcep_dtw = vc_mcep[twf[0]]
    t_mcep_dtw = t_mcep[twf[1]]

    # MCD
    diff2sum = np.sum((vc_mcep_dtw - t_mcep_dtw) ** 2, 1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)

    return mcd


with Path(F0_CONF_PATH).open(mode="r") as conf_file:
    f0_conf = yaml.safe_load(conf_file)

print("Start MCD Cal. ")

steps = []

for step in Path(VC_DATASET_DIR / "clean").iterdir():
    steps.append(step.name)

print("steps: ", steps)

for step in steps:
    result_list = []
    all_mcd = []
    for t_spk in TARGET_SPEAKERS:
        print("Target speaker: ", t_spk)
        mcd_per_t_spk = []

        ref_wav_list = list((REF_DIR / t_spk).glob("*.wav"))
        for vc_spk in VC_SPEAKERS:
            vc_wav_list = list((VC_DATASET_DIR / "clean" / step /"T_{}".format(t_spk) / vc_spk).glob("*.wav"))

            for vc_wav_path in vc_wav_list:

                _, id, _, _ = str(vc_wav_path.name).split("_")
                ref_wav_path = None
                for ref_wav in ref_wav_list:
                    if id in str(ref_wav):
                        ref_wav_path = REF_DIR / t_spk / ref_wav.name
                if ref_wav_path is None:
                    print("No wav file ", id, " found in reference eval data.")
                    exit()

                mcd = calculate_mcd(vc_wav_path, ref_wav_path,
                                        f0_conf[t_spk]["minf0"], f0_conf[t_spk]["maxf0"])
                mcd_per_t_spk.append(mcd)

        result_list.append("Target_Speaker_{} MCD_{}".format(t_spk, np.array(mcd_per_t_spk).mean()))
        #result_per_dataset_list.append("{}_{}".format(t_spk, mean_mcd_per_t_spk))
        all_mcd.append(np.array(mcd_per_t_spk).mean())


    #result_per_dataset_list.append("all_{}".format(mean_all_mcd))
    result_list.append("Mean_MCD_{}".format(np.array(all_mcd).mean()))

    generate_scp(MCD_SAVE_DIR, result_list, name="mcd-{}".format(step))

print("Process finished.")






