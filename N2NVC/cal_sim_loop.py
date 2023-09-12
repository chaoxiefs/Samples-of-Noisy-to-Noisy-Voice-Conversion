from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import numpy as np

#########################################################################
VC_DATASET_DIR = Path("convert_esc50-all")
REF_DIR = Path("/esc50-all/norm_vcc2018_eval_16k_snr5")

SIM_SAVE_DIR = Path("SIM") / VC_DATASET_DIR

VC_SPEAKERS = ["VCC2SF3","VCC2SF4",
               "VCC2SM3","VCC2SM4"]
TARGET_SPEAKERS = ["VCC2TF2", "VCC2TM2"]

##########################################################################


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

print("Start SIM Cal. ")

encoder = VoiceEncoder()

steps = []

for step in Path(VC_DATASET_DIR / "clean").iterdir():
    steps.append(step.name)
for step in steps:
    result_list = []
    all_sim = []
    for t_spk in TARGET_SPEAKERS:
        print("Target speaker: ", t_spk)
        sim_per_t_spk = []

        ref_wav_list = list((REF_DIR / t_spk).glob("*.wav"))
        for vc_spk in VC_SPEAKERS:
            vc_wav_list = list((VC_DATASET_DIR / "clean" /step /"T_{}".format(t_spk) / vc_spk).glob("*.wav"))

            for vc_wav_path in vc_wav_list:

                _, id, _ = str(vc_wav_path.name).split("_", maxsplit=2)
                ref_wav_path = None
                for ref_wav in ref_wav_list:
                    if id in str(ref_wav):
                        ref_wav_path = REF_DIR / t_spk / ref_wav.name
                if ref_wav_path is None:
                    print("No wav file ", id, " found in reference eval data.")
                    exit()

                vc_wav = preprocess_wav(vc_wav_path)
                ref_wav = preprocess_wav(ref_wav_path)

                vc_emb = encoder.embed_utterance(vc_wav)
                rf_emb = encoder.embed_utterance(ref_wav)

                sim = (rf_emb @ vc_emb.T)
                sim_per_t_spk.append(sim)

        result_list.append("Target_Speaker_{} SIM_{}".format(t_spk, np.array(sim_per_t_spk).mean()))
        #result_per_dataset_list.append("{}_{}".format(t_spk, mean_mcd_per_t_spk))
        all_sim.append(np.array(sim_per_t_spk).mean())


    #result_per_dataset_list.append("all_{}".format(mean_all_mcd))
    result_list.append("Mean_SIM_{}".format(np.array(all_sim).mean()))

    generate_scp(SIM_SAVE_DIR, result_list, name="sim-{}".format(step))

print("Process finished.")













