from itertools import chain
from pathlib import Path
from tqdm import tqdm
import os
import yaml
import torch.cuda.amp as amp
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from MyDataset import MyDataset
from model import Encoder, Decoder

####################################################
CONF_PATH = "config/config.yml"
#mannual_gpu_id = "7"
####################################################
#os.environ["CUDA_VISIBLE_DEVICES"] = mannual_gpu_id
def save_checkpoint(encoder, decoder, optimizer, scaler, scheduler, step, checkpoint_dir):
    checkpoint_state = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step}
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))


def train_model(conf):
    exp_dir = Path(conf["path"]["exp_dir"])
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    tensorboard_dir = exp_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True, parents=True)

    pre_trained_path = conf["path"]["pre_trained_dir"]
    #os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    #gpu_id = str(conf["training"]["gpu_id"])
    #if gpu_id is not None:
    #    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    #else:
    #    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")  # if torch.cuda.is_available() else "cpu")

    train_writer = SummaryWriter(tensorboard_dir/"train")
    eval_writer = SummaryWriter(tensorboard_dir / "eval")

    pre_trained_checkpoint = torch.load(pre_trained_path,
                                        map_location=lambda storage, loc: storage)

    encoder = Encoder(**conf["model"]["encoder"])
    decoder = Decoder(**conf["model"]["decoder"])

    encoder.load_state_dict(pre_trained_checkpoint["encoder"])
    try:
        decoder.load_state_dict(pre_trained_checkpoint["decoder"])
    except RuntimeError:
        pass

    encoder.to(device)
    decoder.to(device)

    optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()),
                    lr=float(conf["training"]["lr"]),
                    weight_decay=float(conf["training"]["weight_decay"]))

    scaler = amp.GradScaler(enabled=True)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=conf["training"]["scheduler"]["milestones"],
        gamma=conf["training"]["scheduler"]["gamma"])

    if conf["training"]["resume"]:
        resume_path = conf["training"]["resume_path"]
        print("Resume checkpoint from: {}:".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    tr_dataset = MyDataset(conf=conf, type="train")

    train_loader = DataLoader(
        dataset=tr_dataset,
        batch_size=conf["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        # pin_memory=True,
        num_workers=conf["training"]["n_workers"],
    )

    eval_dataset = MyDataset(conf=conf, type="eval")
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=conf["training"]["batch_size"],
        shuffle=False,
        drop_last=True,
        # pin_memory=True,
        num_workers=conf["training"]["n_workers"],
    )

    n_epochs = conf['training']["n_steps"] // len(train_loader) + 1
    print("n_epochs: ", n_epochs)
    start_epoch = global_step // len(train_loader) + 1
    print("len of train loader: ", len(train_loader))
    for epoch in range(start_epoch, n_epochs + 1):
        average_all_loss = average_recon_loss = average_vq_loss = average_perplexity = 0
        mean_eval_all_loss = mean_eval_recon_loss = mean_eval_vq_loss = mean_eval_perplexity = 0

        encoder.train()

        for i, (audio, mels, speakers) in enumerate(tqdm(train_loader), 1):
            audio, mels, speakers = audio.to(device), mels.to(device), speakers.to(device)

            optimizer.zero_grad()

            with amp.autocast(enabled=True):
                z, vq_loss, perplexity = encoder(mels)
                output = decoder(audio[:, :-1], z, speakers)
                recon_loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])
                loss = recon_loss + vq_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            average_recon_loss += (recon_loss.item() - average_recon_loss) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i
            average_all_loss = average_recon_loss + average_vq_loss

            global_step += 1

            if global_step % conf["training"]["checkpoint_interval"] == 0:
                save_checkpoint(
                    encoder, decoder, optimizer, scaler,
                    scheduler, global_step, checkpoint_dir)

        train_writer.add_scalar("recon_loss", average_recon_loss, global_step)
        train_writer.add_scalar("vq_loss", average_vq_loss, global_step)
        train_writer.add_scalar("all_loss", average_all_loss, global_step)
        train_writer.add_scalar("average_perplexity", average_perplexity, global_step)

        print("epoch:{}, steps:{}, recon loss:{:.2E}, vq loss:{:.2E}, perpexlity:{:.3f}"
              .format(epoch, global_step, average_recon_loss, average_vq_loss, average_perplexity))

        encoder.eval()

        with torch.no_grad():
            for i, (audio, mels, speakers) in enumerate(tqdm(eval_loader), 1):
                audio, mels, speakers = audio.to(device), mels.to(device), speakers.to(device)
                z, eval_vq_loss, eval_perplexity = encoder(mels)
                output = decoder(audio[:, :-1], z, speakers)
                eval_recon_loss = F.cross_entropy(output.transpose(1, 2), audio[:, 1:])

                mean_eval_recon_loss += (eval_recon_loss.item() - mean_eval_recon_loss) / i
                mean_eval_vq_loss += (eval_vq_loss.item() - mean_eval_vq_loss) / i
                mean_eval_perplexity += (eval_perplexity.item() - mean_eval_perplexity) / i
                mean_eval_all_loss = mean_eval_recon_loss + mean_eval_vq_loss

            eval_writer.add_scalar("recon_loss", mean_eval_recon_loss, global_step)
            eval_writer.add_scalar("vq_loss", mean_eval_vq_loss, global_step)
            eval_writer.add_scalar("average_perplexity", mean_eval_perplexity, global_step)
            eval_writer.add_scalar("all_loss", mean_eval_all_loss, global_step)
            print("epoch:{}, steps:{}, EVAL recon loss:{:.2E}, EVAL vq loss:{:.2E}, EVAL perpexlity:{:.3f}"
                  .format(epoch, global_step, mean_eval_recon_loss, mean_eval_vq_loss, mean_eval_perplexity))


if __name__ == "__main__":
    with Path(CONF_PATH).open(mode="r") as conf_file:
        conf = yaml.safe_load(conf_file)
    train_model(conf)
