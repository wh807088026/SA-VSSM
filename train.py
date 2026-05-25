"""Training script for SA-VSSM."""

import argparse
import os
import random
import time

import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from data.dataset import CollectionTextDataset, TextDataset
from models.model import TRGAN
from config import get_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset", default="CVL", type=str)
    parser.add_argument("--wandb", action="store_true", default=True)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--no_real_loss", action="store_true", default=False)
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/vmambav2_small_224.yaml",
        metavar="FILE",
        help="Path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options with 'KEY VALUE' pairs.",
        default=None,
        nargs="+",
    )
    args, unparsed = parser.parse_known_args()
    return get_config(args)


def main():
    args = parse_args()
    set_seed(args.seed)

    num_writers_train = 339
    num_writers_test = 161
    if args.dataset == "CVL":
        num_writers_train = 283
        num_writers_test = 27

    dataset = CollectionTextDataset(
        args.dataset,
        "files",
        TextDataset,
        file_suffix=args.file_suffix,
        num_examples=args.num_examples,
        collator_resolution=args.resolution,
        min_virtual_size=num_writers_train,
        validation=False,
        debug=False,
        height=args.img_height,
    )
    dataset_val = CollectionTextDataset(
        args.dataset,
        "files",
        TextDataset,
        file_suffix=args.file_suffix,
        num_examples=args.num_examples,
        collator_resolution=args.resolution,
        min_virtual_size=num_writers_test,
        validation=True,
        height=args.img_height,
    )
    args.num_writers = dataset.num_writers

    if args.dataset in ("IAM", "CVL"):
        args.alphabet = (
            "Only thewigsofrcvdampbkuq.A-210xT5'MDL,RYHJ\"ISPWENj&BC93VGFKz();"
            "#:!7U64Q8?+*ZX/%"
        )
    else:
        args.alphabet = "".join(
            sorted(set(dataset.alphabet + dataset_val.alphabet))
        )

    args.vocab_size = len(args.alphabet)
    if not args.is_seq:
        args.num_words = args.num_examples

    args.exp_name = (
        f"{args.dataset}-{args.num_writers}-{args.num_examples}"
        f"-E{args.tn_enc_layers}D{args.tn_dec_layers}"
        f"-LR{args.g_lr}-bs{args.batch_size}-{args.tag}"
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset_val.collate_fn,
    )

    model = TRGAN(args)
    start_epoch = 0

    wandb_params = {
        "project": "SA-VSSM",
        "config": {
            k: v
            for k, v in args.__dict__.items()
            if isinstance(v, (bool, int, str, float))
        },
        "name": args.exp_name,
        "mode": "dryrun",
    }

    model_path = os.path.join(args.save_model_path, args.exp_name)
    os.makedirs(model_path, exist_ok=True)
    checkpoint_path = os.path.join(model_path, "model.pth")

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
        wandb_params["id"] = checkpoint.get("wandb_id")
        wandb_params["resume"] = True
        print(f"Loaded checkpoint from {checkpoint_path}")

    if args.test:
        args.wandb = False
        args.epochs = 1
        print("Starting evaluation")
    else:
        print(f"Starting training (epoch {start_epoch})")

    if args.wandb:
        wandb.init(**wandb_params)
        wandb.watch(model)

    writer = SummaryWriter(f"runs/{args.exp_name}")

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        for i, data in enumerate(train_loader):
            if i % args.num_critic_gocr_train == 0:
                model._set_input(data)
                model.optimize_G_only()
                model.optimize_G_step()
            if i % args.num_critic_docr_train == 0:
                model._set_input(data)
                model.optimize_D_OCR()
                model.optimize_D_OCR_step()
            if i % args.num_critic_gwl_train == 0:
                model._set_input(data)
                model.optimize_G_WL()
                model.optimize_G_step()
            if i % args.num_critic_dwl_train == 0:
                model._set_input(data)
                model.optimize_D_WL()
                model.optimize_D_WL_step()

        epoch_time = time.time() - start_time
        losses = model.get_current_losses()

        for name, val in losses.items():
            writer.add_scalar(f"loss/{name}", val, epoch)

        if args.wandb and epoch % 10 == 0:
            val_data = next(iter(val_loader))
            page = model._generate_page(model.sdata, model.input["swids"])
            page_val = model._generate_page(
                val_data["simg"].to(args.device), val_data["swids"]
            )
            wandb.log(
                {
                    **{f"loss/{k}": v for k, v in losses.items()},
                    "epoch": epoch,
                    "time_per_epoch": epoch_time,
                    "samples/page": [wandb.Image(page), wandb.Image(page_val)],
                }
            )

        print(f"Epoch {epoch} | Time {epoch_time:.1f}s | {losses}")

        checkpoint = {
            "model": model.state_dict(),
            "wandb_id": wandb_params.get("id"),
            "epoch": epoch + 1,
        }
        if epoch % args.save_model == 0 and not args.test:
            torch.save(checkpoint, checkpoint_path)
        if epoch % args.save_model_history == 0 and not args.test:
            torch.save(
                checkpoint, os.path.join(model_path, f"{epoch:04d}_model.pth")
            )

    writer.close()


if __name__ == "__main__":
    main()
