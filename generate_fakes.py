"""Batch image generation script for FID evaluation."""

import argparse
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from config import get_config
from data.dataset import FidDataset
from models.model import TRGAN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--checkpoint", default="saved_models/IAM-339-15-E3D3-LR5e-05-bs8-debug", type=str)
    parser.add_argument("--testDataset", default="IAM", type=str)
    parser.add_argument("--testepoch", default="8000", type=int)
    parser.add_argument("--allImg", default=False, type=bool)
    parser.add_argument("--realImg", action="store_true", default=False)
    parser.add_argument("--generate_type", default="iv_u", type=str)
    args, _ = parser.parse_known_args()
    return get_config(args)


def load_checkpoint(model, checkpoint):
    old_state = model.state_dict()

    if len(checkpoint.keys()) == 241:
        counter = 0
        for k, v in checkpoint.items():
            if k in old_state:
                old_state[k] = v
                counter += 1
            elif "netG." + k in old_state:
                old_state["netG." + k] = v
                counter += 1

        ck_keys = [k for k in checkpoint.keys() if "Feat_Encoder" in k]
        ok_keys = [k for k in old_state.keys() if "Feat_Encoder" in k]
        for ck, ok in zip(ck_keys, ok_keys):
            old_state[ok] = checkpoint[ck]
            counter += 1
        state_dict = old_state
    else:
        state_dict = {
            k2: v1
            for (k1, v1), (k2, v2) in zip(checkpoint.items(), old_state.items())
            if v1.shape == v2.shape
        }

    model.load_state_dict(state_dict, strict=False)
    return model


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.TEST.DATASET == "IAM":
        args.dataset_path = "files/IAM-32.pickle"
        args.num_writers = 339
    elif args.TEST.DATASET == "CVL":
        args.dataset_path = "files/CVL-32.pickle"
        args.TEST.CHECKPOINT = "saved_models/CVL-283-15-E3D3-LR5e-05-bs8-debug"
        args.num_writers = 283
    else:
        raise ValueError(f"Unsupported dataset: {args.TEST.DATASET}")

    epoch = args.TEST.EPOCH
    if epoch == 0:
        checkpoint_path = "files/iam_model.pth"
    else:
        checkpoint_path = args.TEST.CHECKPOINT + f"/{epoch:04d}_model.pth"

    test_obj = FidDataset(
        base_path=args.dataset_path,
        num_examples=args.num_examples,
        collator_resolution=args.resolution,
        mode="test",
    )
    test_loader = torch.utils.data.DataLoader(
        test_obj,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=test_obj.collate_fn,
    )

    args.alphabet = (
        "Only thewigsofrcvdampbkuq.A-210xT5'MDL,RYHJ\"ISPWENj&BC93VGFKz();"
        "#:!7U64Q8?+*ZX/%"
    )
    args.vocab_size = len(args.alphabet)

    model = TRGAN(args)
    output_dir = Path("saved_images") / args.TEST.DATASET / f"{epoch}_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    pth = torch.load(checkpoint_path)
    if "epoch" in pth:
        print(f"Checkpoint epoch: {pth['epoch']}")
    if "model" in pth:
        pth = pth["model"]

    load_checkpoint(model, pth)
    model.eval()

    with torch.no_grad():
        model.save_images_for_fid_calculation_I(
            output_dir, test_loader, "test",
            args.TEST.ALLIMG, args.TEST.REALIMG
        )
    print("Done")


if __name__ == "__main__":
    main()
