"""Single-sample generation: given a folder of style images, generate handwritten text."""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

from config import get_config
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
    parser.add_argument(
        "--checkpoint",
        default="saved_models/IAM-339-15-E3D3-LR5e-05-bs8-debug",
        type=str,
    )
    parser.add_argument("--testDataset", default="IAM", type=str)
    parser.add_argument("--testepoch", default="11000", type=int)
    parser.add_argument(
        "--text",
        default="The only limit to our realization of tomorrow is our doubts of today.",
        type=str,
    )
    args, _ = parser.parse_known_args()
    return get_config(args)


def get_transform(grayscale=True, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if convert:
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    return transforms.Compose(transform_list)


class StyleFolderDataset:
    """Load style reference images from a folder."""

    def __init__(self, folder_path, num_examples=15):
        self.imgs = list(Path(folder_path).iterdir())
        self.transform = get_transform()
        self.num_examples = num_examples

    def sample(self):
        """Return a dict with style image tensor and widths."""
        indices = np.random.choice(len(self.imgs), self.num_examples, replace=False)
        imgs = []
        widths = []
        for idx in indices:
            img = Image.open(self.imgs[idx]).convert("L")
            img = img.resize(
                (img.size[0] * 32 // max(img.size[1], 1), 32),
                Image.Resampling.BILINEAR,
            )
            img = 255 - np.array(img)
            h, w = img.shape
            max_w = 192
            out = np.zeros((h, max_w), dtype="float32")
            out[:, :min(w, max_w)] = img[:, :max_w]
            imgs.append(self.transform(Image.fromarray(255 - out.astype(np.uint8))))
            widths.append(w)
        return {
            "simg": torch.cat(imgs, 0),
            "swids": widths,
        }


class TextLabelConverter:
    """CTC-based label encoder/decoder."""

    def __init__(self, alphabet, ignore_case=False):
        if ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + "-"
        self._dict = {c: i + 1 for i, c in enumerate(alphabet)}

    def encode(self, texts):
        """Encode a list of strings into (padded_tensor, lengths, raw)."""
        lengths = []
        results = []
        for item in texts:
            if isinstance(item, bytes):
                item = item.decode("utf-8", "strict")
            lengths.append(len(item))
            results.append(
                torch.LongTensor([self._dict.get(c, 0) for c in item])
            )
        return torch.nn.utils.rnn.pad_sequence(results, batch_first=True), \
            torch.LongTensor(lengths), None

    def decode(self, t, length, raw=False):
        """Decode encoded tensor back to strings."""
        if length.numel() == 1:
            length = length[0]
            if raw:
                return "".join([self.alphabet[i - 1] for i in t if i > 0])
            chars = []
            for i in range(length):
                if t[i] != 0 and not (i > 0 and t[i - 1] == t[i]):
                    chars.append(self.alphabet[t[i] - 1])
            return "".join(chars)
        else:
            texts, idx = [], 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode(t[idx:idx + l], torch.LongTensor([l]), raw=raw))
                idx += l
            return texts


class SAVSSMWriter:
    """High-level inference interface for SA-VSSM."""

    def __init__(self, checkpoint_path, args):
        self.model = TRGAN(args)
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        self._load_checkpoint(checkpoint)
        self.model.eval()
        self.style_dataset = None
        alphabet = (
            "Only thewigsofrcvdampbkuq.A-210xT5'MDL,RYHJ\"ISPWENj&BC93VGFKz()"
            ";:#:!7U64Q8?+*ZX/%"
        )
        self.converter = TextLabelConverter(alphabet)

    def _load_checkpoint(self, checkpoint):
        if not isinstance(checkpoint, dict):
            checkpoint = {"model": checkpoint}
        pth = checkpoint.get("model", checkpoint)
        old_state = self.model.state_dict()

        if len(pth.keys()) == 241:
            counter = 0
            for k, v in pth.items():
                if k in old_state:
                    old_state[k] = v
                    counter += 1
                elif "netG." + k in old_state:
                    old_state["netG." + k] = v
                    counter += 1

            ck_keys = [k for k in pth.keys() if "Feat_Encoder" in k]
            ok_keys = [k for k in old_state.keys() if "Feat_Encoder" in k]
            for ck, ok in zip(ck_keys, ok_keys):
                old_state[ok] = pth[ck]
                counter += 1
            self.model.load_state_dict(old_state, strict=False)
        else:
            state_dict = {
                k2: v1
                for (k1, v1), (k2, v2) in zip(pth.items(), old_state.items())
                if v1.shape == v2.shape
            }
            self.model.load_state_dict(state_dict, strict=False)

    def set_style_folder(self, folder, num_examples=15):
        self.style_dataset = StyleFolderDataset(folder, num_examples=num_examples)

    @torch.no_grad()
    def generate(self, texts):
        """Generate handwritten images for the given texts using the set style."""
        if isinstance(texts, str):
            texts = [texts]
        if self.style_dataset is None:
            raise RuntimeError("Style folder not set. Call set_style_folder() first.")

        gap = np.ones((32, 16))
        fakes = []
        for text in texts:
            style = self.style_dataset.sample()
            style_imgs = style["simg"].unsqueeze(0).to(self.model.device)
            enc, length, _ = self.converter.encode([text])
            enc = enc.to(self.model.device)
            fake = self.model._generate_fakes(style_imgs, enc, length, None)
            fake = np.concatenate(
                sum([[img, gap] for img in fake], []), axis=1
            )[:, :-16]
            fakes.append((fake * 255).astype(np.uint8))
        return fakes


def main():
    args = parse_args()
    set_seed(args.seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.vocab_size = len(args.alphabet)

    epoch = args.TEST.EPOCH
    model_path = args.TEST.CHECKPOINT + f"/{epoch:04d}_model.pth"
    output_dir = "files/output"
    style_dir = "files/style_samples/"

    generate_texts = args.TEST.TEXT.splitlines()
    writer_ids = ["00", "01", "02", "292", "293", "295"]

    for wid in writer_ids:
        style_folder = os.path.join(style_dir, wid)
        writer = SAVSSMWriter(model_path, args)
        writer.set_style_folder(style_folder)
        print(f"Generating for writer {wid}")

        for text in tqdm(generate_texts, desc=f"Writer {wid}"):
            out_folder = os.path.join(output_dir, text)
            os.makedirs(out_folder, exist_ok=True)
            fakes = writer.generate(text)
            for j, fake in enumerate(fakes):
                fname = f"{wid}_{j:04d}_{text.replace(' ', '_')}.png"
                cv2.imwrite(os.path.join(out_folder, fname), fake)

    print("Done")


if __name__ == "__main__":
    main()
