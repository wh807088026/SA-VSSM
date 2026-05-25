"""Main model definition for SA-VSSM: Generator + Discriminators."""

import json
import os
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import CTCLoss
from torch.nn.utils import clip_grad_norm_

from data.dataset import CollectionTextDataset, TextDataset
from models import build_model
from models.inception import InceptionV3
from models.OCR_network import CRNN, strLabelConverter
from models.BigGAN_networks import Discriminator, WDiscriminator
from models.transformer import TransformerEncoder, TransformerEncoderLayer
from util.util import loss_hinge_dis, loss_hinge_gen, toggle_grad, make_one_hot
from models.unifont_module import UnifontModule
from models.positional_encodings import PositionalEncoding1D
import models.config as config
from PIL import Image


class FCNDecoder(nn.Module):
    """Upsampling decoder that reconstructs images from SSM decoder features."""

    def __init__(self, ups=3, n_res=2, dim=512, out_dim=1,
                 res_norm="adain", activ="relu", pad_type="reflect"):
        super().__init__()
        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        for _ in range(ups):
            self.model += [
                nn.Upsample(scale_factor=2),
                Conv2dBlock(dim, dim // 2, 5, 1, 2, norm="in",
                            activation=activ, pad_type=pad_type),
            ]
            dim //= 2
        self.model += [
            Conv2dBlock(dim, out_dim, 7, 1, 3, norm="none",
                        activation="tanh", pad_type=pad_type),
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    """Style-adaptive generator with SSM-based decoder."""

    def __init__(self, args):
        super().__init__()
        self.args = args

        in_channels = self.args.num_examples
        if self.args.is_seq:
            in_channels = 1

        # Style encoder: ResNet18
        self.style_encoder = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.style_encoder.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.style_encoder.fc = nn.Identity()
        self.style_encoder.avgpool = nn.Identity()

        # Style Transformer encoder
        enc_layer = TransformerEncoderLayer(
            self.args.tn_hidden_dim,
            self.args.tn_nheads,
            self.args.tn_dim_feedforward,
            self.args.tn_dropout,
            "relu",
            True,
        )
        enc_norm = nn.LayerNorm(self.args.tn_hidden_dim)
        self.style_transformer = TransformerEncoder(enc_layer, self.args.tn_enc_layers, enc_norm)

        # Mamba decoder (SSM-based)
        self.mamba_decoder = build_model(args, is_decoder=True)

        # Text encoder
        self.query_embed = UnifontModule(
            config.tn_dim_feedforward,
            self.args.alphabet + self.args.special_alphabet,
            input_type=self.args.query_input,
            device=self.args.device,
        )
        self.pos_encoder = PositionalEncoding1D(config.tn_hidden_dim)

        # Feature projection
        self.linear_q = nn.Linear(
            self.args.tn_dim_feedforward,
            self.args.tn_dim_feedforward * 8,
        )

        # Image decoder
        self.decoder = FCNDecoder(res_norm="in")

        # VAE modules (optional)
        self._muE = nn.Linear(512, 512)
        self._logvarE = nn.Linear(512, 512)
        self._muD = nn.Linear(512, 512)
        self._logvarD = nn.Linear(512, 512)

        self.l1_loss = nn.L1Loss()
        self.noise = torch.distributions.Normal(
            loc=torch.tensor([0.0]), scale=torch.tensor([1.0])
        )

    def encode_style(self, style_images):
        """Extract style features from reference images.

        Args:
            style_images: [B, N, H, W] tensor of N style reference images per sample.
        Returns:
            memory: [seq_len, B, dim] style feature sequence.
        """
        B, N, R, C = style_images.shape
        feat = self.style_encoder(style_images.view(B * N, 1, R, C))
        feat = feat.view(B, 512, 1, -1)
        feat = feat.flatten(2).permute(2, 0, 1)
        memory = self.style_transformer(feat)
        return memory

    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick."""
        sigma = torch.exp(logvar * 0.5)
        eps = torch.cuda.FloatTensor(logvar.size(0), 1).normal_(0, 1)
        eps = eps.expand(sigma.size())
        return mu + sigma * eps

    def forward(self, style_images, text_encode, text_encode_fake_js=None):
        """Full forward pass.

        Args:
            style_images: [B, N, H, W] style reference images.
            text_encode: [B, seq_len] target text encoding.
            text_encode_fake_js: optional auxiliary text encodings.
        Returns:
            output: [B, 1, H, W] generated image.
        """
        memory = self.encode_style(style_images)
        memory = memory.unsqueeze(0).permute(2, 3, 0, 1)

        text_emb = self.query_embed(text_encode).permute(1, 0, 2)
        text_emb = text_emb.unsqueeze(0).permute(2, 3, 0, 1)

        hs = self.mamba_decoder(text_emb, memory).permute(2, 3, 0, 1)
        h = hs.transpose(1, 2)[-1]

        if self.args.add_noise:
            h = h + self.noise.sample(h.size()).squeeze(-1).to(self.args.device)

        h = self.linear_q(h)
        h = h.contiguous()
        h = h.view(h.size(0), h.shape[1] * 2, 4, -1)
        h = h.permute(0, 3, 2, 1)
        output = self.decoder(h)
        return output

    def Eval(self, style_images, text_queries):
        """Multi-word inference. Returns a list of generated word images.

        Args:
            style_images: [B, N, H, W] style reference images.
            text_queries: [B, num_words, seq_len] text encoding for multiple words.
        Returns:
            List of [B, 1, H, W] generated images, one per word.
        """
        if self.args.is_seq:
            B, N, R, C = style_images.shape
            feat = self.style_encoder(style_images.view(B * N, 1, R, C))
            feat = feat.view(B, 512, 1, -1)
        else:
            feat = self.style_encoder(style_images)

        memory = feat.flatten(2).permute(2, 0, 1)
        memory = self.style_transformer(memory)
        memory = memory.unsqueeze(0).permute(2, 3, 0, 1)

        if self.args.is_kld:
            Ex = memory.permute(1, 0, 2)
            memory = self.reparameterize(self._muE(Ex), self._logvarE(Ex)).permute(1, 0, 2)

        out_images = []
        for i in range(text_queries.shape[1]):
            text_q = text_queries[:, i, :]
            text_emb = self.query_embed(text_q).permute(1, 0, 2)
            tgt = torch.zeros_like(text_emb)
            text_emb = text_emb.unsqueeze(0).permute(2, 3, 0, 1)

            hs = self.mamba_decoder(text_emb, memory).permute(2, 3, 0, 1)

            if self.args.is_kld:
                Dx = hs[0].permute(1, 0, 2)
                hs = self.reparameterize(self._muD(Dx), self._logvarD(Dx)).permute(1, 0, 2).unsqueeze(0)

            h = hs.transpose(1, 2)[-1]
            if self.args.add_noise:
                h = h + self.noise.sample(h.size()).squeeze(-1).to(self.args.device)

            h = self.linear_q(h)
            h = h.contiguous()
            h = h.view(h.size(0), h.shape[1] * 2, 4, -1)
            h = h.permute(0, 3, 2, 1)
            out = self.decoder(h)
            out_images.append(out.detach())

        return out_images


class TRGAN(nn.Module):
    """Full SA-VSSM model with Generator, Discriminators, and OCR recognizer."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.epsilon = 1e-7

        # Networks
        self.netG = Generator(self.args).to(self.args.device)
        self.netD = nn.DataParallel(
            Discriminator(resolution=self.args.resolution, n_classes=self.args.vocab_size)
        ).to(self.args.device)
        self.netW = nn.DataParallel(
            WDiscriminator(
                resolution=self.args.resolution,
                n_classes=self.args.vocab_size,
                output_dim=self.args.num_writers,
            )
        ).to(self.args.device)
        self.netOCR = CRNN(self.args).to(self.args.device)
        self.OCR_criterion = CTCLoss(zero_infinity=True, reduction="none")

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception = InceptionV3([block_idx]).to(self.args.device)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(), lr=self.args.g_lr,
            betas=(0.0, 0.999), weight_decay=0, eps=1e-8,
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=self.args.d_lr,
            betas=(0.0, 0.999), weight_decay=0, eps=1e-8,
        )
        self.optimizer_OCR = torch.optim.Adam(
            self.netOCR.parameters(), lr=self.args.ocr_lr,
            betas=(0.0, 0.999), weight_decay=0, eps=1e-8,
        )
        self.optimizer_W = torch.optim.Adam(
            self.netW.parameters(), lr=self.args.w_lr,
            betas=(0.0, 0.999), weight_decay=0, eps=1e-8,
        )

        self.optimizers = [
            self.optimizer_G, self.optimizer_D,
            self.optimizer_OCR, self.optimizer_W,
        ]
        for opt in self.optimizers:
            opt.zero_grad()

        # Text converter
        self.netconverter = strLabelConverter(self.args.alphabet)

        # Lexicon for random text generation
        with open(self.args.english_words_path, "rb") as f:
            lex = [w.decode("utf-8") for w in f.read().splitlines() if len(w) < 20]
        self.lex = lex

        # Pre-encode eval text
        with open("mytext.txt", "r") as f:
            self.eval_text = [j.encode() for j in sum([i.split() for i in f.readlines()], [])]
        self.eval_text_encode, self.eval_len_text = self.netconverter.encode(self.eval_text)
        self.eval_text_encode = self.eval_text_encode.to(self.args.device).repeat(
            self.args.batch_size, 1, 1
        )

    def _set_input(self, input_data):
        self.input = input_data

    def _set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_losses(self):
        return {
            "G": self.loss_G,
            "D": self.loss_D,
            "Dfake": self.loss_Dfake,
            "Dreal": self.loss_Dreal,
            "OCR_fake": self.loss_OCR_fake,
            "OCR_real": self.loss_OCR_real,
            "w_fake": self.loss_w_fake,
            "w_real": self.loss_w_real,
            "cycle1": self.Lcycle1,
            "cycle2": self.Lcycle2,
            "lda1": self.lda1,
            "lda2": self.lda2,
            "KLD": self.KLD,
        }

    def _forward(self):
        """Prepare inputs for training step."""
        self.real = self.input["img"].to(self.args.device)
        self.label = self.input["label"]
        self.sdata = self.input["simg"].to(self.args.device)
        self.ST_LEN = self.input["swids"]

        self.text_encode, self.len_text = self.netconverter.encode(self.label)
        self.one_hot_real = make_one_hot(
            self.text_encode, self.len_text, self.args.vocab_size
        ).to(self.args.device).detach()
        self.text_encode = self.text_encode.to(self.args.device).detach()
        self.len_text = self.len_text.detach()

        # Random fake text for discriminator training
        self.words = [
            word.encode("utf-8")
            for word in np.random.choice(self.lex, self.args.batch_size)
        ]
        self.text_encode_fake, self.len_text_fake = self.netconverter.encode(self.words)
        self.text_encode_fake = self.text_encode_fake.to(self.args.device)
        self.one_hot_fake = make_one_hot(
            self.text_encode_fake, self.len_text_fake, self.args.vocab_size
        ).to(self.args.device)

        # Auxiliary text encodings
        self.text_encode_fake_js = []
        for _ in range(self.args.num_words - 1):
            words_j = [
                word.encode("utf-8")
                for word in np.random.choice(self.lex, self.args.batch_size)
            ]
            enc_j, len_j = self.netconverter.encode(words_j)
            self.text_encode_fake_js.append(enc_j.to(self.args.device))

        self.fake = self.netG(self.sdata, self.text_encode_fake, self.text_encode_fake_js)

    def backward_D_OCR(self):
        """Discriminator + OCR backward pass."""
        pred_real = self.netD(self.real.detach())
        pred_fake = self.netD(**{"x": self.fake.detach()})
        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(
            pred_fake, pred_real,
            self.len_text_fake.detach(), self.len_text.detach(), True,
        )
        self.loss_D = self.loss_Dreal + self.loss_Dfake

        pred_real_ocr = self.netOCR(self.real.detach())
        preds_size = torch.IntTensor(
            [pred_real_ocr.size(0)] * self.args.batch_size
        ).detach()
        loss_ocr_real = self.OCR_criterion(
            pred_real_ocr, self.text_encode.detach(), preds_size, self.len_text.detach()
        )
        self.loss_OCR_real = torch.mean(loss_ocr_real[~torch.isnan(loss_ocr_real)])

        loss_total = self.loss_D + self.loss_OCR_real
        loss_total.backward()

        for param in self.netOCR.parameters():
            for bad in (torch.isnan, torch.isinf):
                param.grad[bad(param.grad)] = 0

        return loss_total

    def backward_D_WL(self):
        """Discriminator + Writer classifier backward pass."""
        pred_real = self.netD(self.real.detach())
        pred_fake = self.netD(**{"x": self.fake.detach()})
        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(
            pred_fake, pred_real,
            self.len_text_fake.detach(), self.len_text.detach(), True,
        )
        self.loss_D = self.loss_Dreal + self.loss_Dfake

        self.loss_w_real = self.netW(
            self.real.detach(), self.input["wcl"].to(self.args.device)
        ).mean()

        loss_total = self.loss_D + self.loss_w_real
        loss_total.backward()
        return loss_total

    def backward_G_only(self):
        """Generator + OCR backward pass."""
        self.loss_G = loss_hinge_gen(
            self.netD(**{"x": self.fake}), self.len_text_fake.detach(), True,
        ).mean()

        pred_fake_ocr = self.netOCR(self.fake)
        preds_size = torch.IntTensor(
            [pred_fake_ocr.size(0)] * self.args.batch_size
        ).detach()
        loss_ocr_fake = self.OCR_criterion(
            pred_fake_ocr, self.text_encode_fake.detach(),
            preds_size, self.len_text_fake.detach(),
        )
        self.loss_OCR_fake = torch.mean(loss_ocr_fake[~torch.isnan(loss_ocr_fake)])

        self.loss_G = (
            self.loss_G
            + self.Lcycle1
            + self.Lcycle2
            + self.lda1
            + self.lda2
            - self.KLD
        )
        loss_T = self.loss_G + self.loss_OCR_fake

        # Balanced gradient weighting
        grad_ocr = torch.autograd.grad(
            self.loss_OCR_fake, self.fake, retain_graph=True
        )[0]
        grad_adv = torch.autograd.grad(
            self.loss_G, self.fake, retain_graph=True
        )[0]
        alpha = 0.7 * torch.div(
            torch.std(grad_adv), self.epsilon + torch.std(grad_ocr)
        )
        if alpha > 1000 or alpha < 0.0001:
            print(alpha)
        self.loss_OCR_fake = alpha.detach() * self.loss_OCR_fake

        loss_T = self.loss_G + self.loss_OCR_fake
        loss_T.backward(retain_graph=True)

        grad_ocr = torch.autograd.grad(
            self.loss_OCR_fake, self.fake, create_graph=False, retain_graph=True
        )[0]
        grad_adv = torch.autograd.grad(
            self.loss_G, self.fake, create_graph=False, retain_graph=True
        )[0]
        self.loss_grad_OCR = 1e6 * torch.mean(grad_ocr ** 2)
        self.loss_grad_adv = 1e6 * torch.mean(grad_adv ** 2)

        with torch.no_grad():
            loss_T.backward()

        if torch.isnan(self.loss_OCR_fake) or torch.isnan(self.loss_G):
            print(f"OCR fake: {self.loss_OCR_fake}, G: {self.loss_G}, words: {self.words}")
            sys.exit()

    def backward_G_WL(self):
        """Generator + Writer classifier backward pass."""
        self.loss_G = loss_hinge_gen(
            self.netD(**{"x": self.fake}), self.len_text_fake.detach(), True,
        ).mean()
        self.loss_w_fake = self.netW(
            self.fake, self.input["wcl"].to(self.args.device)
        ).mean()

        self.loss_G = (
            self.loss_G
            + self.Lcycle1
            + self.Lcycle2
            + self.lda1
            + self.lda2
            - self.KLD
        )
        loss_T = self.loss_G + self.loss_w_fake
        loss_T.backward(retain_graph=True)

        grad_w = torch.autograd.grad(
            self.loss_w_fake, self.fake, create_graph=True, retain_graph=True
        )[0]
        grad_adv = torch.autograd.grad(
            self.loss_G, self.fake, create_graph=True, retain_graph=True
        )[0]
        alpha = 0.7 * torch.div(
            torch.std(grad_adv), self.epsilon + torch.std(grad_w)
        )
        if alpha > 1000 or alpha < 0.0001:
            print(alpha)
        self.loss_w_fake = alpha.detach() * self.loss_w_fake

        loss_T = self.loss_G + self.loss_w_fake
        loss_T.backward(retain_graph=True)

        grad_w = torch.autograd.grad(
            self.loss_w_fake, self.fake, create_graph=False, retain_graph=True
        )[0]
        grad_adv = torch.autograd.grad(
            self.loss_G, self.fake, create_graph=False, retain_graph=True
        )[0]
        self.loss_grad_W = 1e6 * torch.mean(grad_w ** 2)
        self.loss_grad_adv = 1e6 * torch.mean(grad_adv ** 2)

        with torch.no_grad():
            loss_T.backward()

    def optimize_G_only(self):
        self._forward()
        self._set_requires_grad([self.netD, self.netOCR, self.netW], False)
        self.backward_G_only()

    def optimize_G_WL(self):
        self._forward()
        self._set_requires_grad([self.netD, self.netOCR, self.netW], False)
        self.backward_G_WL()

    def optimize_D_OCR(self):
        self._forward()
        self._set_requires_grad([self.netD, self.netOCR], True)
        self.optimizer_D.zero_grad()
        self.optimizer_OCR.zero_grad()
        self.backward_D_OCR()

    def optimize_D_WL(self):
        self._forward()
        self._set_requires_grad([self.netD, self.netW], True)
        self.optimizer_D.zero_grad()
        self.optimizer_W.zero_grad()
        self.backward_D_WL()

    def optimize_G_step(self):
        self.optimizer_G.step()
        self.optimizer_G.zero_grad()

    def optimize_D_OCR_step(self):
        self.optimizer_D.step()
        self.optimizer_OCR.step()
        self.optimizer_D.zero_grad()
        self.optimizer_OCR.zero_grad()

    def optimize_D_WL_step(self):
        self.optimizer_D.step()
        self.optimizer_W.step()
        self.optimizer_D.zero_grad()
        self.optimizer_W.zero_grad()

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _generate_fakes(self, style_images, text_encode, len_text, encode_pos, resolution=16):
        """Generate fake images and return as list of numpy arrays."""
        self.fakes = self.netG.Eval(style_images, text_encode)
        np_fakes = []
        for batch_idx in range(self.fakes[0].shape[0]):
            for idx, fake in enumerate(self.fakes):
                fake = (fake[batch_idx, 0, :len_text[idx] * resolution] + 1) / 2
                np_fakes.append(fake.cpu().numpy())
        return np_fakes

    def _generate_page(self, style_images, style_lens):
        """Assemble generated words and style references into a full page image."""
        self.fakes = self.netG.Eval(style_images, self.eval_text_encode)

        page1s, page2s = [], []
        for batch_idx in range(self.args.batch_size):
            word_t, word_l, line_wids = [], [], []
            gap = np.ones([self.args.img_height, 16])

            # --- row 1: generated words ---
            for idx, fake_ in enumerate(self.fakes):
                word_t.append(
                    (fake_[batch_idx, 0, :self.eval_len_text[idx] * self.args.resolution].cpu().numpy() + 1) / 2
                )
                word_t.append(gap)
                if len(word_t) == 16 or idx == len(self.fakes) - 1:
                    line_ = np.concatenate(word_t, -1)
                    word_l.append(line_)
                    line_wids.append(line_.shape[1])
                    word_t = []

            gap_h = np.ones([16, max(line_wids)])
            page_ = []
            for l in word_l:
                pad_ = np.ones([self.args.img_height, max(line_wids) - l.shape[1]])
                page_.append(np.concatenate([l, pad_], 1))
                page_.append(gap_h)
            page1 = np.concatenate(page_, 0)

            # --- row 2: reference style images ---
            word_t, word_l, line_wids = [], [], []
            sdata_ = [i.unsqueeze(1) for i in torch.unbind(style_images, 1)]
            for idx, st in enumerate(sdata_):
                word_t.append(
                    (st[batch_idx, 0, :, :int(style_lens.cpu().numpy()[batch_idx][idx])].cpu().numpy() + 1) / 2
                )
                word_t.append(gap)
                if len(word_t) == 16 or idx == len(sdata_) - 1:
                    line_ = np.concatenate(word_t, -1)
                    word_l.append(line_)
                    line_wids.append(line_.shape[1])
                    word_t = []

            gap_h = np.ones([16, max(line_wids)])
            page_ = []
            for l in word_l:
                pad_ = np.ones([self.args.img_height, max(line_wids) - l.shape[1]])
                page_.append(np.concatenate([l, pad_], 1))
                page_.append(gap_h)
            page2 = np.concatenate(page_, 0)

            # Stack rows vertically
            merge_h = max(page1.shape[0], page2.shape[0])
            if page1.shape[0] < merge_h:
                page1 = np.concatenate([page1, np.ones([merge_h - page1.shape[0], page1.shape[1]])], 0)
            if page2.shape[0] < merge_h:
                page2 = np.concatenate([page2, np.ones([merge_h - page2.shape[0], page2.shape[1]])], 0)

            page1s.append(page1)
            page2s.append(page2)

        page1s_ = np.concatenate(page1s, 0)
        max_wid = max(p.shape[1] for p in page2s)
        padded_page2s = []
        for para in page2s:
            padded_page2s.append(
                np.concatenate([para, np.ones([para.shape[0], max_wid - para.shape[1]])], 1)
            )
        padded_page2s_ = np.concatenate(padded_page2s, 0)
        return np.concatenate([padded_page2s_, page1s_], 1)

    def save_images_for_fid_calculation_I(
        self, path, loader, split="train", allImg=False, realImg=False
    ):
        """Save generated images for FID evaluation."""
        if not isinstance(path, Path):
            path = Path(path)
        path.mkdir(exist_ok=True, parents=True)

        if allImg:
            real_base = path / f"Real_{split}"
            fake_base = path / f"Fake_{split}"
        else:
            real_base = path / f"Real_{split}_wid"
            fake_base = path / f"Fake_{split}_wid"
        real_base.mkdir(exist_ok=True)
        fake_base.mkdir(exist_ok=True)

        ann = {}
        start_time = time.time()

        for step, data in enumerate(loader):
            style_imgs = data["simg"].to(self.args.device)
            texts = [
                t.decode("utf-8").encode("utf-8")
                for t in data["label"]
            ]
            enc, enc_len = self.netconverter.encode(texts)
            enc = enc.to(self.args.device).unsqueeze(1)

            self.fakes = self.netG.Eval(style_imgs, enc)
            fake_imgs = torch.cat(self.fakes, 1).detach().cpu().numpy()
            real_imgs = data["img"].detach().cpu().numpy()
            writer_ids = data["wcl"].int().tolist()

            for i, (fake, real, wid, label, img_id) in enumerate(
                zip(fake_imgs, real_imgs, writer_ids, data["label"], data["idx"])
            ):
                label = label.decode()
                ann[step * self.args.batch_size + i] = label

                if allImg:
                    img_name = f"{wid:03d}_{img_id:05d}.png"
                    real_path = real_base / img_name
                    fake_path = fake_base / img_name
                else:
                    img_name = f"{img_id:05d}.png"
                    real_path = real_base / f"{wid:03d}" / img_name
                    fake_path = fake_base / f"{wid:03d}" / img_name

                if realImg:
                    real_path.parent.mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(real_path), 255 * real.squeeze())

                fake_path.parent.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(fake_path), 255 * fake.squeeze())

            if step % 1000 == 0:
                elapsed = (time.time() - start_time) / (step + 1) * (len(loader) - step - 1)
                print(f"[{(step+1)/len(loader)*100:.1f}%] ETA: {str(timedelta(seconds=elapsed))}")

        with open(path / "ann.json", "w") as f:
            json.dump(ann, f)

        return real_base, fake_base
