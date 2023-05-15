import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from sklearn import metrics
from random import random
from PIL import Image
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
from torch import einsum
from einops import rearrange
import cv2 as cv

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer
from taming.modules.losses.ssim import SSIM


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 classname,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        z_channels = ddconfig["z_channels"] if not ddconfig["double_z"] else 2 * ddconfig["z_channels"]
        self.quantize = GumbelQuantize(embed_dim, n_embed, remap=remap)
        self.classname = classname
        #         self.quantize = VectorQuantizer(n_embed, embed_dim, alpha=1.0, beta=0.25,
        #                                         remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        self.monitor = monitor
        self.suffix = 'rgb' if self.encoder.in_channels == 3 else 'gray'

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info, d = self.quantize(h)
        return quant, emb_loss, info, d

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input_x):
        #         b, c, h, w = input_x.size()
        quant_x, diff, info, d = self.encode(input_x)
        #         hy = self.encoder(input_y)
        #         hy = self.quant_conv(hy)
        # #         mask = torch.randint(2, (1,c,1,1) ,device=input_x.device)
        # #         mix_h = mask * hx + (1 - mask) * hy
        #         alpha = random()
        #         mix_h = alpha * hx + (1-alpha) * hy
        dec_x = self.decode(quant_x)
        #         quant_mix, _ , _ = self.quantize(mix_h)
        #         dec_mix = self.decoder(quant_mix)

        return dec_x, diff, info, d

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        #         x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        classlabel = None
        bsz = x.size(0)
        xrec, qloss, info, d = self(x)
        d = d.reshape(bsz, -1)
        if self.current_epoch > 50:
            self.logger.experiment.add_histogram('train/min_prob', d.min(1).values, self.global_step)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.current_epoch, info, classlabel,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            disc_img, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.current_epoch, info, classlabel,
                                                last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

            return disc_img

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        classlabel = None
        bsz = x.size(0)
        xrec, qloss, info, d = self(x)
        d = d.reshape(bsz, -1)
        if self.current_epoch > 50:
            self.logger.experiment.add_histogram('val/min_prob', d.min(1).values, self.global_step)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.current_epoch, info,classlabel,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.current_epoch, info,classlabel,
                                            last_layer=self.get_last_layer(), split="val")
        nll_loss = log_dict_ae['val/nll_loss']
        del log_dict_ae['val/nll_loss']
        self.log("val/nll_loss", nll_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return aeloss

    def test_step(self, batch, batch_idx):
        para_dict = {}
        x = self.get_input(batch, self.image_key)
        device = x.device
        img_label = batch['label']
        name = batch['name']
        classname = batch['classname']
        mask = self.get_input(batch, 'mask')
        xrec, _, _, d = self(x)

        rec_path = f'rec_imgs/{self.classname}/{self.suffix}/{classname[0]}_test'
        os.makedirs(rec_path, exist_ok=True)

        # class7 mean 0.7670 std :0.0794
        x.add_(1.0).mul_(0.5)
        xrec.add_(1.0).mul_(0.5)

        ssim = SSIM(win_size=11, sigma=1.5, in_channel=self.encoder.in_channels).to(device)
        maps = ssim(x, xrec, data_range=1.0, use_pad=True, return_full=True).mean(1)
        maps = maps.squeeze().cpu().numpy()
        maps = 255 - np.uint8(maps * 255)
        thres, res_map = cv.threshold(maps, 0, 255, cv.THRESH_OTSU)  # cv.THRESH_TRIANGLE, cv.THRESH_OTSU

        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(res_map, connectivity=8)

        area_list = [stats[i][-1] for i in range(num_labels)]
        max_area = max(area_list[1:])
        if self.suffix == 'gray':
            res_map = torch.from_numpy(res_map[None, None, :]).div(255.).to(device)
            grid = torch.cat((x, xrec, mask, res_map))
            torchvision.utils.save_image(grid, os.path.join(rec_path, name[0]), nrow=4)
        else:
            f, axes = plt.subplots(1, 4)
            x = np.clip(x.squeeze(0).cpu().numpy()*255, 0, 255).astype(np.uint8)
            xrec = np.clip(xrec.squeeze(0).cpu().numpy()*255, 0, 255).astype(np.uint8)
            axes[0].imshow(x.transpose(1, 2, 0))
            axes[0].axis('off')
            axes[1].imshow(xrec.transpose(1, 2, 0))
            axes[1].axis('off')
            axes[2].imshow(mask.squeeze().cpu().numpy(), cmap='gray')
            axes[2].axis('off')
            axes[3].imshow(res_map, cmap='gray')
            axes[3].axis('off')

            f.savefig(os.path.join(rec_path, name[0]))
            plt.close()
        # image_filtered = np.zeros_like(res_map)
        # for (i, label) in enumerate(np.unique(labels)):
        #     # 如果是背景，忽略
        #     if label == 0:
        #         continue
        #     if stats[i][-1] >= 246:
        #         image_filtered[labels == i] = 255
        #
        # def dice_coef(pred, target, smooth=0.01):
        #     pred = pred.view(-1)
        #     target = target.view(-1)
        #     intersection = (pred * target).sum()
        #
        #     return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        #
        # seg = torch.from_numpy(image_filtered[None, None, :]).div(255.).to(device)
        # dice = dice_coef(seg, mask).item()
        # if img_label.item() == 1:

        return {'label': img_label, 'class': classname[0],'max_area': max_area, 'thres': thres,} # 'dice': dice 'max_area': max_area, 'thres': thres,

    def test_epoch_end(self, outputs):
        labels = np.array([x['label'].cpu().numpy() for x in outputs]).ravel()
        thres = np.array([x['thres'] for x in outputs])
        max_areas = np.array([x['max_area'] for x in outputs])
        classnames = np.array([x['class'] for x in outputs])
        # dice = np.array([x['dice'] for x in outputs]).mean()
        # print(f'dice: {dice}')
        #         dice = np.array([x['dice'] for x in outputs]).mean()
        #         print(f'dice: {dice}')

        #         plt.hist([min_prob[labels == 0], min_prob[labels == 1]],bins=100, density=True, stacked=True,label=["Normal", "Abnormal"])
        #         plt.title("Discrete distributions of min_prob")
        #         plt.xlabel("min probablites")
        #         plt.ylabel("h")
        #         plt.legend()
        #         plt.savefig(self.classname+'_min_prob.png')
        #         plt.close()

        def normalize(x):
            x_min = x.min()
            x_max = x.max()
            new_x = (x - x_min) / (x_max - x_min)
            return new_x

        def cal_metrics(y, y_pred):
            auroc = metrics.roc_auc_score(y, y_pred)
            precisions, recalls, thresholds = metrics.precision_recall_curve(y, y_pred)
            F1_scores = np.divide(2 * precisions * recalls, precisions + recalls,
                                  out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
            opt_idx = np.argmax(F1_scores)
            opt_thre = thresholds[opt_idx]
            f1 = F1_scores[opt_idx]
            pred = (y_pred >= opt_thre).astype(int)
            acc = np.sum(pred == y) / len(y)
            recall = recalls[opt_idx]
            precision = precisions[opt_idx]

            return {'auroc': auroc,
                    'opt_thre': opt_thre,
                    'f1': f1,
                    'acc': acc,
                    'recall': recall,
                    'precision': precision,
                    }
        ostu_col = []
        area_col = []

        class_all = np.unique(classnames)
        for classname in class_all:
            thre_metrics = cal_metrics(labels[classnames == classname], thres[classnames == classname])
            area_metrics = cal_metrics(labels[classnames == classname], max_areas[classnames == classname])
            ostu_col.append(thre_metrics)
            area_col.append(area_metrics)

        ostu_info = pd.DataFrame(ostu_col, index=class_all)
        area_info = pd.DataFrame(area_col, index=class_all)
        ostu_info.to_csv(f'ostu_{self.suffix}.csv')
        area_info.to_csv(f'area_{self.suffix}.csv')
        print(f"ostu mean auc : {ostu_info['auroc'].mean()}")
        print(f"area mean auc : {area_info['auroc'].mean()}")



    def configure_optimizers(self):
        lr = self.learning_rate
        dis_lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator_img.parameters(),
                                    lr=dis_lr, betas=(0.5, 0.9))
        #         lr_scheduler1 = StepLR(opt_ae,20,0.95)
        #         lr_scheduler2 = StepLR(opt_disc,20,0.95)
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _, _, d = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)

        log["inputs"] = x
        log["reconstructions"] = xrec

        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 monitor='val/aeloss',
                 ignore_keys=['loss.discriminator'],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, info, _ = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, info, split="train")

        self.log("train/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, info, _ = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, info, split="val")
        nll_loss = log_dict_ae['val/nll_loss']
        del log_dict_ae['val/nll_loss']
        self.log("val/nll_loss", nll_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        return aeloss

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _, _, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    #     def test_step(self, batch, batch_idx):
    #         x = self.get_input(batch, self.image_key)
    # #         x_ = x.add(1).mul(0.5)
    #         path = 'test_rec'
    # #         os.makedirs(path, exist_ok=True)
    #         xrec, qloss, _ = self(x)
    #         xrec.add_(1).mul_(0.5)

    # #         X = x_.squeeze().mul(255).add(0.5).clamp(0,255).to(torch.uint8).cpu().numpy()
    #         Y = xrec.squeeze().mul(255).add(0.5).clamp(0,255).to(torch.uint8).cpu().numpy()
    # #         maps = SSIM(X,Y,win_size=11,gaussian=False,full=True,use_sample_covariance=True)[1]
    # #         maps = 255 - np.uint8(maps*255+0.5)
    # #         ret, th = cv.threshold(maps, 0, 255, cv.THRESH_OTSU)
    # #         th = torch.from_numpy(th[None,None,:]).div(255.).cuda()
    # #         grid = torch.cat((x_,xrec,th))
    #         torchvision.utils.save_image(Y,os.path.join(path, f"{self.current_epoch:04}_{batch_idx:04}.jpg"),nrow=1,padding=0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                     list(self.decoder.parameters()) +
                                     list(self.quantize.parameters()) +
                                     list(self.quant_conv.parameters()) +
                                     list(self.post_quant_conv.parameters()),
                                     lr=self.learning_rate, betas=(0.5, 0.9))
        #         lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 40, 120)
        return [optimizer], []


class GumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)  # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.temperature_scheduling()
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


class EMAVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )
        self.quantize = EMAVectorQuantizer(n_embed=n_embed,
                                           embedding_dim=embed_dim,
                                           beta=0.25,
                                           remap=remap)

    def configure_optimizers(self):
        lr = self.learning_rate
        # Remove self.quantize from parameter list since it is updated via EMA
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
