import os
import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn import metrics
import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from main import instantiate_from_config
from random import shuffle, random
from taming.modules.losses.ssim import SSIM


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image"):
        super().__init__()

        self.first_stage_key = first_stage_key

        self.init_first_stage_from_ckpt(first_stage_config)
        self.suffix = 'rgb' if self.first_stage_model.encoder.in_channels == 3 else 'gray'
        self.classname = self.first_stage_model.classname
        if permuter_config is None:
            permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def forward(self, x, class_label, train=True, mode='prob_mean'):
        assert mode in ('prob_mean', 'step', 'step_mean', 'down', 'down_mean')
        # one step to produce the logits
        z_indices, d, z_shape = self.encode_to_z(x)
        device = z_indices.device
        bsz, T = z_indices.shape
        b, c, h, w = z_shape
        assert h * w == T
        if train:
            new_z_indices = (self.transformer.config.vocab_size + 1) * torch.ones((bsz, T + 1), dtype=torch.int64).to(
                device)
            new_z = z_indices.clone().detach()
            cand = int(T * 0.3)
            pos = torch.zeros((bsz, cand), device=device, dtype=torch.int64)
            mask_target = torch.zeros((bsz, cand), device=device, dtype=torch.int64)
            mask = torch.zeros((bsz, T), device=device, dtype=torch.bool)
            for _ in range(bsz):
                permute_idx = torch.randperm(T, device=device, dtype=torch.int64)
                samples = permute_idx[:cand]
                pos[_, :] = samples
                mask_target[_, :] = z_indices[_, :].gather(0, samples)
                for i in samples:
                    mask[_, i] = True
                    if random() <= 0.8:
                        new_z[_, i] = self.transformer.config.vocab_size
                    elif random() <= 0.5:
                        new_z[_, i] = torch.randint(self.transformer.config.vocab_size, (1,), device=device)

            new_z_indices[:, 1:] = new_z
            logit_score, loss = self.transformer(new_z_indices, pos, mask_target, class_label)
            return logit_score, loss, z_indices, z_shape, mask
        else:
            # 依据采样概率决定mask位置,依概率采样4次后平均
            if mode == 'prob_mean':
                d = d.reshape(bsz, -1)
                pos_mask = torch.where(d < 0.55, True, False)
                new_z = torch.where(pos_mask, self.transformer.config.vocab_size, z_indices)
                new_z_indices = (self.transformer.config.vocab_size + 1) * torch.ones((bsz, T + 1),
                                                                                      dtype=torch.int64).to(
                    device)
                new_z_indices[:, 1:] = new_z
                logit_score, loss_classfiy = self.transformer(new_z_indices, None, None, class_label)
                return logit_score, loss_classfiy, z_indices, z_shape, pos_mask

            elif mode in ('step', 'step_mean'):
                logit_scores = []
                masks = []
                classfiy_loss = 0.0
                # 将索引特征图均分成4分，步长为14，生成4张重构图片，取这4张图片中mask部分组合成一张图片/直接平均4张图片
                for i in range(4):
                    new_z_indices = (self.transformer.config.vocab_size + 1) * torch.ones((bsz, T + 1),
                                                                                          dtype=torch.int64).to(
                        device)
                    new_z = z_indices.clone().reshape(bsz, h, w).detach()

                    step = h // 2  # 14
                    mask = torch.zeros((bsz, 1, h, w), device=device, dtype=torch.bool)
                    row_left = step * (i // 2)
                    row_right = step * (i // 2 + 1)
                    col_up = step * (i % 2)
                    col_down = step * (i % 2 + 1)
                    mask[:, :, row_left:row_right, col_up:col_down] = True
                    new_z[:, row_left:row_right, col_up:col_down] = self.transformer.config.vocab_size

                    new_z_indices[:, 1:] = new_z.reshape(bsz, -1)

                    logit_score, loss_classfiy = self.transformer(new_z_indices, None, None, class_label)
                    logit_scores.append(logit_score)
                    masks.append(mask)
                    classfiy_loss += loss_classfiy

                return torch.cat(logit_scores), classfiy_loss / 4.0, z_indices, [4 * b, c, h, w], torch.cat(masks)

            else:
                logit_scores = []
                masks = []
                classfiy_loss = 0.0
                # 将索引特征图的奇数列、偶数列、奇数行、偶数行分别mask

                ops = ['::2, :', '1::2, :', ':, ::2', ':, 1::2']
                for i in range(4):
                    new_z_indices = (self.transformer.config.vocab_size + 1) * torch.ones((bsz, T + 1),
                                                                                          dtype=torch.int64).to(
                        device)
                    new_z = z_indices.clone().reshape(bsz, h, w).detach()

                    mask = torch.zeros((bsz, 1, h, w), device=device, dtype=torch.bool)

                    exec('mask[:, :,' + ops[i] + '] = True')
                    exec('new_z[:,' + ops[i] + '] = self.transformer.config.vocab_size')

                    new_z_indices[:, 1:] = new_z.reshape(bsz, -1)

                    logit_score, loss_classfiy = self.transformer(new_z_indices, None, None, class_label)
                    logit_scores.append(logit_score)
                    masks.append(mask)
                    classfiy_loss += loss_classfiy

                return torch.cat(logit_scores), classfiy_loss / 4.0, z_indices, [4 * b, c, h, w], torch.cat(masks)

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, class_label, train, mode, temperature=0.8, top_k=4):
        assert not self.transformer.training
        bsz = x.size(0)
        if not train:
            logits, _, z_index, z_shape, spatial_masks = self(x, class_label, train, mode)
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
                probs = F.softmax(logits, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)

            if mode == 'prob_mean':
                probs = probs.view(-1, self.transformer.config.vocab_size)
                x_rec = []
                for _ in range(4):
                    sample_enc = probs.multinomial(1).view(bsz, -1)
                    new_z = torch.where(spatial_masks, sample_enc, z_index)
                    x_sample = self.decode_to_img(new_z, z_shape)
                    x_rec.append(x_sample)

                return z_index, z_shape, torch.stack(x_rec).mean(0)
            else:
                greedy_encs = probs.topk(1)[1].squeeze()
                pos_masks = spatial_masks.reshape(4 * bsz, -1)
                new_z = torch.where(pos_masks, greedy_encs, torch.cat([z_index.clone() for _ in range(4)]))

                x_samples = self.decode_to_img(new_z, z_shape)
                b, c, h, w = x_samples.shape

                if mode in ('step_mean', 'down_mean'):
                    x_sample = x_samples.reshape(4, -1, c, h, w).mean(0)
                    return z_index, z_shape, x_sample
                else:
                    spatial_masks = spatial_masks.to(torch.float32)
                    new_masks = F.interpolate(spatial_masks, size=self.first_stage_model.encoder.resolution,
                                              mode='bilinear', align_corners=False)

                    x_rec = x_samples * new_masks
                    x_sub = x_rec.reshape(4, -1, c, h, w)
                    x_sample = x_sub.sum(0)

                    return z_index, z_shape, x_sample

        else:
            logits, _, z_index, z_shape, pos_mask = self(x, class_label, train)
            logits = logits / temperature
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
                probs = F.softmax(logits, dim=-1)

            else:
                probs = F.softmax(logits, dim=-1)
            greedy_enc = probs.topk(1)[1].squeeze()
            if x.size(0) == 1:
                greedy_enc = greedy_enc.unsqueeze(0)
            new_z = torch.where(pos_mask, greedy_enc, z_index)
            x_sample = self.decode_to_img(new_z, z_shape)

        return z_index, z_shape, x_sample

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info, d = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return indices, d, quant_z.shape

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self, batch, mode='prob_mean', temperature=0.8, top_k=8, train=True, **kwargs):
        log = dict()

        N = 4

        x = self.get_input(self.first_stage_key, batch)
        class_label = batch['class_id'][:N].to(self.device)
        x = x[:N].to(device=self.device)

        z_indices, z_shape, ensemble_x = self.sample(x, class_label, train, mode,
                                                     temperature=temperature if temperature is not None else 1.0,
                                                     top_k=top_k if top_k is not None else 8,
                                                     )

        # reconstruction
        b, c, h, w = z_shape
        if train or mode == 'prob_mean':
            new_z_shape = (b, c, h, w)
        else:
            new_z_shape = (b // 4, c, h, w)
        x_rec = self.decode_to_img(z_indices, new_z_shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec
        log["topk_sample"] = ensemble_x

        return log

    def get_input(self, key, batch):
        x = batch[key]
        #         if len(x.shape) == 3:
        #             x = x[..., None]
        #         if len(x.shape) == 4:
        #             x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    #     def get_xc(self, batch, N=None):
    #         x = self.get_input(self.first_stage_key, batch)
    #         if N is not None:
    #             x = x[:N]
    #         return x

    #     def shared_step(self, batch, batch_idx):
    #         x = self.get_xc(batch)
    #         logit, loss, mask_pos = self(x)
    #         return logit, loss, mask_pos

    def training_step(self, batch, batch_idx):
        x = self.get_input(self.first_stage_key, batch)
        class_label = batch['class_id']
        _, loss, _, _, _ = self(x, class_label, train=True)
        self.log("train/bert_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(self.first_stage_key, batch)
        class_label = batch['class_id']
        _, loss, _, _, _ = self(x, class_label, train=False)
        self.log("val/bert_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x = self.get_input(self.first_stage_key, batch)
        class_label = batch['class_id']
        device = x.device
        img_label = batch['label']
        name = batch['name']
        classname = batch['classname']
        mask = batch['mask']
        _, _, ensemble_x = self.sample(x, class_label, train=False,
                                       temperature=0.8,
                                       top_k=8, mode='down_mean')

        rec_path = f'rec_imgs/{self.classname}/transformer_{self.suffix}/{classname[0]}_test'
        os.makedirs(rec_path, exist_ok=True)

        x.add_(1.0).mul_(0.5)
        ensemble_x.add_(1.0).mul_(0.5)

        ssim = SSIM(win_size=11, sigma=1.5, in_channel=x.size(1)).to(device)

        def helper(x, y):
            maps = ssim(x, y, data_range=1.0, use_pad=True, return_full=True).mean(1)
            maps = maps.squeeze().cpu().numpy()
            maps = 255 - np.uint8(maps * 255)
            thres, res_map = cv.threshold(maps, 0, 255, cv.THRESH_OTSU)  # cv.THRESH_TRIANGLE, cv.THRESH_OTSU

            num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(res_map, connectivity=8)
            area_list = [stats[i][-1] for i in range(num_labels)]
            max_area = max(area_list[1:])
            return thres, max_area, res_map, labels, stats

        def dice_coef(pred, target, smooth=0.01):
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()

            return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

        thres, max_area, res_map, labels, stats = helper(x, ensemble_x)

        # image_filtered = np.zeros_like(res_map)
        # for (i, label) in enumerate(np.unique(labels)):
        #     # 如果是背景，忽略
        #     if label == 0:
        #         continue
        #     if stats[i][-1] >= 137:
        #         image_filtered[labels == i] = 255
        #
        if self.suffix == 'gray':
            res_map = torch.from_numpy(res_map[None, None, :]).div(255.).to(device)
            grid = torch.cat((x, ensemble_x, mask, res_map))
            torchvision.utils.save_image(grid, os.path.join(rec_path, name[0]), nrow=4)
        else:
            f, axes = plt.subplots(1, 4)
            x = np.clip(x.squeeze(0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
            xrec = np.clip(ensemble_x.squeeze(0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
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

        return {'label': img_label, 'max_area': max_area, 'thres': thres, 'class': classname[0]}  # 'dice': dice

    def test_epoch_end(self, outputs):
        labels = np.array([x['label'].cpu().numpy() for x in outputs]).ravel()
        thres = np.array([x['thres'] for x in outputs])
        max_areas = np.array([x['max_area'] for x in outputs])
        classnames = np.array([x['class'] for x in outputs])

        # dice = np.array([x['dice'] for x in outputs]).mean()
        # print(f'dice: {dice}')

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
        ostu_info.to_csv(f'ostu_transformer.csv')
        area_info.to_csv(f'area_transformer.csv')
        print(f"ostu mean auc : {ostu_info['auroc'].mean()}")
        print(f"area mean auc : {area_info['auroc'].mean()}")

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer
