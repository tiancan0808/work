import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.lpips import LPIPS
from taming.modules.losses.ssim import SSIM
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init, add_sn


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, current_epoch, threshold=0, value=0.):
    if current_epoch < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = loss_real + loss_fake
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = torch.mean(torch.nn.functional.softplus(-logits_real)) + \
             torch.mean(torch.nn.functional.softplus(logits_fake))
    return d_loss


def ls_d_loss(logits_real, logits_fake):
    loss_real = F.mse_loss(logits_real, torch.ones_like(logits_real))
    loss_fake = F.mse_loss(logits_fake, torch.zeros_like(logits_fake))
    d_loss = loss_real + loss_fake
    return d_loss


def classfiy_loss(real, fake, label):
    real_loss = F.cross_entropy(real.squeeze(), label)
    fake_loss = F.cross_entropy(fake.squeeze(), label)
    return 0.5*(real_loss + fake_loss)


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0, win_size=9, sigma=1.0, alpha=0.5,
                 beta=1.0,
                 disc_num_layers=3, disc_in_channels=1, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, n_classes=1,
                 disc_ndf=64, disc_loss="ls"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", 'ls']
        self.disc_loss_type = disc_loss
        self.codebook_weight = codebook_weight
        self.ssim_loss = SSIM(win_size=win_size, sigma=sigma, in_channel=disc_in_channels)
        #         self.perceptual_loss = LPIPS().eval()\
        #         self.perceputal_loss = SSIM(win_size=win_size,sigma=sigma)
        self.perceptual_weight = perceptual_weight
        self.discriminator_img = NLayerDiscriminator(input_nc=disc_in_channels,
                                                     ndf=disc_ndf,
                                                     n_layers=disc_num_layers,
                                                     n_classes=n_classes).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.classfiy_loss = classfiy_loss
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == 'ls':
            self.disc_loss = ls_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.n_classes = n_classes
        self.win_size = win_size
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-5)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                current_epoch, info, label, last_layer=None, split="train"):

        rec_loss = self.ssim_loss(inputs, reconstructions, alpha=self.alpha, beta=self.beta)

        nll_loss = rec_loss  # + self.perceptual_weight * p_loss.mean()

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake_img, _ = self.discriminator_img(reconstructions.contiguous())

            if self.disc_loss_type == 'ls':
                g_loss_img = F.mse_loss(logits_fake_img, torch.ones_like(logits_fake_img))

            else:
                g_loss_img = -torch.mean(logits_fake_img)

            try:
                d_weight_img = self.calculate_adaptive_weight(nll_loss, g_loss_img, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight_img = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, current_epoch, threshold=self.discriminator_iter_start)
            loss = nll_loss + disc_factor * d_weight_img * g_loss_img + self.codebook_weight * codebook_loss.mean()

            log = {
                "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/perplexity".format(split): info[0].detach(),
                "{}/cluster_use".format(split): info[1].detach(),
                "{}/d_weight_img".format(split): d_weight_img.detach(),
                "{}/g_loss_img".format(split): g_loss_img.detach().mean(),
            }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real_img, classfiy_real = self.discriminator_img(inputs.contiguous().detach())
            logits_fake_img, classfiy_fake = self.discriminator_img(reconstructions.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, current_epoch, threshold=self.discriminator_iter_start)

            d_loss_img = disc_factor * self.disc_loss(logits_real_img, logits_fake_img)
            if self.n_classes > 1:
                classfiy_loss = disc_factor*self.classfiy_loss(classfiy_real, classfiy_fake, label)
            else:
                classfiy_loss = torch.tensor(0.0)
            d_loss = d_loss_img + classfiy_loss

            log = {"{}/d_loss_img".format(split): d_loss_img.clone().detach().mean(),
                   "{}/logits_real_img".format(split): logits_real_img.detach().mean(),
                   "{}/logits_fake_img".format(split): logits_fake_img.detach().mean(),
                   "{}/classfiy_loss".format(split): classfiy_loss.detach().mean(),
                   }
            return d_loss, log


class VQSSIMNoDiscriminator(nn.Module):
    def __init__(self, codebook_weight=1.0, pixelloss_weight=1.0, perceptual_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.perceptual_loss = SSIM()
        self.perceptual_weight = perceptual_weight

    def forward(self, codebook_loss, inputs, reconstructions, global_step, info, cond=None, split="train"):
        #         alpha = 0.5 if global_step < 5000 else 0.1
        #         beta = 1 if global_step < 5000 else 4
        alpha = 1.0
        beta = 10
        rec_loss = self.perceptual_weight * self.perceptual_loss(inputs, reconstructions, alpha=alpha, beta=beta)

        nll_loss = rec_loss
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        loss = nll_loss + self.codebook_weight * codebook_loss.mean()

        log = {
            "{}/quant_loss".format(split): codebook_loss.detach().mean(),
            "{}/nll_loss".format(split): rec_loss.detach().mean(),
            "{}/perplexity".format(split): info[0].detach(),
            "{}/cluster_use".format(split): info[1].detach(),
        }
        return loss, log


class aeloss(nn.Module):
    def __init__(self, codebook_weight=1.0, pixelloss_weight=1.0,
                 perceptual_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.perceptual_loss = SSIM()
        self.perceptual_weight = perceptual_weight

    def forward(self, inputs, reconstructions, global_step, split="train"):
        alpha = 0.5
        beta = 1.0
        rec_loss = self.perceptual_weight * self.perceptual_loss(inputs, reconstructions, alpha=alpha, beta=beta)

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)
        loss = nll_loss

        log = {"{}/nll_loss".format(split): rec_loss.detach().mean()}
        return loss, log
