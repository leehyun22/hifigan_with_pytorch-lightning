import pytorch_lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aim import Image
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import itertools
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from stft_loss import MultiResolutionSTFTLoss
from audio_processing import mel_spectrogram
from models import discriminator_loss, generator_loss, feature_loss

class HiFiGanModuel(L.LightningModule):
    def __init__(self, configs: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.configs = configs
        self.mel_param = configs.data.melkwargs
        # networks
        self.generator = Generator(configs.model)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        self.stft_loss = MultiResolutionSTFTLoss()
        
        self.val_losses = []
    
    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch):
        x, y, _, y_mel = batch
        
        y = y.unsqueeze(1)
        
        y_g_hat = self.generator(x)
        
        sc_loss, mag_loss = self.stft_loss(
            y_g_hat[:, :, :y.size(2)].squeeze(1), y.squeeze(1)
        )
        
        y_g_hat_mel = mel_spectrogram(
            y_g_hat.squeeze(1),
            self.mel_param.n_fft,
            self.mel_param.n_mels,
            self.mel_param.sampling_rate,
            self.mel_param.hop_length,
            self.mel_param.win_length,
            self.mel_param.f_min,
            self.mel_param.fmax_for_loss,
        )

        optimizer_g, optimizer_d = self.optimizers()
        
        self.toggle_optimizer(optimizer_d)
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
        loss_disc_f, _, _ = discriminator_loss(
            y_df_hat_r, y_df_hat_g)
        
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
        loss_disc_s, _, _ = discriminator_loss(
            y_ds_hat_r, y_ds_hat_g)
        
        loss_disc_all = loss_disc_s + loss_disc_f
        
        self.log('train_d_loss', loss_disc_all, prog_bar=True)
        self.manual_backward(loss_disc_all)
        optimizer_d.step()
        optimizer_d.zero_grad()
        
        self.untoggle_optimizer(optimizer_d)
        
        self.toggle_optimizer(optimizer_g)
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
        _, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
        _, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        
        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, _ = generator_loss(y_ds_hat_g)
        loss_gen_all = (
            loss_gen_s + loss_gen_f + loss_fm_s + 
            loss_fm_f + loss_mel + sc_loss + mag_loss
        )

        self.log('train_g_loss', loss_gen_all, prog_bar=True)
        self.log('train_loss_mel', loss_mel.item())
        self.log('train_loss_stft', sc_loss.item() + mag_loss.item())
        self.manual_backward(loss_gen_all)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
    
    def on_train_epoch_end(self):
        scheduler_g, scheduler_d = self.lr_schedulers()
        g_lr, d_lr = self.get_lr()
        self.log("d_learning_rate", d_lr)
        self.log("g_learning_rate", g_lr)
        scheduler_g.step()
        scheduler_d.step()
    
    def validation_step(self, batch, batch_idx):
        x, y, _, y_mel = batch
        y_g_hat = self.generator(x)
        y_g_hat_mel = mel_spectrogram(
            y_g_hat.squeeze(1),
            self.mel_param.n_fft,
            self.mel_param.n_mels,
            self.mel_param.sampling_rate,
            self.mel_param.hop_length,
            self.mel_param.win_length,
            self.mel_param.f_min,
            self.mel_param.fmax_for_loss,
        )
        val_loss = F.l1_loss(y_mel, y_g_hat_mel).item()
        self.log('val_loss', val_loss)
        self.val_losses.append(val_loss)
        self.real_mel = y_mel[0]
        self.gen_mel = y_g_hat_mel[0]
    
    def on_train_start(self):
        # https://github.com/Lightning-AI/lightning/issues/12812 (resume train)
        self.optimizers()[0].param_groups = self.optimizers()[0]._optimizer.param_groups
        self.optimizers()[1].param_groups = self.optimizers()[1]._optimizer.param_groups

    def on_validation_epoch_end(self):
        avg_loss = sum(self.val_losses) / len(self.val_losses)
        self.val_losses.clear()
        self.log('val_avg_loss', avg_loss)
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # axes[0].imshow(self.real_mel.detach().cpu().numpy(), cmap='viridis', origin='lower', aspect='auto')
        # axes[0].set_title('real_mel')

        # axes[1].imshow(self.gen_mel.detach().cpu().numpy(), cmap='viridis', origin='lower', aspect='auto')
        # axes[1].set_title('gen_mel')
        # aim_figure = Image(fig)
        # self.logger.experiment.track(aim_figure, name='mel_spec_image')
        
        # log sampled images

    def configure_optimizers(self):
        lr = self.configs.model.learning_rate
        b1 = self.configs.model.adam_b1
        b2 = self.configs.model.adam_b2
        # lr_decay = self.configs.model.lr_decay

        optim_g = torch.optim.AdamW(
            self.generator.parameters(), lr, betas=[b1, b2])
        optim_d = torch.optim.AdamW(
            itertools.chain(self.msd.parameters(), self.mpd.parameters()),
            lr, betas=[b1, b2])
        
        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_g,
            eta_min=1e-5,
            T_max=50,
        )
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_d,
            eta_min=1e-5,
            T_max=50,
        )
        
        return [optim_g, optim_d], [scheduler_g, scheduler_d]

    def get_lr(self):
        optim_g, optim_d = self.optimizers()
        for g in optim_g.param_groups:
            g_lr = g['lr']
        
        for d in optim_d.param_groups:
            d_lr = d['lr']
        
        return g_lr, d_lr


