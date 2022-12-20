# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import random as r
import torch
from copy import deepcopy
from torch.nn import functional as F
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from training.networks import transform_pos_to_cond
from torch_utils.ops import upfirdn2d

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, G_vr, G_superres_0, G_superres_1, D, augment_pipe=None, style_mixing_prob=0, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.G_vr = G_vr
        self.G_superres_0 = G_superres_0
        self.G_superres_1 = G_superres_1
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg

    def run_G(self, z, c, sync, swap_prob=.5):
        with torch.no_grad():
            _c = c.detach().clone()
            # index -> mapped_index
            B = _c.size(0)
            ix = np.array(list(range(B)))
            will_swap = np.random.random(B) <= swap_prob
            after_swap = np.random.permutation(ix[will_swap])
            ix[will_swap] = after_swap
            _c = _c[ix]
            cond = transform_pos_to_cond(_c)
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, cond)
            if False: # self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), cond, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            planes = self.G_synthesis(ws[:, :-5])
        with misc.ddp_sync(self.G_vr, sync):
            xy_plane, yz_plane, xz_plane = torch.chunk(planes, 3, dim=1)
            features = self.G_vr(xy_plane, yz_plane, xz_plane, c)["img"]
            # features = F.interpolate(features, (128, 128), mode='bilinear')
            low_img = features[:, :3]
        with misc.ddp_sync(self.G_superres_0, sync):
            sup_ws = ws[:, -5:]
            block_super_ws = [sup_ws.narrow(1, 0, 3), sup_ws.narrow(1, 2, 3)]
            x, high_img = features, low_img
            x, high_img = self.G_superres_0(x, high_img, block_super_ws[0])
        with misc.ddp_sync(self.G_superres_1, sync):
            x, high_img = self.G_superres_1(x, high_img, block_super_ws[1])
        return low_img, high_img, ws

    def run_D(self, img, c, sync, blur_sigma=0):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, cur_nimg):
        # Used when Resuming
        # cur_nimg = cur_nimg + 25000000
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        swap_prob = 1. - 0.5 * min(cur_nimg / 1e3 * 1e3, 1)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_low_img, gen_high_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl), swap_prob=swap_prob) # May get synced by Gpl.
                gen_img = torch.cat((F.interpolate(gen_low_img, gen_high_img.shape[-2:], mode='bilinear'), gen_high_img), dim=1)
                gen_logits = self.run_D(gen_img, gen_c, sync=False, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if False: # do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_low_img, gen_high_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                gen_img = torch.cat((F.interpolate(gen_low_img, gen_high_img.shape[-2:], mode='bilinear'), gen_high_img), dim=1)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_low_img, gen_high_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False, swap_prob=swap_prob)
                gen_img = torch.cat((F.interpolate(gen_low_img, gen_high_img.shape[-2:], mode='bilinear'), gen_high_img), dim=1)
                gen_logits = self.run_D(gen_img, gen_c, sync=False, blur_sigma=blur_sigma) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                with torch.no_grad():
                    real_img_tmp_low = F.interpolate(real_img_tmp, (128, 128), mode='bilinear')
                    real_img_tmp = torch.cat((F.interpolate(real_img_tmp_low, real_img_tmp.shape[-2:], mode='bilinear'), real_img_tmp), dim=1)
                real_img_tmp = real_img_tmp.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
