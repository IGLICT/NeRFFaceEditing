import os
import copy
from typing import List
import lpips
import torch

from torch import nn
from torch.nn import functional as F

import pickle
import dnnlib
import numpy as np
from PIL import Image
from typing import List

from utils import *
from camera_utils import FOV_to_intrinsics, LookAtPoseSampler

CHECKPOINTS = {
    0: './networks/NeRFFaceEditing-ffhq-64.pkl', 
    1: './networks/ffhqrebalanced512-128.pkl', 
}

class Projector(object):
    def __init__(self, device: torch.device, checkpoint_type: int=0, home: str = "./"):
        self.device = device
        self.lpips_loss = lpips.LPIPS(net='alex').to(device)
        self.synthesis_kwargs = {'noise_mode': 'const'}
        
        print("Load Generator ...")
        with open(os.path.join(home, CHECKPOINTS[checkpoint_type]), 'rb') as f:
            self.G = pickle.load(f)['G_ema'].to(device).eval()
        self.G.rendering_kwargs['depth_resolution'] = 96
        self.G.rendering_kwargs['depth_resolution_importance'] = 96
        
        self.fov_deg = 18.837
        self.intrinsics = FOV_to_intrinsics(self.fov_deg, device=device)
        self.cam_pivot = torch.tensor(self.G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        self.cam_radius = self.G.rendering_kwargs.get('avg_camera_radius', 2.7)
        self.conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, self.cam_pivot, radius=self.cam_radius, device=device)
        self.conditioning_params = torch.cat([self.conditioning_cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1, 9)], 1)
        
        print("Load VGG16 ...")
        vgg16_path = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(vgg16_path) as f:
            self.vgg16 = torch.jit.load(f).eval().to(device)
        
        print("Init Inversion ...")
        self.w_avg_samples              = 10000
        self.num_steps                  = 500
        self.initial_learning_rate      = 0.01
        self.initial_noise_factor       = 0.05
        self.lr_rampdown_length         = 0.25
        self.lr_rampup_length           = 0.05
        self.noise_ramp_length          = 0.75
        self.regularize_noise_weight    = 1e5

        with torch.no_grad():
            # Compute w stats.
            self.z_samples = np.random.RandomState(123).randn(self.w_avg_samples, self.G.z_dim)
            self.w_samples = self.mapping(self.G, torch.from_numpy(self.z_samples).to(device), self.conditioning_params.expand(self.w_avg_samples, -1), truncation_psi=1.)
            self.w_samples = self.w_samples[:, :1, :].cpu().numpy().astype(np.float32)
            self.w_avg = np.mean(self.w_samples, axis=0, keepdims=True)
            self.w_std = (np.sum((self.w_samples - self.w_avg) ** 2) / self.w_avg_samples) ** 0.5
        
        self.steps = 500
    @staticmethod
    def mapping(G, z: torch.Tensor, conditioning_params: torch.Tensor, truncation_psi=1.):
        return G.backbone.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=14, update_emas=False)
    @staticmethod
    def encode(G, ws, **synthesis_kwargs):
        planes = G.backbone.synthesis(ws, update_emas=False, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return planes.detach().clone()
    @staticmethod
    def decode(G, 
        ws: torch.Tensor, 
        cam: torch.Tensor, 
        planes: torch.Tensor, 
        **synthesis_kwargs
    ):
        cam2world_matrix = cam[:, :16].view(-1, 4, 4)
        intrinsics = cam[:, 16:25].view(-1, 3, 3)
        neural_rendering_resolution = G.neural_rendering_resolution
        
        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = G.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        N, M, _ = ray_origins.shape
        
        # Perform volume rendering
        feature_samples = G.renderer(planes, G.decoder, ray_origins, ray_directions, G.rendering_kwargs)[0]
        
        # Reshape into 'raw' neural-rendered image
        H = W = G.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        
        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = G.superresolution(
            rgb_image, 
            feature_image, 
            ws, 
            noise_mode=G.rendering_kwargs['superresolution_noise_mode'], 
            **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'}
        )
        
        return {
            'image_raw': rgb_image, 
            'image': sr_image, 
        }
    def __call__(self, img_s: List[Image.Image], pose_s: List[List[float]], w_opt_s=None):
        for img in img_s:
            assert img.size == (512, 512)
        
        for pose_idx in range(len(pose_s)):
            if pose_s[pose_idx] == None:
                pose_s[pose_idx] = self.conditioning_params.cpu().numpy().reshape(-1).tolist()
        
        camera_params = torch.from_numpy(np.array(pose_s, dtype=np.float32)).to(self.device).reshape(-1, 25)
        
        # Compute w pivot
        G = copy.deepcopy(self.G).eval().requires_grad_(False)
        
        if w_opt_s == None:
            start_w = self.w_avg
            noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}
            
            target_images = torch.stack([torch.from_numpy(np.array(img, dtype=np.uint8).transpose([2, 0, 1])).to(self.device).to(torch.float32) for img in img_s])
            target_images = F.interpolate(target_images, size=(256, 256), mode='area')
            target_features = self.vgg16(target_images, resize_images=False, return_lpips=True)
            
            with torch.no_grad():
                w_opt = torch.tensor(start_w, dtype=torch.float32, device=self.device).repeat(len(img_s), 1, 1)
            w_opt.requires_grad_(True)
            optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=self.initial_learning_rate)
            
            for step in range(self.steps):
                # Learning rate schedule.
                t = step / self.steps
                w_noise_scale = self.w_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
                lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
                lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
                lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
                lr = self.initial_learning_rate * lr_ramp
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # Synth images from opt_w.
                w_noise = torch.randn_like(w_opt) * w_noise_scale
                ws = (w_opt + w_noise).repeat([1, G.backbone.mapping.num_ws, 1])
                
                synth_images = G.synthesis(ws, camera_params, **self.synthesis_kwargs)["image"]

                # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
                synth_images = (synth_images + 1) * (255/2)
                synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')
                
                # Features for synth images.
                synth_features = self.vgg16(synth_images, resize_images=False, return_lpips=True)
                dist = (target_features - synth_features).square().sum()
                
                # Noise regularization.
                reg_loss = 0.0
                for v in noise_bufs.values():
                    noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                    while True:
                        reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                        reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                        if noise.shape[2] <= 8:
                            break
                        noise = F.avg_pool2d(noise, kernel_size=2)
                
                loss = dist + reg_loss * self.regularize_noise_weight
                
                # Step
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            
            w_opt = w_opt.detach().repeat([1, G.backbone.mapping.num_ws, 1]).requires_grad_(False)
        else:
            w_opt = w_opt_s.detach().requires_grad_(False)
        
        G.train().requires_grad_(True)
        # G.superresolution.requires_grad_(False)
        optimizer = torch.optim.Adam(G.parameters(), lr=3e-4)
        
        for step in range(self.steps):
            with torch.no_grad():
                target_images = torch.stack([torch.from_numpy(
                    np.array(img, dtype=np.uint8).transpose([2, 0, 1])
                ).to(self.device).to(torch.float32) / 255. * 2 - 1 for img in img_s])
            
            synth_images = G.synthesis(w_opt, camera_params, **self.synthesis_kwargs)["image"]
            
            l2_loss_val = torch.nn.L1Loss()(synth_images, target_images)
            loss_lpips = self.lpips_loss(synth_images, target_images).mean()
            
            loss = l2_loss_val * 1. + loss_lpips * 1.
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            B = w_opt.size(0)
            out = G.synthesis(w_opt, camera_params, **self.synthesis_kwargs)
            
        return G.eval().requires_grad_(False), w_opt.detach().clone(), [render_tensor(out["image"][b][None, ...].clamp(-1, 1)) for b in range(B)], [render_tensor(vis_parsing_maps(F.interpolate(out["image_seg"][b][None, ...], (512, 512), mode='bilinear'))) if "image_seg" in out else None for b in range(B)]