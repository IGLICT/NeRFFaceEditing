# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.renderer import ImportanceRenderer, DisentangledImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs           = {},
        disable_disentangle = False, 
        disable_alignment   = False, 
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.disable_disentangle = disable_disentangle
        self.disable_alignment = disable_alignment
        assert not self.disable_alignment or disable_disentangle
        
        self.renderer = DisentangledImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        if not self.disable_alignment:
            self.decoder = DisentangledOSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32, 'decoder_seg_dim': 15})
        else:
            self.decoder = SegmentationOSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32, 'decoder_seg_dim': 15})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
    
        self._last_planes = None
    def compute_mean_var(self, planes):
        """Compute the mean and variance of tri-plane"""
        mean = torch.mean(planes, dim=(-1, -2), keepdim=True)
        var = torch.sqrt(torch.var(planes, dim=(-1, -2), keepdim=True))
        return mean, var
    def normalize_plane(self, planes):
        """Normalize the tri-plane"""
        mean, var = self.compute_mean_var(planes)
        planes = (planes - mean) / (var + 1e-8)
        return planes, mean, var
    def denormalize_plane(self, planes, mean, var):
        """Denormalize the normalized tri-plane"""
        return planes * var + mean
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, planes_mean=None, planes_var=None, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        
        if not self.disable_disentangle:
            # Normalization Procedure
            norm_planes, mean, var = self.normalize_plane(planes)
            
            # Overrided Denormalization Procedure
            if planes_mean != None and planes_var != None:
                # Special case
                if type(planes_mean) == int and type(planes_var) == int:
                    planes = self.denormalize_plane(norm_planes, mean[planes_mean][None, ...], var[planes_var][None, ...])
                else:
                    planes = self.denormalize_plane(norm_planes, planes_mean, planes_var)
        else:
            norm_planes = None
            mean = None
            var = None
        
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        if not self.disable_disentangle:
            norm_planes = norm_planes.view(len(norm_planes), 3, 32, norm_planes.shape[-2], norm_planes.shape[-1])
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, seg_samples, depth_samples, weights_samples = \
            self.renderer(norm_planes if not self.disable_disentangle else planes, planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        seg_image = seg_samples.permute(0, 2, 1).reshape(N, seg_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {
            'image': sr_image, 
            'image_seg': seg_image, 
            'image_raw': rgb_image, 
            'image_depth': depth_image, 
            'plane_mean': mean, 
            'plane_var': var, 
        }
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if not self.disable_disentangle:
            norm_planes, _, _ = self.normalize_plane(planes)
            norm_planes = norm_planes.view(len(norm_planes), 3, 32, norm_planes.shape[-2], norm_planes.shape[-1])
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(norm_planes if not self.disable_disentangle else planes, planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if not self.disable_disentangle:
            norm_planes, _, _ = self.normalize_plane(planes)
            norm_planes = norm_planes.view(len(norm_planes), 3, 32, norm_planes.shape[-2], norm_planes.shape[-1])
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(norm_planes if not self.disable_disentangle else planes, planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, planes_mean=None, planes_var=None, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, planes_mean=planes_mean, planes_var=planes_var, **synthesis_kwargs)


from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

class SegmentationOSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
        self.seg_net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, options['decoder_seg_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_norm_features, sampled_denorm_features, ray_directions):
        # Aggregate features
        sampled_denorm_features = sampled_denorm_features.mean(1)
        x = sampled_denorm_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        
        x = sampled_denorm_features
        N, M, C = x.shape
        x = x.view(N*M, C)
        
        x = self.seg_net(x)
        x = x.view(N, M, -1)
        seg = x[..., :]
        
        return {'rgb': rgb, 'sigma': sigma, 'seg': seg}

class DisentangledOSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64
        
        self.geo_net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_seg_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
        self.app_net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_norm_features, sampled_denorm_features, ray_directions):
        # Aggregate features
        sampled_norm_features = sampled_norm_features.mean(1)
        sampled_denorm_features = sampled_denorm_features.mean(1)
        
        x = sampled_norm_features
        N, M, C = x.shape
        x = x.view(N*M, C)
        
        x = self.geo_net(x)
        x = x.view(N, M, -1)
        sigma = x[..., 0:1]
        seg = x[..., 1:]
        
        x = sampled_denorm_features
        N, M, C = x.shape
        x = x.view(N*M, C)
        
        x = self.app_net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x)*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        return {'rgb': rgb, 'sigma': sigma, 'seg': seg}