import os
dir_path = os.path.dirname(__file__)

#----------------------------------------------------------------------------
# Cropping the Image

import face_alignment
import numpy as np
import PIL
import scipy
ffhq_landmarks_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

def ffhq_image_align(src_img, face_landmarks, output_size=256, transform_size=1024, enable_padding=True):
        lm = np.array(face_landmarks)
        lm_chin          = lm[0  : 17, :2]  # left-right
        lm_eyebrow_left  = lm[17 : 22, :2]  # left-right
        lm_eyebrow_right = lm[22 : 27, :2]  # left-right
        lm_nose          = lm[27 : 31, :2]  # top-down
        lm_nostrils      = lm[31 : 36, :2]  # top-down
        lm_eye_left      = lm[36 : 42, :2]  # left-clockwise
        lm_eye_right     = lm[42 : 48, :2]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60, :2]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68, :2]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2
        
        img = src_img

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        return img

#----------------------------------------------------------------------------
# Estimate the Pose

import sys
sys.path.append(os.path.join(dir_path, "./external_dependencies"))
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from PIL import Image
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import detectors
from skimage.transform import estimate_transform, warp

class PoseEstimator(object):
    def __init__(self, device: torch.device):
        deca_cfg.model.use_tex = False
        self.deca = DECA(config = deca_cfg, device=device)
        self.scale = 1.25
        self.crop_size = 224
        self.resolution_inp = 224
        self.face_detector = detectors.FAN(device)
        self.device = device
    @staticmethod
    def bbox2point(left, right, top, bottom, type='bbox'):
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center
    def deca_image_align(self, image):
        h, w, _ = image.shape
        bbox, bbox_type = self.face_detector.run(image)
        if len(bbox) < 4:
            raise "Error: Unable to recognize at least one face."
        else:
            left = bbox[0]; right=bbox[2]
            top = bbox[1]; bottom=bbox[3]
        old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
        size = int(old_size*self.scale)
        src_pts = np.array([
            [center[0]-size/2, center[1]-size/2], 
            [center[0] - size/2, center[1]+size/2], 
            [center[0]+size/2, center[1]-size/2]
        ])

        DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image/255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2,0,1)
        return {
            'image': torch.tensor(dst_image).float(),
            'tform': torch.tensor(tform.params).float(),
            'original_image': torch.tensor(image.transpose(2,0,1)).float(),
        }
    def get_landmarks(self, image: torch.Tensor):
        with torch.no_grad():
            np_image = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            landmarks = self.face_detector.model.get_landmarks(np_image)
        return landmarks
    @staticmethod
    def transform_code_dict_to_pos(code_dict) -> torch.Tensor:
        with torch.no_grad():
            pos_yaw = code_dict["pose"][..., 1] + np.pi / 2
            pos_pitch = -code_dict["pose"][..., 0] + np.pi / 2
            pos_roll = code_dict["pose"][..., 2]
            scale = code_dict["cam"][..., 0]
            radius = 5.3041 / scale
            horizontal_shift = - code_dict["cam"][..., 1] * 0.53041 # -1: Nose at left side 1: Nose at right side (x - left ; + right) 
            vertical_shift = - code_dict["cam"][..., 2] * 0.53041 # -1: lower side 1: upper side (y - lower ; + upper)
            pos = torch.stack((pos_yaw, pos_pitch, pos_roll, torch.ones_like(radius), torch.zeros_like(horizontal_shift / radius), torch.zeros_like(vertical_shift / radius)), dim=-1)
            return pos
    def get_pose(self, img: Image.Image):
        sample = self.deca_image_align(np.array(img))
        landmark = self.get_landmarks(F.interpolate(sample['original_image'].unsqueeze(0), (224, 224))[0])
        landmark = torch.from_numpy(landmark[0] / 224 * 2 - 1).to(self.device)[None, ...] # (1, 68, 2)
        image = sample['image'].to(self.device)[None, ...]
        with torch.no_grad():
            codedicts = self.deca.encode(image)
        
        # Fitting the Landmarks
        codedicts['cam'].requires_grad_(True)
        codedicts['pose'].requires_grad_(True)
        optimizer = torch.optim.Adam([codedicts['cam'], codedicts['pose']], lr=0.1)
        for _ in range(100):
            optimizer.zero_grad()
            opdict = self.deca.decode(codedicts, rendering=False, vis_lmk=False, return_vis=False, use_detail=False)
            loss = nn.MSELoss()(opdict['landmarks2d'], landmark)
            loss.backward()
            optimizer.step()
        return self.transform_code_dict_to_pos(codedicts)

#----------------------------------------------------------------------------
# Projector
import dnnlib
from zoo import *

class Project(object):
    def __init__(self, zoo: ModelZoo):
        self.zoo = zoo
        self.device = zoo.device
        self.lpips_loss = zoo.lpips_loss
        self.synthesis_kwargs = {'noise_mode': 'const'}
        self.frontal_cam = zoo.frontal_cam
        self.G = zoo.get_EG3D(0)
        
        vgg16_path = os.path.join(dir_path, "external_dependencies", "data", "vgg16.pt")
        with dnnlib.util.open_url(vgg16_path) as f:
            self.vgg16 = torch.jit.load(f).eval().to(self.device)
        
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
            self.w_samples = mapping(self.G, torch.from_numpy(self.z_samples).to(zoo.device), self.frontal_cam.expand(self.w_avg_samples, -1), truncation_psi=1.)
            self.w_samples = self.w_samples[:, :1, :].cpu().numpy().astype(np.float32)
            self.w_avg = np.mean(self.w_samples, axis=0, keepdims=True)
            self.w_std = (np.sum((self.w_samples - self.w_avg) ** 2) / self.w_avg_samples) ** 0.5
        
        self.pti_steps = 350
        self.LPIPS_value_threshold = 0.06
    
    def project_w_pivot(self, G, image, pose):
        G = copy.deepcopy(G).eval().requires_grad_(False)
        target_cam = pose
        target_image = torch.from_numpy(np.array(image, dtype=np.uint8).transpose([2, 0, 1])).to(self.device).unsqueeze(0).to(torch.float32)
        target_image = F.interpolate(target_image, size=(256, 256), mode='area')
        target_features = self.vgg16(target_image, resize_images=False, return_lpips=True)
        
        w_opt = torch.tensor(self.w_avg, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=self.initial_learning_rate)
        
        for step in range(self.num_steps):
            # Learning rate schedule.
            t = step / self.num_steps
            w_noise_scale = self.w_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
            lr = self.initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Synth images from opt_w.
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
            synth_image = inference(G, ws, target_cam)[1]
            
            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            synth_image = (synth_image + 1) * (255/2)
            synth_image = F.interpolate(synth_image, size=(256, 256), mode='area')
            
            # Features for synth images.
            synth_features = self.vgg16(synth_image, resize_images=False, return_lpips=True)
            dist = (target_features - synth_features).square().sum()
            
            loss = dist
            
            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        num_ws = G.mapping.num_ws
        del G
        return w_opt.repeat([1, num_ws, 1]).detach().clone()
    def project(self, img: Image.Image, pose: torch.Tensor):
        assert img.size == (512, 512)
        w_pivot = self.project_w_pivot(self.G, img, pose)

        _G = copy.deepcopy(self.G).train().requires_grad_(True)
        _G.superres_0.requires_grad_(False)
        _G.superres_1.requires_grad_(False)
        optimizer = torch.optim.Adam(_G.parameters(), lr=3e-4)

        for _ in range(self.pti_steps):
            target_cam = pose
            with torch.no_grad():
                target_images = torch.from_numpy(
                    np.array(img, dtype=np.uint8).transpose([2, 0, 1])
                ).to(self.device).unsqueeze(0).to(torch.float32) / 255. * 2 - 1
            _, generated_images, _ = inference(_G, w_pivot, target_cam)
            
            l2_loss_val = torch.nn.L1Loss()(generated_images, target_images)
            loss_lpips = torch.squeeze(self.lpips_loss(generated_images, target_images))
            loss = l2_loss_val * 1. + loss_lpips * 1.
            
            if loss_lpips < self.LPIPS_value_threshold:
                break

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
        
        return _G.requires_grad_(False), w_pivot.detach().clone()

#----------------------------------------------------------------------------

class Projector(object):
    def __init__(self, zoo: ModelZoo):
        super().__init__()
        self.pose_estimator = PoseEstimator(zoo.device)
        self.project = Project(zoo)
    def forward(self, image: Image.Image):
        face_landmarks = ffhq_landmarks_detector.get_landmarks(np.array(image))[0]
        image = ffhq_image_align(image, face_landmarks, 512)
        pose = self.pose_estimator.get_pose(image)
        return image, pose, self.project.project(image, pose)