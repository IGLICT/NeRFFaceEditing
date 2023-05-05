import os
import math
import torch
import imageio
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from camera_utils import FOV_to_intrinsics, LookAtPoseSampler

@torch.no_grad()
def render_tensor(img: torch.Tensor, normalize: bool = True, nrow: int = 8) -> Image.Image:
    if type(img) == list:
        img = torch.cat(img, dim=0).expand(-1, 3, -1, -1)
    elif len(img.shape) == 3:
        img = img.expand(3, -1, -1)
    elif len(img.shape) == 4:
        img = img.expand(-1, 3, -1, -1)
    
    img = img.squeeze()
    
    if normalize:
        img = img / 2 + .5
    
    if len(img.shape) == 3:
        return Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    elif len(img.shape) == 2:
        return Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8))
    elif len(img.shape) == 4:
        return Image.fromarray((make_grid(img, nrow=nrow).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

@torch.no_grad()
def render_video(
    G: torch.nn.Module, 
    fn: str, 
    ws: torch.Tensor, 
    norm_planes: torch.Tensor, 
    denorm_planes: torch.Tensor, 
    frames: int = 150, 
    fps: int = 30, 
    a_degree: float = 15., 
    b_degree: float = 12., 
    
    init_pitch: float = 5*np.pi/12, 
    init_yaw: float = np.pi/2, 
) -> None:
    frames_interp = frames // 4
    a = a_degree / 180 * np.pi
    b = b_degree / 180 * np.pi
    start_pitch = np.pi / 2 - a
    start_yaw = np.pi / 2
    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device=ws.device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=ws.device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    synthesis_kwargs = {"noise_mode": "const"}
    
    camera_schedulers = []
    
    if start_pitch != init_pitch or start_yaw != start_yaw:
        # Perform Interpolation
        for index in range(frames_interp):
            ratio = index / (frames_interp - 1)
            camera_schedulers.append((
                start_pitch * ratio + init_pitch * (1 - ratio), 
                start_yaw * ratio + init_yaw * (1 - ratio), 
            ))
    
    for index in range(frames):
        theta = index / (frames - 1) * 2 * np.pi
        camera_schedulers.append((
            np.pi / 2 - a * np.cos(theta), 
            np.pi / 2 + b * np.sin(theta), 
        ))
    
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    writer = imageio.get_writer(fn, fps=fps, quality=8)
    
    for pitch, yaw in camera_schedulers:
        cam = torch.cat([LookAtPoseSampler.sample(pitch, yaw, cam_pivot, radius=cam_radius, device=ws.device).reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        img = decode(G, ws, cam, norm_planes, denorm_planes, **synthesis_kwargs)["image"]
        img = np.asarray(img.cpu().numpy(), dtype=np.float32)
        img = (img - (-1)) * (255 / (1 - (-1)))
        img = np.rint(img).clip(0, 255).astype(np.uint8)[0]
        img = img.transpose(1, 2, 0)
        
        writer.append_data(img)
    
    writer.close()

@torch.no_grad()
def vis_parsing_maps(im: torch.Tensor, inverse: bool = False, argmax: bool = True):
    part_colors = [
        [0, 0, 0], # Background
        [127, 212, 255], # Skin
        [255, 212, 255], # Eye Brow
        [255, 255, 170], # Eye
        [255, 255, 130], # Glass
        [76, 153, 0], # Ear
        [0, 255, 170], # Ear Ring
        [244, 124, 244], # Nose
        [30, 162, 230], # Mouth
        [127, 255, 255], # Lip
        [127, 170, 255], # Neck
        [85, 0, 255], # Neck-lace
        [255, 170, 127], # Cloth
        [212, 127, 255], # Hair
        [0, 170, 255], # Hat
        [255, 255, 255]
    ]
    if inverse == False:
        if argmax:
            im = torch.argmax(im, dim=1, keepdim=True)
        out = torch.zeros((im.size(0), 3, im.size(2), im.size(3)), device=im.device, dtype=torch.float32)

        for index in range(len(part_colors)):
            color = torch.from_numpy(np.array(part_colors[index])).to(out.device).to(out.dtype).view(1, 3, 1, 1).expand_as(out)
            out = torch.where(im == index, color, out)

        out = out / 255.0 * 2 - 1
        return out
    else:
        out = torch.zeros((im.size(0), 1, im.size(2), im.size(3)), device=im.device, dtype=torch.int64)
        
        for index in range(len(part_colors)):
            color = torch.from_numpy(np.array(part_colors[index])).to(im.device).to(im.dtype).view(1, 3, 1, 1).expand_as(im) / 255.0 * 2 - 1
            out = torch.where(torch.all((im - color).abs() <= 1e-2, dim=1, keepdim=True), torch.ones((im.size(0), 1, im.size(2), im.size(3)), device=out.device, dtype=torch.int64) * index, out)
        
        return out

def get_camera_samples(G, device: torch.device):
    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    
    pitch_s = [5*np.pi/12, 6*np.pi/12, 7*np.pi/12]
    yaw_s = [5*np.pi/12, 6*np.pi/12, 7*np.pi/12]
    
    cam_s = []
    
    for pitch in pitch_s:
        for yaw in yaw_s:
            cam_s.append(torch.cat([LookAtPoseSampler.sample(pitch, yaw, cam_pivot, radius=cam_radius, device=device).reshape(-1, 16), intrinsics.reshape(-1, 9)], 1))
    return cam_s

def compute_mean_var(planes):
    # (N, 3, C, H, W)
    mean = torch.mean(planes, dim=(-1, -2), keepdim=True)
    var = torch.sqrt(torch.var(planes, dim=(-1, -2), keepdim=True))
    return mean, var

def normalize_plane(planes):
    mean, var = compute_mean_var(planes)
    planes = (planes - mean) / (var + 1e-8)
    return planes, mean, var

def denormalize_plane(planes, mean, var):
    return planes * var + mean

def encode(G, ws, **synthesis_kwargs):
    planes = G.backbone.synthesis(ws, **synthesis_kwargs)
    planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
    return planes

def decode(G, ws, cam, norm_planes, denorm_planes, **synthesis_kwargs):
    cam2world_matrix = cam[:, :16].view(-1, 4, 4)
    intrinsics = cam[:, 16:25].view(-1, 3, 3)
    neural_rendering_resolution = G.neural_rendering_resolution
    
    # Create a batch of rays for volume rendering
    ray_origins, ray_directions = G.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
    N, M, _ = ray_origins.shape
    
    # Perform volume rendering
    feature_samples, seg_samples, depth_samples, weights_samples = \
        G.renderer(norm_planes, denorm_planes, G.decoder, ray_origins, ray_directions, G.rendering_kwargs)
    
    # Reshape into 'raw' neural-rendered image
    H = W = G.neural_rendering_resolution
    feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
    seg_image = seg_samples.permute(0, 2, 1).reshape(N, seg_samples.shape[-1], H, W).contiguous()
    depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
    
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
        'image_depth': depth_image, 
        'image_seg': seg_image, 
    }