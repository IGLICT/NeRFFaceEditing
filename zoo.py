import os
import warnings
warnings.filterwarnings("ignore")
import copy
import pickle
import torch
import lpips
from PIL import Image
from torchvision.utils import make_grid
from types import MethodType
from training.networks import *

synthesis_kwargs = {'noise_mode': 'const', 'force_fp32': True}

#----------------------------------------------------------------------------
# Utility Functions

@torch.no_grad()
def transform_code_dict_to_pos(
    code_dict: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    This utility function transforms the `pose` predicted by FLAME into the pose used by our model when performing volume rendering.
    """
    pos_yaw = code_dict["pose"][..., 1] + np.pi / 2
    pos_pitch = -code_dict["pose"][..., 0] + np.pi / 2
    pos_roll = code_dict["pose"][..., 2]
    scale = code_dict["cam"][..., 0]
    radius = 5.3041 / scale
    horizontal_shift = - code_dict["cam"][..., 1] * 0.53041 # -1: Nose at left side 1: Nose at right side (x - left ; + right) 
    vertical_shift = - code_dict["cam"][..., 2] * 0.53041 # -1: lower side 1: upper side (y - lower ; + upper)
    pos = torch.stack(
        (
            pos_yaw, 
            pos_pitch, 
            pos_roll, 
            torch.ones_like(radius), 
            torch.zeros_like(horizontal_shift / radius), 
            torch.zeros_like(vertical_shift / radius), 
        ), dim=-1)
    return pos

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

def resample_semantic_masks(semantic_masks: torch.Tensor) -> torch.Tensor:
    if semantic_masks.size(1) > 1:
        # Usually for up-sampling
        return F.interpolate(semantic_masks, (512, 512), mode='bilinear')
    else:
        # Usually for down-sampling
        return F.interpolate(semantic_masks.float(), (128, 128), mode='nearest').long()

#----------------------------------------------------------------------------
# Calling function for EG3D & NeRFFaceEditing

def inference(G, w, pose, **kwargs):
    xy_plane, yz_plane, xz_plane = encode(G, w, **synthesis_kwargs)
    low_img, high_img, vr_out = decode(G, w, xy_plane, yz_plane, xz_plane, pose, **kwargs, **synthesis_kwargs)
    return low_img, high_img, vr_out

def mapping(G, z: torch.Tensor, c: torch.Tensor, truncation_psi=.5):
    with torch.no_grad():
        cond = transform_pos_to_cond(c)
    ws = G.mapping(z, cond, truncation_psi=truncation_psi, truncation_cutoff=None)
    return ws

def encode(G, ws, **synthesis_kwargs):
    planes = G.synthesis(ws[:, :-5], **synthesis_kwargs)
    xy_plane, yz_plane, xz_plane = torch.chunk(planes, 3, dim=1)
    return xy_plane, yz_plane, xz_plane

def decode(G, 
    ws: torch.Tensor, 
    xy_plane: torch.Tensor, 
    yz_plane: torch.Tensor, 
    xz_plane: torch.Tensor, 
    cam: torch.Tensor, 
    disable_super: bool = False, 
    double_sample_points: bool = False, 
    **synthesis_kwargs
):
    vr_out = G.vr(xy_plane, yz_plane, xz_plane, cam, double_sample_points=double_sample_points)
    features = vr_out["img"]
    low_img = features[:, :3]
    if disable_super == True:
        return low_img, None, vr_out
    
    sup_ws = ws[:, -5:].to(torch.float32)
    block_super_ws = [sup_ws.narrow(1, 0, 3), sup_ws.narrow(1, 2, 3)]
    x, high_img = features, low_img
    x, high_img = G.superres_0(x, high_img, block_super_ws[0], **synthesis_kwargs)
    x, high_img = G.superres_1(x, high_img, block_super_ws[1], **synthesis_kwargs)
    return low_img, high_img, vr_out

def project_point(
    ray: RayBundle, 
    xy_plane: torch.Tensor, 
    xz_plane: torch.Tensor, 
    yz_plane: torch.Tensor, 
) -> torch.Tensor:
    """
    Args:
        xy_plane: tensor of shape (B, C, N, N)
        xz_plane: tensor of shape (B, C, N, N)
        yz_plane: tensor of shape (B, C, N, N)
    Returns:
        tensor of shape (B, H * W, num_points_per_ray, C)
    """

    x = ray_bundle_to_ray_points(ray) # (B, H * W, num_points_per_ray, 3)

    normalize_x = lambda x: x / 0.2243 # 3
    normalize_y = lambda y: y / 0.2243 # 3
    normalize_z = lambda z: z / 0.2243 # 3

    norm_x = normalize_x(x[:, :, :, 0]) # (B, H*W, num)
    norm_y = normalize_y(x[:, :, :, 1]) # (B, H*W, num)
    norm_z = normalize_z(x[:, :, :, 2]) # (B, H*W, num)

    xy = torch.stack((norm_x, norm_y), dim=-1) # (B, H*W, num, 2)
    yz = torch.stack((norm_y, norm_z), dim=-1) # (B, H*W, num, 2)
    xz = torch.stack((norm_x, norm_z), dim=-1) # (B, H*W, num, 2)

    F_xy = F.grid_sample(xy_plane, xy, padding_mode="zeros").permute(0, 2, 3, 1) # (B, H*W, num, C)
    F_yz = F.grid_sample(yz_plane, yz, padding_mode="zeros").permute(0, 2, 3, 1) # (B, H*W, num, C)
    F_xz = F.grid_sample(xz_plane, xz, padding_mode="zeros").permute(0, 2, 3, 1) # (B, H*W, num, C)

    features = F_xy + F_yz + F_xz
    
    mask = torch.logical_and(
        torch.logical_and(
            torch.logical_and( norm_x > -1., norm_x < 1. ), 
            torch.logical_and( norm_y > -1., norm_y < 1. ), 
        ), 
        torch.logical_and( norm_z > -1., norm_z < 1. )
    )[..., None]
    features = features * mask
    return features

def vr_forward(self, 
    xy_plane: torch.Tensor, 
    yz_plane: torch.Tensor, 
    xz_plane: torch.Tensor, 
    camera: torch.Tensor, 
    double_sample_points: bool = False, 
):
    """
    Args:
        xy_plane: tensor of shape (B, C, N, N)
        xz_plane: tensor of shape (B, C, N, N)
        yz_plane: tensor of shape (B, C, N, N)
        camera: The tensor of shape (B, 6)
            yaw: The tensor of shape (B, 1) denoting the yaw angle in radian.
            pitch: The tensor of shape (B, 1) denoting the pitch angle in radian.
            roll: The tensor of shape (B, 1) denoting the roll angle in radian.
            radius: The tensor of shape (B, 1) denoting the radius.
            delta_x: The tensor of shape (B, 1).
            delta_y: The tensor of shape (B, 1).
        image_size: tuple as (H, W)
    """
    # Generate Rays
    yaw, pitch, roll, radius, delta_x, delta_y = torch.split(camera, 1, -1)
    delta_depth = get_corrected_delta_depth(yaw, pitch, roll, radius, delta_x, delta_y) + radius - 1

    batch_size = camera.size(0)
    with torch.no_grad():
        initial_rays = get_initial_rays(
            batch_size, 
            (self.img_resolution, self.img_resolution), 
            self.fov, 
            self.n_pts_per_ray * (1 if not double_sample_points else 2), 
            torch.ones((batch_size, 1), device=camera.device, dtype=camera.dtype) * self.min_depth + delta_depth, 
            torch.ones((batch_size, 1), device=camera.device, dtype=camera.dtype) * self.max_depth + delta_depth, 
            camera.device, 
        )

        # Transform rays into World Space
        coarse_rays = transform_rays(initial_rays, yaw, pitch, roll, radius, delta_x, delta_y, False)

    coarse_features = self.project_point(coarse_rays, xy_plane, xz_plane, yz_plane) # (B, H * W, num_points_per_ray, C)

    coarse_lengths = coarse_rays.lengths
    coarse_density, coarse_color = self.decoder(coarse_features)

    with torch.no_grad():
        _, _, weights = march_ray(coarse_density, coarse_color, coarse_lengths, density_noise_std=0.)
        lengths_mid = 0.5 * (coarse_lengths[..., 1:] + coarse_lengths[..., :-1]) # (B, H * W, num_points_per_ray - 1)
        fine_lengths = sample_pdf(lengths_mid, weights[..., 1:-1], self.n_pts_per_ray * (1 if not double_sample_points else 2)).detach() # (B, H * W, num_points_per_ray)
        fine_rays = RayBundle(coarse_rays.origins, coarse_rays.directions, fine_lengths, None)

    fine_features = self.project_point(fine_rays, xy_plane, xz_plane, yz_plane) # (B, H * W, num_points_per_ray, C)

    fine_lengths = fine_rays.lengths
    fine_density, fine_color = self.decoder(fine_features)

    with torch.no_grad():
        lengths = torch.cat((coarse_lengths, fine_lengths), dim=-1) # (B, H * W, num_points_per_ray * 2)
        lengths, indices = torch.sort(lengths, dim=-1)

    density = torch.cat((coarse_density, fine_density), dim=-2)
    color = torch.cat((coarse_color, fine_color), dim=-2)
    features = torch.cat((coarse_features, fine_features), dim=-2)

    density = torch.gather(density, -2, indices[..., None].expand_as(density))
    color = torch.gather(color, -2, indices[..., None].expand_as(color))
    features = torch.gather(features, -2, indices[..., None].expand_as(features))

    img, depth, weights = march_ray(density, color, lengths)
    img = img.reshape(img.size(0), self.img_resolution, self.img_resolution, -1).permute(0, 3, 1, 2).contiguous() # (B, 3, H, W)
    depth = depth.reshape(depth.size(0), self.img_resolution, self.img_resolution, -1).permute(0, 3, 1, 2).contiguous() # (B, 1, H, W)

    rays = RayBundle(coarse_rays.origins, coarse_rays.directions, lengths, None)

    return {
        "img": img, 
        "depth": depth, 
        "delta_depth": delta_depth, 
        "rays": rays, 
        "weights": weights, 
        "features": features, 
        "color": color, 
    }

#----------------------------------------------------------------------------
# Calling functions for NeRFFaceEditing

@torch.no_grad()
def extract_plane_features(G, ws):
    xy_plane, yz_plane, xz_plane = encode(G, ws.expand(-1, G.mapping.num_ws, -1), **synthesis_kwargs)
    
    _, xy_plane_mean, xy_plane_var = normalize_plane(xy_plane)
    _, yz_plane_mean, yz_plane_var = normalize_plane(yz_plane)
    _, xz_plane_mean, xz_plane_var = normalize_plane(xz_plane)
    return {
        "xy_plane_mean": xy_plane_mean, 
        "xy_plane_var": xy_plane_var, 
        "yz_plane_mean": yz_plane_mean, 
        "yz_plane_var": yz_plane_var, 
        "xz_plane_mean": xz_plane_mean, 
        "xz_plane_var": xz_plane_var, 
    }

def decode_with_features(G, 
    ws, 
    pose, 
    xy_plane_mean, 
    xy_plane_var, 
    yz_plane_mean, 
    yz_plane_var, 
    xz_plane_mean, 
    xz_plane_var, 
    **kwargs, 
):
    xy_plane, yz_plane, xz_plane = encode(G, ws.expand(-1, G.mapping.num_ws, -1), **synthesis_kwargs)
    
    norm_xy_plane, _, _ = normalize_plane(xy_plane)
    norm_yz_plane, _, _ = normalize_plane(yz_plane)
    norm_xz_plane, _, _ = normalize_plane(xz_plane)
    
    unified_xy_plane = denormalize_plane(norm_xy_plane, xy_plane_mean, xy_plane_var)
    unified_yz_plane = denormalize_plane(norm_yz_plane, yz_plane_mean, yz_plane_var)
    unified_xz_plane = denormalize_plane(norm_xz_plane, xz_plane_mean, xz_plane_var)
    
    low_img, high_img, vr_out = decode(
        G, 
        ws.expand(-1, G.mapping.num_ws, -1), 
        unified_xy_plane, 
        unified_yz_plane, 
        unified_xz_plane, 
        pose, 
        **kwargs, 
        **synthesis_kwargs
    )

    return low_img, high_img, vr_out["depth"], vr_out["seg"]

def compute_mean_var(planes):
    mean = torch.mean(planes, dim=(-1, -2), keepdim=True)
    var = torch.sqrt(torch.var(planes, dim=(-1, -2), keepdim=True))
    return mean, var

def normalize_plane(planes):
    mean, var = compute_mean_var(planes)
    planes = (planes - mean) / (var + 1e-10)
    return planes, mean, var

def denormalize_plane(planes, mean, var):
    return planes * var + mean

def vr_forward_with_adain_seg(self, 
    xy_plane: torch.Tensor, 
    yz_plane: torch.Tensor, 
    xz_plane: torch.Tensor, 
    camera: torch.Tensor, 
    double_sample_points: bool = False, 
):
    """
    Args:
        xy_plane: tensor of shape (B, C, N, N)
        xz_plane: tensor of shape (B, C, N, N)
        yz_plane: tensor of shape (B, C, N, N)
        camera: The tensor of shape (B, 6)
            yaw: The tensor of shape (B, 1) denoting the yaw angle in radian.
            pitch: The tensor of shape (B, 1) denoting the pitch angle in radian.
            roll: The tensor of shape (B, 1) denoting the roll angle in radian.
            radius: The tensor of shape (B, 1) denoting the radius.
            delta_x: The tensor of shape (B, 1).
            delta_y: The tensor of shape (B, 1).
        image_size: tuple as (H, W)
    """
    # Generate Rays
    yaw, pitch, roll, radius, delta_x, delta_y = torch.split(camera, 1, -1)
    delta_depth = get_corrected_delta_depth(yaw, pitch, roll, radius, delta_x, delta_y) + radius - 1

    batch_size = camera.size(0)
    with torch.no_grad():
        initial_rays = get_initial_rays(
            batch_size, 
            (self.img_resolution, self.img_resolution), 
            self.fov, 
            self.n_pts_per_ray * (1 if not double_sample_points else 2), 
            torch.ones((batch_size, 1), device=camera.device, dtype=camera.dtype) * self.min_depth + delta_depth, 
            torch.ones((batch_size, 1), device=camera.device, dtype=camera.dtype) * self.max_depth + delta_depth, 
            camera.device, 
        )

        # Transform rays into World Space
        coarse_rays = transform_rays(initial_rays, yaw, pitch, roll, radius, delta_x, delta_y, False)

    norm_xy_plane, _, _ = normalize_plane(xy_plane)
    norm_yz_plane, _, _ = normalize_plane(yz_plane)
    norm_xz_plane, _, _ = normalize_plane(xz_plane)
    
    coarse_norm_features = self.project_point(coarse_rays, norm_xy_plane, norm_xz_plane, norm_yz_plane)
    coarse_features = self.project_point(coarse_rays, xy_plane, xz_plane, yz_plane) # (B, H * W, num_points_per_ray, C)

    coarse_lengths = coarse_rays.lengths
    coarse_density, coarse_seg, coarse_color = self.decoder(coarse_norm_features, coarse_features)

    with torch.no_grad():
        _, _, weights = march_ray(coarse_density, coarse_color, coarse_lengths, density_noise_std=0.)
        lengths_mid = 0.5 * (coarse_lengths[..., 1:] + coarse_lengths[..., :-1]) # (B, H * W, num_points_per_ray - 1)
        fine_lengths = sample_pdf(lengths_mid, weights[..., 1:-1], self.n_pts_per_ray * (1 if not double_sample_points else 2)).detach() # (B, H * W, num_points_per_ray)
        fine_rays = RayBundle(coarse_rays.origins, coarse_rays.directions, fine_lengths, None)
    
    fine_norm_features = self.project_point(fine_rays, norm_xy_plane, norm_xz_plane, norm_yz_plane)
    fine_features = self.project_point(fine_rays, xy_plane, xz_plane, yz_plane) # (B, H * W, num_points_per_ray, C)

    fine_lengths = fine_rays.lengths
    fine_density, fine_seg,  fine_color = self.decoder(fine_norm_features, fine_features)

    with torch.no_grad():
        lengths = torch.cat((coarse_lengths, fine_lengths), dim=-1) # (B, H * W, num_points_per_ray * 2)
        lengths, indices = torch.sort(lengths, dim=-1)

    density = torch.cat((coarse_density, fine_density), dim=-2)
    color = torch.cat((coarse_color, fine_color), dim=-2)
    seg = torch.cat((coarse_seg, fine_seg), dim=-2)

    density = torch.gather(density, -2, indices[..., None].expand_as(density))
    color = torch.gather(color, -2, indices[..., None].expand_as(color))
    seg = torch.gather(seg, -2, indices[..., None].expand_as(seg))

    img, depth, weights = march_ray(density, color, lengths)
    seg = (seg * weights[..., None]).sum(dim=-2)
    img = img.reshape(img.size(0), self.img_resolution, self.img_resolution, -1).permute(0, 3, 1, 2).contiguous() # (B, 3, H, W)
    depth = depth.reshape(depth.size(0), self.img_resolution, self.img_resolution, -1).permute(0, 3, 1, 2).contiguous() # (B, 1, H, W)
    seg = seg.reshape(seg.size(0), self.img_resolution, self.img_resolution, -1).permute(0, 3, 1, 2).contiguous() # (B, C, H, W)

    rays = RayBundle(coarse_rays.origins, coarse_rays.directions, lengths, None)

    return {
        "img": img, 
        "depth": depth, 
        "seg": seg, 
        "delta_depth": delta_depth, 
        "rays": rays, 
        "weights": weights, 
    }

#----------------------------------------------------------------------------

dir_path = os.path.dirname(__file__)

class ModelZoo(object):
    def __init__(self, 
        device: torch.device, 
    ) -> None:
        super().__init__()
        self.device = device
        # Loading the Network
        with open(os.path.join(dir_path, "pretrained", "EG3D.pkl"), 'rb') as f:
            self._G = pickle.load(f)['G_ema'].to(device).eval().requires_grad_(False)
            self._G.vr.min_depth = 0.93
            self._G.vr.max_depth = 1.25
            self._G.vr.forward = MethodType(vr_forward, self._G.vr)
            self._G.vr.project_point = project_point
        
        # Load the LPIPS Loss used for Editing
        self.lpips_loss = lpips.LPIPS(net='alex').to(device)
        # Predefine the Frontal Camera
        self.frontal_cam = torch.from_numpy(np.array([[ np.pi / 2,  np.pi / 2,  0.,  1.,  0., 0.]], dtype=np.float32)).to(device)
    def get_EG3D(self, load_type: int, load_state_dict: bool = True) -> torch.nn.Module:
        """
        Args:
            load_type: A list of options including:
                        0: Original EG3D
                        1: NeRFFaceEditing
            load_state_dict: Whether to load the pretrained state dict for NeRFFaceEditing.
        Returns:
            Retrieved Corresponding Model
        """
        if load_type == 0:
            return copy.deepcopy(self._G).eval().requires_grad_(False).to(self.device)
        elif load_type == 1:
            # Define the Geometry Decoder & Appearance Decoder in one Module.
            class Decoder(torch.nn.Module):
                def __init__(self, activation='relu'):
                    super().__init__()
                    
                    self.fc0 = self.create_block(32, 64, activation=activation)
                    self.fcm = self.create_block(32, 64, activation=activation)
                    self.fcd = self.create_block(64, 1 + 15, activation='linear')
                    self.fcc = self.create_block(64, 32, activation='linear')
                def create_block(self, in_features: int, out_features: int, activation: str):
                    return FullyConnectedLayer(in_features, out_features, activation=activation)
                def forward(self, normalized_feature: torch.Tensor, denormalized_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    feat1 = self.fc0(normalized_feature)
                    out1 = self.fcd(feat1)
                    
                    feat2 = self.fcm(denormalized_feature)
                    out2 = self.fcc(feat2)
                    return out1[..., :1], out1[..., 1:1+15], out2
            
            G = copy.deepcopy(self._G).eval().requires_grad_(False).to(self.device)
            G.vr.decoder = Decoder().to(self.device).eval()
            G.vr.forward = MethodType(vr_forward_with_adain_seg, G.vr)
            
            # Load the state dict if required.
            if load_state_dict:
                G.load_state_dict(torch.load(os.path.join(dir_path, "pretrained", "NeRFFaceEditing.pth")))
            
            return G
        else:
            raise NotImplementedError(f"Unrecognized Model Type `{load_type}`.")