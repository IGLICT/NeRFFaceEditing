#----------------------------------------------------------------------------
# This source code is largely borrowed and modified from pytorch3d.

from typing import NamedTuple, Dict, Tuple
import torch
import torch.nn.functional as F
import numpy as np

import math

class RayBundle(NamedTuple):
    """
    RayBundle parametrizes points along projection rays by storing ray `origins`,
    `directions` vectors and `lengths` at which the ray-points are sampled.
    Furthermore, the xy-locations (`xys`) of the ray pixels are stored as well.
    Note that `directions` don't have to be normalized; they define unit vectors
    in the respective 1D coordinate systems; see documentation for
    :func:`ray_bundle_to_ray_points` for the conversion formula.
    """

    origins: torch.Tensor
    directions: torch.Tensor
    lengths: torch.Tensor
    xys: torch.Tensor


def ray_bundle_to_ray_points(ray_bundle: RayBundle) -> torch.Tensor:
    """
    Converts rays parametrized with a `ray_bundle` (an instance of the `RayBundle`
    named tuple) to 3D points by extending each ray according to the corresponding
    length.

    E.g. for 2 dimensional tensors `ray_bundle.origins`, `ray_bundle.directions`
        and `ray_bundle.lengths`, the ray point at position `[i, j]` is:
        ```
            ray_bundle.points[i, j, :] = (
                ray_bundle.origins[i, :]
                + ray_bundle.directions[i, :] * ray_bundle.lengths[i, j]
            )
        ```
    Note that both the directions and magnitudes of the vectors in
    `ray_bundle.directions` matter.

    Args:
        ray_bundle: A `RayBundle` object with fields:
            origins: A tensor of shape `(..., 3)` or `(..., num_points_per_ray, 3)`
            directions: A tensor of shape `(..., 3)` or `(..., num_points_per_ray, 3)`
            lengths: A tensor of shape `(..., num_points_per_ray)`

    Returns:
        rays_points: A tensor of shape `(..., num_points_per_ray, 3)`
            containing the points sampled along each ray.
    """
    return ray_bundle_variables_to_ray_points(
        ray_bundle.origins, ray_bundle.directions, ray_bundle.lengths
    )


def ray_bundle_variables_to_ray_points(
    rays_origins: torch.Tensor,
    rays_directions: torch.Tensor,
    rays_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Converts rays parametrized with origins and directions
    to 3D points by extending each ray according to the corresponding
    ray length:

    E.g. for 2 dimensional input tensors `rays_origins`, `rays_directions`
    and `rays_lengths`, the ray point at position `[i, j]` is:
        ```
            rays_points[i, j, :] = (
                rays_origins[i, :]
                + rays_directions[i, :] * rays_lengths[i, j]
            )
        ```
    Note that both the directions and magnitudes of the vectors in
    `rays_directions` matter.

    Args:
        rays_origins: A tensor of shape `(..., 3)` or `(..., num_points_per_ray, 3)`
        rays_directions: A tensor of shape `(..., 3)` or `(..., num_points_per_ray, 3)`
        rays_lengths: A tensor of shape `(..., num_points_per_ray)`

    Returns:
        rays_points: A tensor of shape `(..., num_points_per_ray, 3)`
            containing the points sampled along each ray.
    """
    if len(rays_origins.shape) == len(rays_lengths.shape):
        rays_origins = rays_origins[..., None, :]
    
    if len(rays_directions.shape) == len(rays_lengths.shape):
        rays_directions = rays_directions[..., None, :]

    rays_points = (
        rays_origins
        + rays_lengths[..., :, None] * rays_directions
    )
    return rays_points

def _shifted_cumprod(x, shift=1):
    """
    Computes `torch.cumprod(x, dim=-1)` and prepends `shift` number of
    ones and removes `shift` trailing elements to/from the last dimension
    of the result.
    """
    x_cumprod = torch.cumprod(x, dim=-1)
    x_cumprod_shift = torch.cat(
        [torch.ones_like(x_cumprod[..., :shift]), x_cumprod[..., :-shift]], dim=-1
    )
    return x_cumprod_shift

#----------------------------------------------------------------------------

# This source code is largely borrowed from pi-GAN and pytorch3d official implementation of NeRF https://github.com/facebookresearch/pytorch3d/blob/main/projects/nerf/nerf/utils.py and https://github.com/facebookresearch/pytorch3d@main/-/blob/projects/nerf/nerf/raymarcher.py.
# Differentiable volumetric implementation used by pi-GAN generator.

def march_ray(
    density: torch.Tensor, 
    color: torch.Tensor, 
    lengths: torch.Tensor, 
    density_noise_std: float = 0.0, 
    eps: float = 1e-10, 
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    March the ray to accumulate approximate accumulated `density` and `color`.
    Args:
        density: A tensor of shape `(..., height * width, num_points_per_ray, 1)`
            denoting the raw density of each ray point.
        color: A tensor of shape `(..., height * width, num_points_per_ray, C)`
            denoting the color of each ray point.
        lengths: A tensor of shape `(..., height * width, num_points_per_ray)`
            containing the lengths at which the rays are sampled.
        density_noise_std: A floating point value representing the
            variance of the random normal noise added to the output of
            the opacity function. This can prevent floating artifacts.
        eps: A lower bound added to `density` before computing the absorption
            function (cumprod of `1-density` along each ray). This prevents the
            cumprod to yield exact 0 which would inhibit any gradient-based learning.
    Returns:
        color: A tensor of shape `(..., C)` denoting the expected color
            along the ray.
        depth: A tensor of shape `(..., 1)` denoting the expected depth
            along the ray, which could be used to generate Depth Map.
            If `lengths` is `None`, `depth` is `None` too.
        weights: A tensor of shape `(..., num_points_per_ray)` containing
            the ray-specific emission-absorption distribution.
            Each ray distribution `(..., :)` is a valid probability distribution, 
            i.e. it contains non-negative values that integrate to 1, such that 
            `(weights.sum(dim=-1)==1).all()` yields `True`.
    """
    # Calculate `alpha` based on `length`
    deltas = torch.cat(
        (
            lengths[..., 1:] - lengths[..., :-1], 
            1e10 * torch.ones_like(lengths[..., :1])
        ), 
        dim = -1, 
    )[..., None] # (..., height * width, num_points_per_ray, 1)
    if density_noise_std > 0.0:
        density = density + torch.randn_like(density) * density_noise_std
    
    alpha = 1 - (-deltas * F.relu(density)).exp() # (..., height * width, num_points_per_ray, 1)
    alpha = alpha[..., 0] # (..., height * width, num_points_per_ray)

    # Compute T_i in the paper
    absorption = _shifted_cumprod(
        (1.0 + eps) - alpha, shift=1
    ) # (..., height * width, num_points_per_ray)
    weights = alpha * absorption # (..., height * width, num_points_per_ray)

    color = (weights[..., None] * color).sum(dim=-2)
    # Compute expected depth
    depth = (weights[..., None] * lengths[..., None]).sum(dim=-2)
    return color, depth, weights

def get_initial_rays(
    batch_size: int, 
    image_size: Tuple[int, int], 
    fov: float, 
    n_pts_per_ray: int, 
    min_depth: float, 
    max_depth: float, 
    device: torch.device
) -> RayBundle:
    r"""
    Args:
        batch_size: The size of batch.
        image_size: The size of the rendered image (`[height, width]`).
        fov: The field of view in angle. It specifies the range of camera view
                                 Camera
                                   /|\
                                  / | \
                                 /  |  \
                                /   |   \
                               / --FOV-- \
                              /     |     \
                             / <--- 2 ---> \
                            _________________ Screen
        n_pts_per_ray: The number of points sampled along each ray for the
            rendering pass.
        min_depth: The minimum depth of a sampled ray-point for the rendering.
        max_depth: The maximum depth of a sampled ray-point for the rendering.
        device: The device on which operations are performed.
    Returns:
        ray_bundle: A RayBundle object containing the following variables:
            origins: A tensor of shape `(batch_size, 1, 3)` denoting the
                origin of the sampling rays in camera coords.
                Duplications here spare room for the diverse camera positions.
            directions: A tensor of shape `(batch_size, height * width, 3)`
                containing the direction vectors of sampling rays in camera coords.
            lengths: A tensor of shape `(batch_size, height * width, num_points_per_ray)`
                containing the lengths at which the rays are sampled.
                Duplications here spare room for the noise added to each ray and each
                element of batch.
    """
    height, width = image_size
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(
        torch.linspace(-1, 1, width, device=device),
        torch.linspace(1, -1, height, device=device)
    )
    x = x.T.flatten() # (height * width, )
    y = y.T.flatten() # (height * width, )
    # `z` reflects the position of camera.
    # As `fov` becomes larger, the camera gets closer to the screen, 
    # namely its `z` coordinate gets smaller.
    z = -torch.ones_like(x) / np.tan((2 * math.pi * fov / 360) / 2)

    origins = torch.zeros((3,), device=device) # (3, )
    directions = F.normalize(torch.stack([x, y, z], -1), dim=-1) # (height * width, 3)

    lengths_ratio = torch.linspace(0., 1., n_pts_per_ray, device=device).view(1, -1) # (num_points_per_ray, )
    lengths = min_depth + (max_depth - min_depth) * lengths_ratio # (batch_size, num_points_per_ray)

    return RayBundle(
        origins.view(1, 1, 3).expand(batch_size, -1, -1), 
        directions.view(1, -1, 3).expand(batch_size, -1, -1), 
        lengths[:, None, :].expand(batch_size, height * width, -1), 
        None # No need to use xys
    )

def apply_deformation(
    rays: RayBundle, 
    R: torch.Tensor, 
    p: torch.Tensor, 
) -> RayBundle:
    """
    Args:
        rays: A RayBundle object containing the following variables:
            origins: A tensor of shape `(batch_size, 1, 3)` denoting the
                origin of the sampling rays in camera coords.
                Duplications here spare room for the diverse camera positions.
            directions: A tensor of shape `(batch_size, height * width, 3)`
                containing the direction vectors of sampling rays in camera coords.
            lengths: A tensor of shape `(batch_size, height * width, num_points_per_ray)`
                containing the lengths at which the rays are sampled.
                Duplications here spare room for the noise added to each ray and each
                element of batch.
        R: The tensor of shape `(batch, height * width, num_points_per_ray, 3, 3)`
            denoting the rotation matrix.
        p: The tensor of shape `(batch, height * width, num_points_per_ray, 3, 1)`
            denoting the translation vector.
    Returns:
        rays: A RayBundle object containing the following variables:
            origins: A tensor of shape `(batch, height * width, num_points_per_ray, 3)` denoting the
                origin of the sampling rays in camera coords.
                Duplications here spare room for the diverse camera positions.
            directions: A tensor of shape `(batch, height * width, num_points_per_ray, 3)`
                containing the direction vectors of sampling rays in camera coords.
                Duplications here spare room for the diverse camera positions.
            lengths: A tensor of shape `(batch, height * width, num_points_per_ray)`
                containing the lengths at which the rays are sampled.
                Duplications here spare room for the noise added to each ray and each
                element of batch.
    """
    # Apply Translation
    # origins: (batch, 1, 3)
    p = p.squeeze(-1) # (batch, height * width, num_points_per_ray, 3)
    origins = rays.origins.view(rays.origins.size(0), 1, 1, 3, 1) # (batch, 1, 1, 3, 1)
    origins = torch.matmul(R, origins).squeeze(-1) # (batch, height * width, num_points_per_ray, 3)
    origins = origins + p

    # Apply Rotation
    # directions: (batch, height * width, 3)
    directions = rays.directions.view(*rays.directions.shape[:2], 1, 3, 1).expand(-1, -1, R.shape[-3], -1, -1) # (batch, height * width, num_points_per_ray, 3, 1)
    directions = torch.matmul(R, directions).squeeze(-1) # (batch, height * width, num_points_per_ray, 3)

    return RayBundle(origins, directions, rays.lengths, rays.xys)

def perturb_length(
    lengths: torch.Tensor
) -> torch.Tensor:
    """
    Add noise to the lengths of each ray in-place.
    Args:
        lengths: A tensor of shape `(..., num_points_per_ray)`
            containing the lengths at which the rays are sampled.
            Duplications here spare room for the noise added to each ray and each
            element of batch.
    Returns:
        The same with argument `lengths`
    """
    # Since `lengths` is generated by `linspace`, to get the common interval, 
    # we only need to get the interval between the second and first one.
    with torch.no_grad():
        dist = (lengths.reshape(-1)[1] - lengths.reshape(-1)[0]).item()
    delta_length = torch.cat(
        (
            torch.rand_like(lengths[..., :1]) * .5, 
            torch.rand_like(lengths[..., 1:-1]) - .5, 
            torch.rand_like(lengths[..., -1:]) * (-.5)
        ), dim=-1
    )
    return lengths + delta_length * dist

def get_corrected_delta_depth(
    yaw: torch.Tensor, 
    pitch: torch.Tensor, 
    roll: torch.Tensor, 
    radius: torch.Tensor, 
    delta_x: torch.Tensor, 
    delta_y: torch.Tensor, 
) -> torch.Tensor:
    camera_pos = get_camera_positions(yaw, pitch, radius) # \vec{OC}
    camera_dir = F.normalize(camera_pos - 0, dim=-1) # direction of \vec{OC}
    delta_pos = torch.cat((delta_x, delta_y, torch.zeros_like(delta_x)), dim=-1)
    delta_depth = torch.sum(camera_dir * delta_pos, dim=-1, keepdim=True)
    return delta_depth

def transform_pos_to_cond(
    pos: torch.Tensor, 
) -> torch.Tensor:
    yaw, pitch, roll, radius, delta_x, delta_y = torch.split(pos, 1, -1)
    # Sample Camera Positions
    # cameras_origin: (batch, 3)
    cameras_origin = get_camera_positions(yaw, pitch, radius)
    cameras_origin_ref = get_camera_positions(yaw, torch.ones_like(pitch) * np.pi / 2, radius)
    cameras_forward = F.normalize(-cameras_origin, dim=-1)
    cameras_forward_ref = F.normalize(-cameras_origin_ref, dim=-1)
    # Add Delta
    cameras_origin[:, 0:1] = cameras_origin[:, 0:1] + delta_x
    cameras_origin[:, 1:2] = cameras_origin[:, 1:2] + delta_y

    # Transform `origins` and `directions`
    # translation: (batch, 3)
    translation = cameras_origin
    # rotation: (batch, 3, 3)
    rotation = create_camera2world_matrix(cameras_forward, cameras_forward_ref, roll)
    # matrix: (batch, 4, 4)
    matrix = torch.eye(4, device=translation.device, dtype=translation.dtype).unsqueeze(0).repeat(translation.size(0), 1, 1)
    matrix[:, :3, :3] = rotation
    matrix[:, :3, 3] = translation
    return matrix.view(-1, 16)
    

def transform_rays(
    rays: RayBundle, 
    yaw: torch.Tensor, 
    pitch: torch.Tensor, 
    roll: torch.Tensor, 
    radius: torch.Tensor, 
    delta_x: torch.Tensor, 
    delta_y: torch.Tensor, 
    add_noise: bool = True
):
    r"""
    Sample a camera position and Transform rays in Camera Space into World Space.
    Args:
        rays: A RayBundle object containing the following variables:
            origins: A tensor of shape `(batch_size, ..., [num_points_per_ray], 3)`
                origin of the sampling rays in camera coords.
                Duplications here spare room for the diverse camera positions.
            directions: A tensor of shape `(batch_size, ..., [num_points_per_ray], 3)`
                containing the direction vectors of sampling rays in camera coords.
            lengths: A tensor of shape `(batch_size, ..., num_points_per_ray)`
                containing the lengths at which the rays are sampled.
                Duplications here spare room for the noise added to each ray and each
                element of batch.
        yaw: The tensor of shape (batch_size, 1) denoting the yaw angle in radian.
        pitch: The tensor of shape (batch_size, 1) denoting the pitch angle in radian.
        roll: The tensor of shape (batch_size, 1) denoting the roll angle in radian.

                                Y
                                |    Roll
                                |    (-)
                                |    /|
                                |   / |
                                |  /  |
                                | /P  |
                                |/ i  |
                                ---t--|-------------Z
                               /\  c  |
                              /  \ h  |
                             /    \   |
                            / Yaw  \  |
                           /        \ |
                          /          \|
                         X
        radius: The tensor of shape (batch_size, 1) denoting the radius.
        delta_x: The tensor of shape (batch_size, 1).
        delta_y: The tensor of shape (batch_size, 1).
    Returns:
        rays: A RayBundle object containing the following variables:
            origins: A tensor of shape `(batch_size, ..., num_points_per_ray, 3)` denoting the
                origin of the sampling rays in camera coords.
                Duplications here spare room for the diverse camera positions.
            directions: A tensor of shape `(batch_size, ..., num_points_per_ray, 3)`
                containing the direction vectors of sampling rays in camera coords.
                Duplications here spare room for the diverse camera positions.
            lengths: A tensor of shape `(batch_size, ..., num_points_per_ray)`
                containing the lengths at which the rays are sampled.
                Duplications here spare room for the noise added to each ray and each
                element of batch.
        pitch: A tensor of shape (batch_size, 1) denoting the pitch of cameras.
        yaw: A tensor of shape (batch_size, 1) denoting the yaw of cameras.
    """
    assert rays.origins.shape[0] == rays.directions.shape[0] and rays.directions.shape[0] == rays.lengths.shape[0] and rays.lengths.shape[0] == pitch.shape[0] and rays.lengths.shape[0] == yaw.shape[0]

    # Expand `rays`
    if len(rays.origins.shape) == len(rays.lengths.shape) + 1:
        origins = rays.origins # (batch, ..., num_points_per_ray, 3)
    elif len(rays.origins.shape) == len(rays.lengths.shape):
        origins = rays.origins[..., None, :] # (batch, ..., 1, 3)

    if len(rays.directions.shape) == len(rays.lengths.shape) + 1:
        directions = rays.directions # (batch, ..., num_points_per_ray, 3)
    elif len(rays.directions.shape) == len(rays.lengths.shape):
        directions = rays.directions[..., None, :] # (batch, ..., 1, 3)
    
    lengths = rays.lengths # (batch, minibatch, height * width, num_points_per_ray)

    # Add noise to `lengths`
    # Mutate Here
    if add_noise == True:
        lengths = perturb_length(lengths)

    # Sample Camera Positions
    # cameras_origin: (batch, 3)
    cameras_origin = get_camera_positions(yaw, pitch, radius)
    cameras_origin_ref = get_camera_positions(yaw, torch.ones_like(pitch) * np.pi / 2, radius)
    cameras_forward = F.normalize(-cameras_origin, dim=-1)
    cameras_forward_ref = F.normalize(-cameras_origin_ref, dim=-1)
    # Add Delta
    cameras_origin[:, 0:1] = cameras_origin[:, 0:1] + delta_x
    cameras_origin[:, 1:2] = cameras_origin[:, 1:2] + delta_y

    # Transform `origins` and `directions`
    # translation: (batch, 3)
    translation = cameras_origin
    # rotation: (batch, 3, 3)
    rotation = create_camera2world_matrix(cameras_forward, cameras_forward_ref, roll)

    # R(\vec{o}+\vec{d}l)+p = R\vec{o} + R\vec{d}l + p
    origins = torch.matmul(
        rotation[:, None, :, :], # (batch, 1, 3, 3)
        origins.reshape(origins.size(0), -1, 3, 1) # (batch, ?, 3, 1), 
    ).reshape(origins.shape)
    origins = origins + translation.view(translation.size(0), *[1 for _ in range(len(origins.shape) - 2)], translation.size(1))
    
    directions = torch.matmul(
        rotation[:, None, :, :], # (batch, 1, 3, 3)
        directions.reshape(directions.size(0), -1, 3, 1) # (batch, ?, 3, 1)
    ).reshape(directions.shape)

    return RayBundle(origins, directions, lengths, rays.xys)

def get_camera_positions(
    yaw: torch.Tensor, 
    pitch: torch.Tensor, 
    radius: torch.Tensor, 
) -> torch.Tensor:
    """
    Get Camera Positions by transforming given vertical position `v`
    and horizontal position `h`.
    Args:
        yaw: Tensor of the shape (..., 1) denoting the
            Horizontal positions.
        pitch: Tensor of the shape (..., 1) denoting the 
            Vertical positions.
        radius: Tensor of the shape (..., 1) denoting the
            radius.
    Returns:
        points: tensor with shape (..., 3) denoting the position of 
            the point on the sphere with radius 1 given `phi` and `theta`.
    """

    x = torch.sin(pitch) * torch.cos(yaw) * radius
    y = torch.cos(pitch) * radius
    z = torch.sin(pitch) * torch.sin(yaw) * radius

    points = torch.cat((x, y, z), dim=-1)
    return points

def sample_position(
    n: int, 
    yaw_mean: float, 
    pitch_mean: float, 
    roll_mean: float, 
    radius_mean: float, 
    delta_x_mean: float, 
    delta_y_mean: float, 
    yaw_std: float, 
    pitch_std: float, 
    roll_std: float, 
    radius_std: float, 
    delta_x_std: float, 
    delta_y_std: float, 
    device: torch.device, 
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample Camera Positions given `mean` and `std`.
    Args:
        n: Number of cameras
        yaw_mean: Mean of Yaw
        pitch_mean: Mean of Pitch
        roll_mean: Mean of Roll
        yaw_std: Standard deviation of Yaw
        pitch_std: Standard deviation of Pitch
        roll_std: Standard deviation of Yaw
        device : The device on which operations are performed
    Returns:
        yaw: tensor with shape of `yaw_mean`
        pitch: tensor with shape of `pitch_mean`
        roll: tensor with shape of `roll_mean`
    """
    pitch = torch.randn((n, 1), device=device) * pitch_std + pitch_mean
    yaw = torch.randn((n, 1), device=device) * yaw_std + yaw_mean
    roll = torch.randn((n, 1), device=device) * roll_std + roll_mean
    radius = torch.randn((n, 1), device=device) * radius_std + radius_mean
    delta_x = torch.randn((n, 1), device=device) * delta_x_std + delta_x_mean
    delta_y = torch.randn((n, 1), device=device) * delta_y_std + delta_y_mean

    return torch.cat((yaw, pitch, roll, radius, delta_x, delta_y), dim=-1)

def create_camera2world_matrix(
    direction: torch.Tensor, 
    direction_ref: torch.Tensor, 
    roll: torch.Tensor, 
) -> torch.Tensor:
    """
    Compute a transformation matrix mapping from Camera Space to World Space.
    Args:
        origin: Tensor with shape (..., 3) denoting the origin of the camera in World
            Space
        direction: Tensor with shape (..., 3) denoting the normalized direction vector
            of the camera
        roll: The tensor of shape (batch_size, 1) denoting the roll angle in radian.
    Returns:
        rotation: Tensor with shape (..., 3, 3) denoting the rotation matrix.
            To transform `vectors` (..., 3) from Camera Space to World Space, you only
            need to do torch.bmm(`rotation`, `vectors`)
    """
    forward = F.normalize(direction, dim=-1) # (..., 3)
    forward_ref = F.normalize(direction_ref, dim=-1) # (..., 3)
    up_ref = torch.tensor([0., 1., 0.], dtype=torch.float, device=forward_ref.device).view(*[1 for _ in range(len(forward_ref.shape) - 1)], -1).expand_as(forward_ref) # (..., 3)
    roll_mat = axis_angle_to_matrix(forward * roll) # (..., 3, 3)
    axis = F.normalize(torch.cross(up_ref, forward_ref, dim=-1), dim=-1)
    up = -F.normalize(torch.cross(axis, forward, dim=-1), dim=-1).unsqueeze(-1) # (..., 3, 1)
    up = torch.matmul(roll_mat, up).squeeze(-1) # (..., 3)
    left = F.normalize(torch.cross(up, forward, dim=-1), dim=-1)
    up = F.normalize(torch.cross(forward, left, dim=-1), dim=-1) # Yes, necessary. The result is changed from World Up direction to Camera Up direction.

    rotation = torch.stack((-left, up, -forward), axis=-1) # (..., 3, 3)

    return rotation

def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    N_samples: int,
    det: bool = False,
    eps: float = 1e-5,
):
    """
    Samples a probability density functions defined by bin edges `bins` and
    the non-negative per-bin probabilities `weights`.
    Note: This is a direct conversion of the TensorFlow function from the original
    release [1] to PyTorch.
    Args:
        bins: Tensor of shape `(..., n_bins+1)` denoting the edges of the sampling bins.
        weights: Tensor of shape `(..., n_bins)` containing non-negative numbers
            representing the probability of sampling the corresponding bin.
        N_samples: The number of samples to draw from each set of bins.
        det: If `False`, the sampling is random. `True` yields deterministic
            uniformly-spaced sampling from the inverse cumulative density function.
        eps: A constant preventing division by zero in case empty bins are present.
    Returns:
        samples: Tensor of shape `(..., N_samples)` containing `N_samples` samples
            drawn from each set probability distribution.
    Refs:
        [1] https://github.com/bmild/nerf/blob/55d8b00244d7b5178f4d003526ab6667683c9da9/run_nerf_helpers.py#L183  # noqa E501
    """

    # Get pdf
    weights = weights + eps  # prevent nans
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, N_samples, device=cdf.device, dtype=cdf.dtype)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples]).contiguous()
    else:
        u = torch.rand(
            list(cdf.shape[:-1]) + [N_samples], device=cdf.device, dtype=cdf.dtype
        )

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = (inds - 1).clamp(0)
    above = inds.clamp(max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], -1).view(
        *below.shape[:-1], below.shape[-1] * 2
    )

    cdf_g = torch.gather(cdf, -1, inds_g).view(*below.shape, 2)
    bins_g = torch.gather(bins, -1, inds_g).view(*below.shape, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def create_samples(
    N: int, 
    voxel_origin: Tuple[float, float, float], 
    cube_length: float, 
    device: torch.device
) -> torch.Tensor:
    """
    Create Sample Points by dividing Screen Space into Grids
    Args:
        N: The number of grid
        voxel_origin: The origin of voxel
        cube_length: The length of cube
        device: The device on which operations are conducted
    Returns:
        samples: tensor of shape (1, 1, N ** 3, 3) denoting the sample points
    """
    voxel_origin = torch.tensor(voxel_origin, device=device) - cube_length / 2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, device=device, dtype=torch.long)
    samples = torch.zeros(N ** 3, 3, device=device)

    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = (overall_index.float() / N ** 2) % N

    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[2]

    return samples[None, None, ...]

def compute_opaqueness_mask(weights: torch.Tensor, depth_threshold=0.5) -> torch.Tensor:
    """Computes a mask which will be 1.0 at the depth point.

    Args:
        weights: the density weights from NeRF.
        depth_threshold: the accumulation threshold which will be used as the depth
        termination point.

    Returns:
        A tensor containing a mask with the same size as weights that has one
        element long the sample dimension that is 1.0. This element is the point
        where the 'surface' is.
    """
    cumulative_contribution = torch.cumsum(weights, dim=-1)
    opaqueness = cumulative_contribution >= depth_threshold
    false_padding = torch.zeros_like(opaqueness[..., :1])
    padded_opaqueness = torch.cat(
        (
            false_padding, 
            opaqueness[..., :-1]
        ), 
        dim=-1, 
    )
    opaqueness_mask = torch.logical_xor(opaqueness, padded_opaqueness).float()
    return opaqueness_mask

def compute_depth_index(weights: torch.Tensor, depth_threshold=0.5) -> torch.Tensor:
    """Compute the sample index of the median depth accumulation."""
    opaqueness_mask = compute_opaqueness_mask(weights, depth_threshold)
    return torch.argmax(opaqueness_mask, dim=-1)

def rgb_to_grayscale(img: torch.Tensor, num_output_channels: int = 3) -> torch.Tensor:
    if num_output_channels not in (1, 3):
        raise ValueError('num_output_channels should be either 1 or 3')

    r, g, b = img.unbind(dim=-1)
    # This implementation closely follows the TF one:
    # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-1)

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img

# Copied from PyTorch 3D
# src: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

#----------------------------------------------------------------------------

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import Tuple
import numpy as np
import torch
from torch.nn import functional as F
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma

#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, x.size(-1))
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        x = x.reshape(*shape[:-1], x.size(-1))
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
        decoder_kwargs      = {},   # Arguments for DecoderNetwork
        vr_kwargs           = {},   # Arguments for VolumeRenderingNetwork
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws + 2 + 3
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        decoder_kwargs["out_features"] = img_channels + 1
        self.vr = VolumeRendering(decoder_kwargs=decoder_kwargs, img_resolution=img_resolution, **vr_kwargs)
        self.superres_0 = SynthesisBlock(32, 128, w_dim, 256, 3, False, conv_clamp=256, use_noise=False, use_fp16=True)
        self.superres_1 = SynthesisBlock(128, 64, w_dim, 512, 3, True, conv_clamp=256, use_noise=False, use_fp16=True)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        with torch.no_grad():
            cond = transform_pos_to_cond(c)
        ws = self.mapping(z, cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        planes = self.synthesis(ws[:, :-5], **synthesis_kwargs)
        xy_plane, yz_plane, xz_plane = torch.chunk(planes, 3, dim=1)
        features = self.vr(xy_plane, yz_plane, xz_plane, c)["img"]
        # features = F.interpolate(features, (128, 128), mode='bilinear')
        low_img = features[:, :3]
        
        sup_ws = ws[:, -5:]
        block_super_ws = [sup_ws.narrow(1, 0, 3), sup_ws.narrow(1, 2, 3)]
        x, high_img = features, low_img
        x, high_img = self.superres_0(x, high_img, block_super_ws[0], **synthesis_kwargs)
        x, high_img = self.superres_1(x, high_img, block_super_ws[1], **synthesis_kwargs)
        return low_img, high_img

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, **block_kwargs):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            with torch.no_grad():
                c = transform_pos_to_cond(c)
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Decoder(torch.nn.Module):
    def __init__(self, 
        in_features: int, 
        hidden_features: int, 
        out_features: int, 
        num_layers: int, 
        lr_multiplier: float, 
        activation: str, 
    ):
        super().__init__()
        self.num_layers = num_layers

        self.fc0 = self.create_block(in_features, hidden_features, activation=activation)

        for idx in range(1, num_layers - 1):
            layer = self.create_block(hidden_features, hidden_features, activation=activation)
            setattr(self, f'fc{idx}', layer)
        
        setattr(self, f'fc{num_layers - 1}', self.create_block(hidden_features, out_features, activation='linear'))

    def create_block(self, in_features: int, out_features: int, activation: str):
        if activation == 'relu':
            return torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features), 
                torch.nn.ReLU()
            )
        elif activation == 'linear':
            return torch.nn.Linear(in_features, out_features)
        # return FullyConnectedLayer(in_features, out_features, activation=activation)
    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = feature
        # Main layers
        for idx in range(self.num_layers - 1):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)
        x = getattr(self, f'fc{self.num_layers - 1}')(x)
        
        # Density, Color
        return x[..., :1], x[..., 1:]

#----------------------------------------------------------------------------

@persistence.persistent_class
class VolumeRendering(torch.nn.Module):
    def __init__(self, 
        fov: float, 
        n_pts_per_ray: int, 
        min_depth: float, 
        max_depth: float, 
        img_resolution: int, 
        decoder_kwargs = {}, 
    ):
        super().__init__()

        self.fov = fov
        self.n_pts_per_ray = n_pts_per_ray
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.img_resolution = img_resolution

        self.decoder = Decoder(**decoder_kwargs)
    
    @staticmethod
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
            
        mask = torch.logical_and(
            torch.logical_and(
                torch.logical_and( norm_x > -1., norm_x < 1. ), 
                torch.logical_and( norm_y > -1., norm_y < 1. ), 
            ), 
            torch.logical_and( norm_z > -1., norm_z < 1. )
        )[..., None]

        xy = torch.stack((norm_x, norm_y), dim=-1) # (B, H*W, num, 2)
        yz = torch.stack((norm_y, norm_z), dim=-1) # (B, H*W, num, 2)
        xz = torch.stack((norm_x, norm_z), dim=-1) # (B, H*W, num, 2)

        F_xy = F.grid_sample(xy_plane, xy, padding_mode="zeros").permute(0, 2, 3, 1) # (B, H*W, num, C)
        F_yz = F.grid_sample(yz_plane, yz, padding_mode="zeros").permute(0, 2, 3, 1) # (B, H*W, num, C)
        F_xz = F.grid_sample(xz_plane, xz, padding_mode="zeros").permute(0, 2, 3, 1) # (B, H*W, num, C)
        
        features = (F_xy + F_yz + F_xz) * mask

        return features

    def forward(self, 
        xy_plane: torch.Tensor, 
        yz_plane: torch.Tensor, 
        xz_plane: torch.Tensor, 
        camera: torch.Tensor, 
        verbose: bool = False, 
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
                self.n_pts_per_ray, 
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
            _, _, weights = march_ray(coarse_density, coarse_color, coarse_lengths)
            lengths_mid = 0.5 * (coarse_lengths[..., 1:] + coarse_lengths[..., :-1]) # (B, H * W, num_points_per_ray - 1)
            fine_lengths = sample_pdf(lengths_mid, weights[..., 1:-1], self.n_pts_per_ray).detach() # (B, H * W, num_points_per_ray)
            fine_rays = RayBundle(coarse_rays.origins, coarse_rays.directions, fine_lengths, None)

        fine_features = self.project_point(fine_rays, xy_plane, xz_plane, yz_plane) # (B, H * W, num_points_per_ray, C)

        fine_lengths = fine_rays.lengths
        fine_density, fine_color = self.decoder(fine_features)

        with torch.no_grad():
            lengths = torch.cat((coarse_lengths, fine_lengths), dim=-1) # (B, H * W, num_points_per_ray * 2)
            lengths, indices = torch.sort(lengths, dim=-1)

        density = torch.cat((coarse_density, fine_density), dim=-2)
        color = torch.cat((coarse_color, fine_color), dim=-2)
        features = torch.cat((coarse_features, fine_features), dim=-2) if verbose else None

        density = torch.gather(density, -2, indices[..., None].expand_as(density))
        color = torch.gather(color, -2, indices[..., None].expand_as(color))
        features = torch.gather(features, -2, indices[..., None].expand_as(features)) if verbose else None

        color, depth, weights = march_ray(density, color, lengths)
        color = color.reshape(color.size(0), self.img_resolution, self.img_resolution, -1).permute(0, 3, 1, 2).contiguous() # (B, 3, H, W)
        depth = depth.reshape(depth.size(0), self.img_resolution, self.img_resolution, -1).permute(0, 3, 1, 2).contiguous() # (B, 1, H, W)

        rays = RayBundle(coarse_rays.origins, coarse_rays.directions, lengths, None) if verbose else None

        return {
            "img": color, 
            "depth": depth, 
            "delta_depth": delta_depth, 
            "weights": weights, 
            "features": features, 
            "rays": rays, 
        }

#----------------------------------------------------------------------------
