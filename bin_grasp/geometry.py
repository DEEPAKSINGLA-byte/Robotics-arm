"""Small, explicit pinhole and rigid-transform utilities. All 3D lengths are metres.

These functions do not estimate depth, hidden surfaces, or robot calibration.
They assume pinhole-compatible pixels; lens distortion must be handled upstream.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if (not isinstance(self.width, int) or not isinstance(self.height, int)
                or min(self.width, self.height) <= 0):
            raise ValueError('Image dimensions must be positive integers')
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all() or min(self.fx, self.fy) <= 0:
            raise ValueError('Finite camera values and positive focal lengths required')

    def to_dict(self):
        return asdict(self)

    def crop_resize(self, x0, y0, width, height, out_width, out_height):
        """Pixel-centre convention: u_new = sx * (u_old - x0 + .5) - .5."""
        values = [x0, y0, width, height, out_width, out_height]
        if not all(isinstance(v, int) for v in values) or min(width, height, out_width, out_height) <= 0:
            raise ValueError('Integer crop coordinates and positive sizes required')
        if x0 < 0 or y0 < 0 or x0+width > self.width or y0+height > self.height:
            raise ValueError('Crop outside image')
        sx, sy = out_width/width, out_height/height
        return Camera(out_width, out_height, self.fx*sx, self.fy*sy,
                      (self.cx-x0+.5)*sx-.5, (self.cy-y0+.5)*sy-.5)


def unproject_pixels(uv, depth_m, camera):
    """Camera axes: +X right, +Y down, +Z forward; depth is Z, not ray distance."""
    uv, z = np.asarray(uv, dtype=float), np.asarray(depth_m, dtype=float)
    if uv.shape[-1:] != (2,) or z.shape != uv.shape[:-1]:
        raise ValueError('Pixel/depth shape mismatch')
    valid = np.isfinite(uv).all(axis=-1) & np.isfinite(z) & (z > 0)
    safe_z = np.where(valid, z, 0.)
    safe_uv = np.where(valid[..., None], uv, [camera.cx, camera.cy])
    points = np.stack(((safe_uv[..., 0]-camera.cx)*safe_z/camera.fx,
                       (safe_uv[..., 1]-camera.cy)*safe_z/camera.fy, safe_z), axis=-1)
    return np.where(valid[..., None], points, np.nan)


def unproject_depth(depth_m, camera):
    depth_m = np.asarray(depth_m)
    if depth_m.shape != (camera.height, camera.width):
        raise ValueError('Depth must align with camera resolution')
    v, u = np.indices(depth_m.shape)
    return unproject_pixels(np.stack((u, v), axis=-1), depth_m, camera)


def project_points(points, camera):
    points = np.asarray(points, dtype=float)
    if points.shape[-1:] != (3,):
        raise ValueError('Expected XYZ points')
    valid = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    z = np.where(valid, points[..., 2], 1.)
    uv = np.stack((camera.fx*points[..., 0]/z+camera.cx,
                   camera.fy*points[..., 1]/z+camera.cy), axis=-1)
    return np.where(valid[..., None], uv, np.nan)


def validate_transform(transform):
    """T_destination_source transforms source-frame points into destination frame."""
    t = np.asarray(transform, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError('Finite 4x4 transform required')
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError('Invalid homogeneous final row')
    r = t[:3, :3]
    if not np.allclose(r.T@r, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(r), 1., atol=1e-6):
        raise ValueError('Rotation must be orthonormal with determinant +1; scale/reflection not allowed')
    return t


def transform_points(points, transform):
    points = np.asarray(points, dtype=float)
    if points.shape[-1:] != (3,):
        raise ValueError('Expected XYZ points')
    t = validate_transform(transform)
    return points@t[:3, :3].T+t[:3, 3]


def inverse_transform(transform):
    t = validate_transform(transform)
    inverse = np.eye(4)
    inverse[:3, :3] = t[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3]@t[:3, 3]
    return inverse


def masked_surface(points, mask):
    points, mask = np.asarray(points), np.asarray(mask)
    if points.ndim != 3 or points.shape[-1] != 3 or mask.shape != points.shape[:2] or mask.dtype != bool:
        raise ValueError('Expected aligned HxWx3 points and boolean HxW mask')
    return points[mask & np.isfinite(points).all(axis=-1)]
