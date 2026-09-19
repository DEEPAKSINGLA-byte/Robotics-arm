"""Suggest suction contacts from the selected object's predicted 3D surface.

Geometry checks use only the observed/predicted surface. They do not certify
hidden-space clearance, suction seals, robot reachability or physical success.
"""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from bin_grasp.geometry import validate_transform


@dataclass(frozen=True)
class GraspConfig:
    cup_radius_m: float = .010
    approach_distance_m: float = .080
    tool_clearance_m: float = .005
    max_plane_rms_m: float = .003
    contact_tolerance_m: float = .006
    min_footprint_coverage: float = .90
    min_patch_points: int = 24
    max_candidates: int = 48
    max_proposals: int = 8

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not 0 < self.min_footprint_coverage <= 1:
            raise ValueError('Footprint coverage must be in (0, 1]')
        if self.contact_tolerance_m >= self.approach_distance_m:
            raise ValueError('Approach distance must exceed contact tolerance')
        for key in ('min_patch_points', 'max_candidates', 'max_proposals'):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f'{key} must be an integer')


def project(points, k, shape):
    """Convert 3D points to pixels using MoGe's normalized camera matrix."""
    h, w = shape
    z = points[..., 2]
    safe = np.where(np.isfinite(z) & (z > 0), z, 1.)
    uv = np.stack(((k[0, 0]*points[..., 0]/safe+k[0, 2])*w-.5,
                   (k[1, 1]*points[..., 1]/safe+k[1, 2])*h-.5), -1)
    good = np.isfinite(points).all(-1) & (z > 0) & np.isfinite(uv).all(-1)
    return uv, good


def tool_rotation(outward):
    """Tool +Z points toward contact; columns give tool axes in camera frame."""
    z = -np.asarray(outward, float)
    z /= np.linalg.norm(z)
    ref = np.array([1., 0., 0.]) if abs(z[0]) < .9 else np.array([0., 1., 0.])
    x = ref-z*np.dot(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def observed_collision_count(scene_points, tree, center, normal, radius, distance, tolerance):
    """Count predicted points blocking the cup's straight approach."""
    near = tree.query_ball_point(center+normal*distance*.5, np.hypot(distance*.5, radius))
    delta = scene_points[near]-center
    along = delta@normal
    radial = np.linalg.norm(delta-along[:, None]*normal, axis=1)
    return int(((along > tolerance) & (along <= distance) & (radial <= radius)).sum())


def sample_surface(xyz, valid, target, k, queries, tolerance):
    """Check whether the proposed cup footprint lies on the selected surface."""
    uv, good = project(queries, k, target.shape)
    safe_uv = np.where(good[:, None], uv, 0)
    coords = np.rint(safe_uv).astype(int)
    h, w = target.shape
    good &= (coords[:, 0] >= 0) & (coords[:, 0] < w) & (coords[:, 1] >= 0) & (coords[:, 1] < h)
    coords[:, 0] = np.clip(coords[:, 0], 0, w-1)
    coords[:, 1] = np.clip(coords[:, 1], 0, h-1)
    u, v = coords.T
    good &= valid[v, u]
    match = good & target[v, u] & (np.linalg.norm(xyz[v, u]-queries, axis=1) <= tolerance)
    return match, good, coords


def propose(xyz, valid, target, intrinsics, config=None, camera_to_robot=None):
    """Try flat patches inside the mask and keep contacts that pass our checks."""
    c = config or GraspConfig()
    xyz, valid, target = np.asarray(xyz, float), np.asarray(valid, bool), np.asarray(target, bool)
    k = np.asarray(intrinsics, float)
    if xyz.shape != (*target.shape, 3) or valid.shape != target.shape or target.ndim != 2:
        raise ValueError('Expected aligned HxWx3 points, HxW target mask and HxW validity')
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('Invalid normalized camera intrinsics')
    transform = validate_transform(camera_to_robot) if camera_to_robot is not None else None
    valid = valid & np.isfinite(xyz).all(-1) & (xyz[..., 2] > 0)
    target_valid = valid & target
    proposals, rejected = [], Counter()
    base = dict(kind='geometric_suction_hypotheses', config=asdict(c),
                geometry_source='RGB-predicted camera-frame metres',
                tool_model='Circular suction contact plus cylindrical approach envelope; no full wrist/arm model',
                execution_authorized=False, physical_success_verified=False,
                material_and_seal_verified=False, hidden_geometry_verified=False,
                robot_reachability_verified=False, robot_calibration_supplied=transform is not None,
                coordinate_convention='Camera +X right, +Y down, +Z forward. Tool +Z points toward contact.',
                config_note='Default 20-mm-diameter cup is an explicit prototype assumption, not measured hardware.')
    if target_valid.sum() < c.min_patch_points:
        return {**base, 'status': 'no_proposal', 'proposals': [], 'rejected': {'insufficient_target_geometry': 1},
                'sampled_candidates': 0, 'serial_instructions': recovery_instructions()}
    target_points = xyz[target_valid]
    target_tree = cKDTree(target_points)
    scene_points = xyz[valid]
    scene_tree = cKDTree(scene_points)
    distance = cv2.distanceTransform(np.pad(target_valid.astype('uint8'), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    coords = np.argwhere(target_valid)
    order = np.argsort(-distance[target_valid], kind='stable')
    candidates = []
    for index in order:
        pixel = coords[index]
        if all(np.linalg.norm(pixel-p) >= 6 for p in candidates):
            candidates.append(pixel)
        if len(candidates) >= c.max_candidates:
            break
    angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
    for v, u in candidates:
        seed = xyz[v, u]
        neighbors = target_tree.query_ball_point(seed, c.cup_radius_m*1.6)
        if len(neighbors) < c.min_patch_points:
            rejected['insufficient_patch_points'] += 1
            continue
        patch = target_points[neighbors]
        mean = patch.mean(0)
        # The least-changing direction is perpendicular to the fitted plane.
        eigenvalues, axes = np.linalg.eigh((patch-mean).T@(patch-mean)/len(patch))
        if eigenvalues[1] < 1e-8:
            rejected['degenerate_patch'] += 1
            continue
        normal = axes[:, 0]
        if np.dot(normal, mean) > 0:
            normal = -normal
        rms = float(np.sqrt(max(eigenvalues[0], 0)))
        if rms > c.max_plane_rms_m:
            rejected['surface_not_flat_enough'] += 1
            continue
        if np.dot(normal, -mean/np.linalg.norm(mean)) < .25:
            rejected['surface_too_oblique'] += 1
            continue
        center = seed-normal*np.dot(seed-mean, normal)
        rotation = tool_rotation(normal)
        tangent = np.cos(angles)[:, None]*rotation[:, 0] + np.sin(angles)[:, None]*rotation[:, 1]
        footprint = np.concatenate([center[None], center+tangent*c.cup_radius_m,
                                    center+tangent*c.cup_radius_m*.5])
        support, _, _ = sample_surface(xyz, valid, target, k, footprint,
                                       max(c.contact_tolerance_m, .4*c.cup_radius_m))
        coverage = float(support.mean())
        if coverage < c.min_footprint_coverage:
            rejected['footprint_crosses_edge_or_missing_surface'] += 1
            continue
        envelope_radius = c.cup_radius_m+c.tool_clearance_m
        approach = center+normal*c.approach_distance_m
        collisions = observed_collision_count(scene_points, scene_tree, center, normal, envelope_radius,
                                               c.approach_distance_m, c.contact_tolerance_m)
        if collisions:
            rejected['observed_surface_blocks_approach'] += 1
            continue
        # Check centreline visibility: behind an observed surface is occluded,
        # whereas being in front of its depth is observed ray free-space evidence.
        path = center + np.linspace(c.contact_tolerance_m*1.5, c.approach_distance_m, 12)[:, None]*normal
        uv, finite = project(path, k, target.shape)
        pixels = np.rint(np.where(finite[:, None], uv, 0)).astype(int)
        h, w = target.shape
        in_view = finite & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, w-1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, h-1)
        pu, pv = pixels.T
        known = in_view & valid[pv, pu]
        if not known.all():
            rejected['approach_has_unknown_depth_or_out_of_view'] += 1
            continue
        if (path[:, 2] > xyz[pv, pu, 2]+c.contact_tolerance_m).any():
            rejected['approach_hidden_behind_observed_surface'] += 1
            continue
        if any(np.linalg.norm(center-np.array(p['contact_camera_m'])) < c.cup_radius_m*.75 for p in proposals):
            rejected['near_duplicate_contact'] += 1
            continue
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, center
        pregrasp = pose.copy()
        pregrasp[:3, 3] = approach
        score = float(coverage-.25*rms/c.max_plane_rms_m)
        proposal = dict(contact_pixel_xy=[int(u), int(v)], contact_camera_m=center.tolist(),
                        outward_normal_camera=normal.tolist(), pregrasp_camera_m=approach.tolist(),
                        T_camera_tool_contact=pose.tolist(), T_camera_tool_pregrasp=pregrasp.tolist(),
                        plane_rms_m=rms, footprint_coverage=coverage,
                        geometric_rank_score=score, score_is_success_probability=False,
                        observed_approach_collision_points=0,
                        centreline_visibility='in front of predicted observed surface',
                        validity='Geometric proposal requiring material, calibration, reachability and full collision verification')
        if transform is not None:
            proposal['T_robot_tool_contact'] = (transform@pose).tolist()
            proposal['T_robot_tool_pregrasp'] = (transform@pregrasp).tolist()
        proposals.append(proposal)
    proposals.sort(key=lambda x: x['geometric_rank_score'], reverse=True)
    proposals = proposals[:c.max_proposals]
    for i, p in enumerate(proposals):
        p['rank'] = i+1
    return {**base, 'status': 'proposals_require_verification' if proposals else 'no_proposal',
            'sampled_candidates': len(candidates), 'rejected': dict(rejected), 'proposals': proposals,
            'serial_instructions': instructions(proposals[0]) if proposals else recovery_instructions()}


def recovery_instructions():
    return [dict(step=1, action='hold', reason='No supported grasp proposal; do not infer a contact from the object centroid'),
            dict(step=2, action='acquire_another_RGB_view', reason='Resolve occlusion, missing surface or insufficient contact area'),
            dict(step=3, action='ground_and_plan_again', reason='Re-identify the requested object in the updated scene')]


def instructions(proposal):
    return [dict(step=1, action='verify_target_and_scene', required=True,
                 checks=['requested object identity', 'current scene has not moved']),
            dict(step=2, action='verify_execution_preconditions', required=True,
                 checks=['actual cup dimensions and material seal', 'camera-to-robot calibration',
                         'robot inverse kinematics and full arm/wrist collision check',
                         'uncertainty and unobserved space assessed']),
            dict(step=3, action='move_to_pregrasp', conditional_on_steps=[1, 2],
                 camera_pose=proposal['T_camera_tool_pregrasp']),
            dict(step=4, action='approach_contact_with_force_or_contact_monitoring', conditional_on_steps=[1, 2, 3],
                 camera_pose=proposal['T_camera_tool_contact']),
            dict(step=5, action='activate_suction_and_verify_seal', on_failure='retract_then_observe_and_replan'),
            dict(step=6, action='retract_along_checked_approach', requires='verified seal and updated collision check'),
            dict(step=7, action='verify_lift_then_plan_destination', note='Destination and subsequent trajectory are not supplied')]


def save_grasps(run, config=None, camera_to_robot=None):
    run = Path(run)
    result = json.loads((run/'result.json').read_text())
    mask = np.asarray(Image.open(run/'selection/selected_mask.png')) > 0
    with np.load(run/'depth/prediction.npz') as a:
        report = propose(a['points_m'], a['mask'], mask, a['intrinsics'], config, camera_to_robot)
        intrinsics = a['intrinsics']
    report.update(sentence=result['sentence'], selected_candidate=result['selected_candidate'])
    (run/'grasp_proposals.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    image = np.asarray(Image.open(run/'selected_object.png').convert('RGB')).copy()
    for p in report['proposals']:
        pixel = tuple(p['contact_pixel_xy'])
        cv2.circle(image, pixel, 4, (255, 180, 10), -1)
        cv2.putText(image, str(p['rank']), (pixel[0]+5, pixel[1]-5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 180, 10), 1)
    if report['proposals']:
        best = report['proposals'][0]
        points = np.array([best['contact_camera_m'], best['pregrasp_camera_m']])
        uv, _ = project(points, intrinsics, mask.shape)
        cv2.arrowedLine(image, tuple(np.rint(uv[1]).astype(int)), tuple(np.rint(uv[0]).astype(int)),
                       (255, 180, 10), 2, tipLength=.2)
    Image.fromarray(image).save(run/'grasp_preview.png')
    result['grasp_proposals_file'] = 'grasp_proposals.json'
    result['grasp_proposal_count'] = len(report['proposals'])
    result['grasp_status'] = report['status']
    result['best_grasp_hypothesis_camera'] = report['proposals'][0]['T_camera_tool_contact'] if report['proposals'] else None
    result['robot_execution_ready'] = False
    (run/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Completed run_task3 output directory')
    parser.add_argument('--cup-radius-m', type=float, default=.010)
    parser.add_argument('--camera-to-robot', type=Path, help='JSON 4x4 rigid T_robot_camera in metres')
    args = parser.parse_args()
    t = json.loads(args.camera_to_robot.read_text()) if args.camera_to_robot else None
    report = save_grasps(args.run, GraspConfig(cup_radius_m=args.cup_radius_m), t)
    print(json.dumps({k: report[k] for k in ('status', 'sampled_candidates', 'rejected')}, indent=2))
    print(f'Grasp hypotheses: {len(report["proposals"])}. Robot execution ready: false.')


if __name__ == '__main__':
    main()
