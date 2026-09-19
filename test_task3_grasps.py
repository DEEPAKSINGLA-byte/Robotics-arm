"""Fast geometric checks; these do not measure physical grasp success."""
import unittest
import numpy as np
from scipy.spatial import cKDTree
from task3_grasp_proposals import (GraspConfig, propose, project, tool_rotation,
                                   validate_transform, observed_collision_count)


def flat_scene():
    height = width = 101
    v, u = np.indices((height, width))
    k = np.array([[2., 0., .5], [0., 2., .5], [0., 0., 1.]])
    points = np.stack((((u+.5)/width-.5)/2, ((v+.5)/height-.5)/2, np.ones_like(u)), -1)
    target = np.zeros((height, width), bool)
    target[20:81, 20:81] = True
    return points, np.ones_like(target), target, k


class GraspChecks(unittest.TestCase):
    def test_flat_surface_proposes_contact_and_pregrasp(self):
        p, valid, mask, k = flat_scene()
        r = propose(p, valid, mask, k, GraspConfig(cup_radius_m=.025, max_candidates=8))
        self.assertTrue(r['proposals'])
        best = r['proposals'][0]
        self.assertLess(best['plane_rms_m'], 1e-8)
        np.testing.assert_allclose(best['outward_normal_camera'], [0, 0, -1], atol=1e-8)
        self.assertAlmostEqual(best['contact_camera_m'][2]-best['pregrasp_camera_m'][2], .08)
        self.assertFalse(r['physical_success_verified'])
        self.assertFalse(r['execution_authorized'])
        self.assertEqual(r['serial_instructions'][2]['conditional_on_steps'], [1, 2])

    def test_pixel_projection_roundtrip(self):
        p, _, mask, k = flat_scene()
        uv, valid = project(p, k, mask.shape)
        v, u = np.indices(mask.shape)
        np.testing.assert_allclose(uv, np.stack([u, v], -1), atol=1e-10)
        self.assertTrue(valid.all())

    def test_approach_rejects_obstacle_but_ignores_contact_plane(self):
        points = np.array([[0., 0., 1.], [0., 0., .96], [.2, 0., .96], [0., 0., .8]])
        count = observed_collision_count(points, cKDTree(points), np.array([0., 0., 1.]),
                                         np.array([0., 0., -1.]), .015, .08, .006)
        self.assertEqual(count, 1)
        self.assertEqual(observed_collision_count(points[[0, 2, 3]], cKDTree(points[[0, 2, 3]]),
                         np.array([0., 0., 1.]), np.array([0., 0., -1.]), .015, .08, .006), 0)

    def test_missing_geometry_requires_observation(self):
        p, valid, mask, k = flat_scene()
        valid[:] = False
        r = propose(p, valid, mask, k)
        self.assertEqual(r['status'], 'no_proposal')
        self.assertEqual(r['serial_instructions'][0]['action'], 'hold')

    def test_tiny_object_cannot_support_large_cup(self):
        p, valid, mask, k = flat_scene()
        mask[:] = False
        mask[46:55, 46:55] = True
        r = propose(p, valid, mask, k, GraspConfig(cup_radius_m=.06, max_candidates=8))
        self.assertFalse(r['proposals'])
        self.assertGreater(r['rejected'].get('footprint_crosses_edge_or_missing_surface', 0), 0)

    def test_frame_convention_and_robot_composition(self):
        p, valid, mask, k = flat_scene()
        transform = np.eye(4)
        transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        transform[:3, 3] = [.1, .2, .3]
        r = propose(p, valid, mask, k, GraspConfig(cup_radius_m=.025, max_candidates=2), transform)
        best = r['proposals'][0]
        np.testing.assert_allclose(best['T_robot_tool_contact'], transform@np.array(best['T_camera_tool_contact']))
        rot = tool_rotation([.2, .3, -.9])
        np.testing.assert_allclose(rot.T@rot, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rot), 1.)

    def test_invalid_transform_or_configuration_rejected(self):
        bad = np.eye(4)
        bad[0, 0] = -1
        with self.assertRaises(ValueError):
            validate_transform(bad)
        with self.assertRaises(ValueError):
            GraspConfig(cup_radius_m=-.01)
        with self.assertRaises(ValueError):
            GraspConfig(max_candidates=1.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
