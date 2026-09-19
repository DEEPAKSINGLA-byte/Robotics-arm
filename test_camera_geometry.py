import unittest
import numpy as np
from camera_geometry import (Camera, unproject_pixels, unproject_depth, project_points,
                             transform_points, inverse_transform, validate_transform, masked_surface)
from audit_ocid_geometry import fit_camera, select_training_scenes


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(640, 480, 500., 510., 320., 240.)

    def test_known_pixel_and_projection(self):
        uv = np.array([[320., 240.], [340., 240.]])
        points = unproject_pixels(uv, np.array([.8, .8]), self.camera)
        np.testing.assert_allclose(points, [[0, 0, .8], [.032, 0, .8]])
        np.testing.assert_allclose(project_points(points, self.camera), uv)

    def test_invalid_depth_stays_unknown(self):
        uv = np.tile([320., 240.], (4, 1))
        with np.errstate(invalid='raise', divide='raise'):
            self.assertTrue(np.isnan(unproject_pixels(uv, [0, -1, np.nan, np.inf], self.camera)).all())
        self.assertTrue(np.isnan(project_points([[0, 0, 0], [0, 0, -1]], self.camera)).all())
        with self.assertRaises(ValueError):
            unproject_depth(np.ones((224, 224)), self.camera)

    def test_camera_crop_resize_preserves_rays(self):
        uv = np.array([[200., 150.], [400., 300.]])
        new = self.camera.crop_resize(100, 50, 400, 350, 224, 224)
        mapped = (uv-[100, 50]+.5)*[224/400, 224/350]-.5
        np.testing.assert_allclose(unproject_pixels(uv, np.ones(2), self.camera),
                                   unproject_pixels(mapped, np.ones(2), new))

    def test_rigid_transform_roundtrip_and_known_answer(self):
        t = np.array([[0, -1, 0, .1], [1, 0, 0, .2], [0, 0, 1, .3], [0, 0, 0, 1]])
        points = np.array([[1., 0., 0.], [0., 0., .8]])
        moved = transform_points(points, t)
        np.testing.assert_allclose(moved, [[.1, 1.2, .3], [.1, .2, 1.1]])
        np.testing.assert_allclose(transform_points(moved, inverse_transform(t)), points, atol=1e-15)
        with self.assertRaises(ValueError):
            validate_transform(np.diag([2, 1, 1, 1]))
        with self.assertRaises(ValueError):
            validate_transform(np.diag([-1, 1, 1, 1]))

    def test_projection_fit_recovers_known_camera(self):
        rng = np.random.default_rng(42)
        uv = rng.uniform([0, 0], [639, 479], (500, 2))
        points = unproject_pixels(uv, rng.uniform(.5, 1.5, 500), self.camera)
        camera = fit_camera(points, uv, 640, 480)
        np.testing.assert_allclose([camera.fx, camera.fy, camera.cx, camera.cy], [500, 510, 320, 240])

    def test_reference_audit_rejects_val_and_test(self):
        for split in ('val', 'test'):
            with self.assertRaises(ValueError):
                select_training_scenes(dict(split=split, mode='sequence', rows=[]))

    def test_masked_surface_excludes_invalid_points(self):
        camera = Camera(2, 2, 1., 1., 0., 0.)
        points = unproject_depth([[1, 0], [1, 1]], camera)
        surface = masked_surface(points, np.array([[True, True], [False, False]]))
        np.testing.assert_allclose(surface, [[0, 0, 1]])


if __name__ == '__main__':
    unittest.main()
