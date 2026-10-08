"""CPU regressions for Lamp contact gating, full seating and hollow panel geometry."""

import math
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull

from isaaclab_hiveboard.assets import HONEYCOMB_COLLISION_USD, HONEYCOMB_USD
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    RotateFrameCfg,
    ScrewFrameCfg,
    ScrewJointCouplingCfg,
    SequentialPoseCommand,
    _RotateFrameHandler,
    _ScrewFrameHandler,
)
from isaaclab_hiveboard.tasks.anymal.lamp.env import AnymalLampEnvCfg, AnymalLampEnvCfg_PLAY
from isaaclab_hiveboard.tasks.franka.lamp.env import FrankaLampEnvCfg, FrankaLampEnvCfg_PLAY
from isaaclab_hiveboard.tasks.spot.lamp.env import SpotLampEnvCfg, SpotLampEnvCfg_PLAY
from isaaclab_hiveboard.utils.command_setup import apply_setup, load_setup


def proxy(value):
    return NS(torch=torch.as_tensor(value, dtype=torch.float32))


def handler_fixture():
    robot = NS(data=NS(root_pos_w=proxy([[0, 0, 0]] * 2), root_quat_w=proxy([[0, 0, 0, 1]] * 2)))
    frame = NS(data=NS(target_frame_names=["bulb"], target_pos_w=proxy([[[0, 0, 0]]] * 2),
                       target_quat_w=proxy([[[0, 0, 0, 1]]] * 2)))
    scene = {"frame": frame}
    for name in ("left", "right"):
        scene[name] = NS(data=NS(force_matrix_w=proxy(torch.zeros(2, 1, 1, 3))))
    coupling = ScrewJointCouplingCfg(asset_name="lamp", command_joint_angle_scale=-1.0,
                                    grasp_sensor_names=("left", "right"), grasp_timeout_s=0.15)
    term = NS(_asset=robot, _env=NS(scene=scene, step_dt=0.05), num_envs=2, device="cpu",
              cfg=NS(debug_vis=False, screw_coupling=coupling), _valve_asset=None,
              _screw_asset=NS(data=NS(joint_pos=proxy([[0, 0.024]] * 2))), _screw_revolute_idx=0,
              _screw_revolute_target=torch.zeros(2))
    term._get_ee_in_base_frame = lambda ids: (
        torch.tensor([[0.076, 0, 0]] * len(ids)), torch.tensor([[0, 0, 0, 1.0]] * len(ids))
    )
    handler = _ScrewFrameHandler(
        ScrewFrameCfg(frame_name="frame", target_frame_name="bulb", axis=(1, 0, 0),
                      angle_deg=90, use_valve_angle=False, angular_velocity=1.5, axial_distance=0.0015), term
    )
    handler.reset(torch.arange(2))
    term._current_command_idx = torch.zeros(2, dtype=torch.long)
    term._command_handlers = [handler]
    return handler, term


class GraspTests(unittest.TestCase):
    def test_both_filtered_fingers_are_required_per_environment(self):
        handler, term = handler_fixture()
        scene = term._env.scene
        scene["left"].data.force_matrix_w.torch[:, 0, 0, 0] = 0.2
        scene["right"].data.force_matrix_w.torch[0, 0, 0, 0] = 0.2
        handler.update(torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(handler._progress_abs, torch.tensor([0.075, 0.0]))
        torch.testing.assert_close(term._screw_revolute_target, torch.tensor([-0.075, 0.0]))

    def test_lost_grasp_pauses_motion_and_holds_the_measured_joint(self):
        handler, term = handler_fixture()
        for sensor in ("left", "right"):
            term._env.scene[sensor].data.force_matrix_w.torch[..., 0] = 0.2
        handler.update(torch.ones(2, dtype=torch.bool))
        term._screw_asset.data.joint_pos.torch[:, 0] = -0.03
        term._env.scene["right"].data.force_matrix_w.torch.zero_()
        handler.update(torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(handler._progress_abs, torch.tensor([0.075, 0.075]))
        torch.testing.assert_close(term._screw_revolute_target, torch.tensor([-0.03, -0.03]))

    def test_missing_grasp_fails_and_reset_clears_only_selected_environments(self):
        handler, term = handler_fixture()
        for _ in range(4):
            handler.update(torch.ones(2, dtype=torch.bool))
        self.assertTrue(SequentialPoseCommand.screw_grasp_failed(term).all())
        handler.reset(torch.tensor([0]))
        self.assertEqual(handler.grasp_failed.tolist(), [False, True])
        self.assertFalse(handler.is_done(torch.arange(2)).any())

    def test_reacquired_contact_resets_the_continuous_loss_timer(self):
        handler, term = handler_fixture()
        handler.update(torch.ones(2, dtype=torch.bool))
        for name in ("left", "right"):
            term._env.scene[name].data.force_matrix_w.torch[..., 0] = 0.2
        handler.update(torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(handler._grasp_wait_s, torch.zeros(2))

    def test_finished_or_inactive_screw_segments_cannot_fail_the_next_segment(self):
        handler, term = handler_fixture()
        handler.grasp_failed[:] = True
        term._current_command_idx[:] = 1
        self.assertFalse(SequentialPoseCommand.screw_grasp_failed(term).any())

    def test_completion_requires_the_mechanism_to_follow_the_command(self):
        handler, term = handler_fixture()
        handler._progress_abs[:] = math.pi / 2
        self.assertFalse(handler.is_done(torch.arange(2)).any())
        term._screw_asset.data.joint_pos.torch[:, 0] = -math.pi / 2
        self.assertTrue(handler.is_done(torch.arange(2)).all())

    def test_unfiltered_sensors_fail_closed(self):
        handler, term = handler_fixture()
        term._env.scene["left"].data.force_matrix_w = None
        with self.assertRaisesRegex(RuntimeError, "contact filter"):
            handler.update(torch.ones(2, dtype=torch.bool))

    def test_other_screw_tasks_can_keep_ungated_motion(self):
        handler, term = handler_fixture()
        term.cfg.screw_coupling.grasp_sensor_names = ()
        handler.update(torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(handler._progress_abs, torch.full((2,), 0.075))


class SequenceTests(unittest.TestCase):
    def test_wrist_reset_keeps_an_off_axis_tcp_at_its_measured_position(self):
        _, term = handler_fixture()
        position = torch.tensor([[0.076, 0.02, -0.01]] * 2)
        term._get_ee_in_base_frame = lambda ids: (
            position[ids], torch.tensor([[0, 0, 0, 1.0]] * len(ids))
        )
        handler = _RotateFrameHandler(
            RotateFrameCfg(frame_name="frame", target_frame_name="bulb", axis=(1, 0, 0),
                           angle_deg=-180, rotate_in_place=True, use_valve_angle=False), term
        )
        handler.reset(torch.arange(2))
        for _ in range(5):
            command = handler.update(torch.ones(2, dtype=torch.bool))
            torch.testing.assert_close(command[:, 1:4], position)
        target, _ = handler.get_target_in_base_frame(torch.arange(2))
        torch.testing.assert_close(target, position)

    def check_sequence(self, cfg):
        commands = cfg.commands.pose_command.commands
        turns = [cmd for cmd in commands if isinstance(cmd, ScrewFrameCfg)]
        self.assertEqual([cmd.angle_deg for cmd in turns], [180.0] * 7 + [90.0])
        self.assertAlmostEqual(sum(cmd.axial_distance for cmd in turns), 0.024)
        self.assertAlmostEqual(cfg.commands.pose_command.screw_coupling.pitch_m_per_revolution, 0.0064)
        for turn in turns:
            self.assertAlmostEqual(turn.axial_distance, turn.angle_deg / 360 * 0.0064)
        unwinds = [cmd for cmd in commands if isinstance(cmd, RotateFrameCfg) and not isinstance(cmd, ScrewFrameCfg)]
        self.assertEqual([cmd.angle_deg for cmd in unwinds], [-180.0] * 7)
        for index, cmd in enumerate(commands):
            if cmd in unwinds:
                self.assertTrue(cmd.rotate_in_place)
                self.assertEqual(cmd.target_frame_name, "screw_frame")
                self.assertTrue(commands[index - 1].open_gripper)
                self.assertFalse(commands[index + 1].open_gripper)
        move_frames = [cmd.target_frame_name for cmd in commands if hasattr(cmd, "hold_current_orientation")]
        self.assertEqual(move_frames, ["approaching", "lamp_grasp", "release_frame"])
        self.assertGreaterEqual(cfg.episode_length_s, 100)
        self.assertEqual(cfg.commands.pose_command.body_offset.pos, cfg.actions.arm_action.body_offset.pos)
        self.assertEqual(cfg.commands.pose_command.body_offset.pos, cfg.scene.ee_frame.target_frames[0].offset.pos)
        for index, cmd in enumerate(commands):
            if getattr(cmd, "target_frame_name", None) == "release_frame":
                if isinstance(cfg, AnymalLampEnvCfg):
                    self.assertTrue(cmd.gripper_open)
                    self.assertTrue(commands[index - 1].open_gripper)
                    self.assertIsInstance(commands[index - 2], ScrewFrameCfg)
                else:
                    self.assertFalse(cmd.gripper_open)
                    self.assertTrue(commands[index + 1].open_gripper)
        release = next(frame for frame in cfg.scene.target_frame.target_frames if frame.name == "release_frame")
        self.assertTrue(release.prim_path.endswith("/Geometry/World"))
        for name in cfg.commands.pose_command.screw_coupling.grasp_sensor_names:
            sensor = getattr(cfg.scene, name)
            self.assertTrue(sensor.filter_prim_paths_expr[0].endswith("/lamp_pivot"))

    def test_every_robot_and_play_variant_seats_the_bulb(self):
        for cls in (SpotLampEnvCfg, SpotLampEnvCfg_PLAY, FrankaLampEnvCfg, FrankaLampEnvCfg_PLAY,
                    AnymalLampEnvCfg, AnymalLampEnvCfg_PLAY):
            with self.subTest(task=cls.__name__):
                self.check_sequence(cls())

    def test_saved_setups_keep_full_travel_and_safe_release(self):
        root = Path(__file__).resolve().parents[1]
        for robot, cls in (("Spot", SpotLampEnvCfg), ("Franka", FrankaLampEnvCfg)):
            task = f"Isaac-HiveBoard-{robot}-Lamp-v0"
            cfg = cls()
            apply_setup(cfg, load_setup(root / "configs" / f"{task}.json"), task=task)
            self.check_sequence(cfg)


class PanelTests(unittest.TestCase):
    def test_colliders_preserve_visual_bounds_and_hollow_centres(self):
        stage = Usd.Stage.Open(HONEYCOMB_COLLISION_USD)
        colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        self.assertEqual(len(colliders), 84)
        cache = UsdGeom.XformCache()
        world_points = []
        for prim in colliders:
            points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            # Origin of each cell must remain in the hole, at every axial depth.
            for depth in (2.5, 10.0, 19.0):
                equations = ConvexHull(points).equations
                self.assertTrue(np.any(equations[:, :3] @ [0, 0, depth] + equations[:, 3] > 1e-4))
            transform = cache.GetLocalToWorldTransform(prim)
            world_points.extend(transform.Transform(Gf.Vec3d(*(float(v) for v in point))) for point in points)
        visual_stage = Usd.Stage.Open(HONEYCOMB_USD)
        visual_points = []
        # Rotated AABBs overestimate the diagonal cells; compare real vertices.
        for prim in visual_stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                transform = cache.GetLocalToWorldTransform(prim)
                visual_points.extend(transform.Transform(Gf.Vec3d(point))
                                     for point in UsdGeom.Mesh(prim).GetPointsAttr().Get())
        np.testing.assert_allclose(np.min(world_points, axis=0), np.min(visual_points, axis=0), atol=0.001)
        np.testing.assert_allclose(np.max(world_points, axis=0), np.max(visual_points, axis=0), atol=0.001)


if __name__ == "__main__":
    unittest.main()
