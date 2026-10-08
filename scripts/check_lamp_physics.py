#!/usr/bin/env python3
"""Check full Lamp seating, bulb contact gating and imported panel colliders.

uv run python scripts/check_lamp_physics.py --visualizer none
uv run python scripts/check_lamp_physics.py --visualizer none --no-grasp
Use --setup to also check a saved command-editor sequence.
"""

import argparse
import math
import sys

import gymnasium as gym
import torch

import isaaclab_hiveboard  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_newton.physics import NewtonManager
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import ScrewFrameCfg
from isaaclab_hiveboard.utils.command_setup import apply_setup, load_setup

from play import _all_finite, _route_command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-HiveBoard-Franka-Lamp-Play-v0")
    parser.add_argument("--setup")
    parser.add_argument("--no-grasp", action="store_true", help="Hold the fingers open: must fail without seating.")
    parser.add_argument("--seed", type=int, default=42)
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *overrides]
    cfg, _ = resolve_task_config(args.task, "")
    if args.setup:
        apply_setup(cfg, load_setup(args.setup), task=args.task)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.recorders = None
    cfg.commands.pose_command.log_transitions = True
    cfg.commands.pose_command.stall_timeout_s = 5.0
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        try:
            base = env.unwrapped
            term = base.command_manager.get_term("pose_command")
            model = NewtonManager.get_model()
            colliders = [label for label in model.shape_label if "/Honeycomb/Collisions/" in label]
            assert len(colliders) == 84, f"Expected 84 hollow-cell panel colliders, got {len(colliders)}"
            lamp = base.scene["lamp"]
            axial = lamp.find_joints("PrismaticJoint")[0][0]
            angle = lamp.find_joints("RevoluteJoint")[0][0]
            obs, _ = env.reset(seed=args.seed)
            minimum_axial = 0.024
            minimum_angle = 0.0
            contact_turns = set()
            for step in range(base.max_episode_length):
                segment = int(term._current_command_idx[0])
                minimum_axial = min(minimum_axial, float(lamp.data.joint_pos.torch[0, axial]))
                minimum_angle = min(minimum_angle, float(lamp.data.joint_pos.torch[0, angle]))
                if segment < len(term.cfg.commands) and isinstance(term.cfg.commands[segment], ScrewFrameCfg):
                    handler = term._command_handlers[segment]
                    if bool(handler._grasp_contact_mask(torch.tensor([0], device=base.device))[0]):
                        contact_turns.add(segment)
                    if step % 10 == 0:
                        forces = [
                            float(torch.linalg.vector_norm(base.scene[name].data.force_matrix_w.torch, dim=-1).max())
                            for name in term.cfg.screw_coupling.grasp_sensor_names
                        ]
                        print(f"[LAMP TRACE] t={step * base.step_dt:.2f} seg={segment} "
                              f"slide={minimum_axial:.6f} angle={minimum_angle:.3f} forces={forces}", flush=True)
                assert _all_finite(obs), f"Non-finite observations at step {step}"
                if float(term._seg_time[0]) > 10.0:
                    print(f"[LAMP] FAIL: segment {segment} stalled for ten seconds.", flush=True)
                    return 1
                command = obs["policy"]["command"].clone()
                if args.no_grasp:
                    command[:, 0] = 1.0
                with torch.inference_mode():
                    obs, _, terminated, truncated, info = env.step(_route_command(base, command))
                if bool(terminated[0] or truncated[0]):
                    success = bool(info["log"].get("Episode_Termination/success", False))
                    lost = bool(info["log"].get("Episode_Termination/grasp_failed", False))
                    passed = (lost and not success and minimum_axial >= 0.0235) if args.no_grasp else (
                        success and len(contact_turns) == 8 and minimum_axial <= 0.0005
                        and minimum_angle <= -7.5 * math.pi + 0.5
                    )
                    print(f"[LAMP] {'PASS' if passed else 'FAIL'} task={args.task} no_grasp={args.no_grasp} "
                          f"time={(step + 1) * base.step_dt:.2f}s contact_turns={len(contact_turns)} "
                          f"slide={minimum_axial:.6f} angle={minimum_angle:.3f} success={success} "
                          f"grasp_failed={lost} panel_colliders={len(colliders)}", flush=True)
                    return 0 if passed else 1
            print("[LAMP] FAIL: sequence exceeded the episode limit.", flush=True)
            return 1
        finally:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
