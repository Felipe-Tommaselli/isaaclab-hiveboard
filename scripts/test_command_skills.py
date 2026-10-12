"""Phase-tagged skills and mechanism goals of the cuRobo expert sequences.

Config-only checks (no simulation): the skill factories, the phase vocabulary,
and the JSON round trip of the saved command setups.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    CuroboPlannedRotateFrameCfg,
    GripperCommand,
)
from isaaclab_hiveboard.mdp.commands.skills import PHASES, MechanismGoalCfg, SkillSet
from isaaclab_hiveboard.tasks.anymal.mechanism import ANYMAL_CUROBO
from isaaclab_hiveboard.utils.command_setup import decode_command, encode_command, load_setup, validate_setup

_REPO = Path(__file__).resolve().parents[1]
_BALL_VALVE_SETUP = _REPO / "configs/Isaac-HiveBoard-Anymal-BallValve-v0.json"
_SMALL_VALVE_SETUP = _REPO / "configs/Isaac-HiveBoard-Anymal-SmallValve-v0.json"


def test_skills_tag_their_phase_and_robot():
    anymal = SkillSet(**ANYMAL_CUROBO)
    commands = [
        anymal.approach("approaching"),
        anymal.engage("lever_pivot"),
        anymal.grip(),
        anymal.turn(angle_deg=-90.0),
        anymal.move("pressed"),
        anymal.release(),
        anymal.retreat("approaching"),
    ]
    assert [c.phase for c in commands] == ["approach", "engage", "grip", "actuate", "actuate", "release", "retreat"]
    assert set(c.phase for c in commands) == set(PHASES)
    for cmd in commands:
        if isinstance(cmd, (CuroboPlannedGoToFrameCfg, CuroboPlannedRotateFrameCfg)):
            assert cmd.robot_joint_names == ANYMAL_CUROBO["robot_joint_names"]
            assert cmd.frame_name == "target_frame"
    assert commands[0].gripper_open and not commands[3].gripper_open
    assert commands[2].open_gripper is False and commands[5].open_gripper is True


def test_skill_keywords_override_defaults():
    anymal = SkillSet(**ANYMAL_CUROBO)
    cmd = anymal.approach("approaching", velocity=0.4, chain_with_next=True)
    assert cmd.velocity == 0.4 and cmd.chain_with_next


def test_phase_and_until_survive_the_json_round_trip():
    anymal = SkillSet(**ANYMAL_CUROBO)
    goal = MechanismGoalCfg(goal_tolerance=0.05, settle_speed=0.1)
    cmd = anymal.turn(angle_deg=-90.0, until=goal)
    data = json.loads(json.dumps(encode_command(cmd)))
    assert data["parameters"]["phase"] == "actuate"
    assert data["parameters"]["until"]["goal_tolerance"] == 0.05
    back = decode_command(data)
    assert back.phase == "actuate"
    assert isinstance(back.until, MechanismGoalCfg)
    assert back.until.goal_tolerance == 0.05 and back.until.settle_speed == 0.1 and back.until.low is None


def test_untagged_setups_still_load():
    data = encode_command(GripperCommand(open_gripper=True, duration_s=0.3))
    del data["parameters"]["phase"], data["parameters"]["until"]
    cmd = decode_command(data)
    assert cmd.phase is None and cmd.until is None


@pytest.mark.parametrize(
    "bad",
    [
        {"phase": "wiggle"},
        {"until": {"low": 0.0, "colour": 1.0}},
        {"until": {}},
        {"until": {"settle_speed": -1.0}},
        {"until": {"low": float("nan")}},
    ],
)
def test_bad_phase_or_goal_is_rejected(bad):
    data = encode_command(GripperCommand(open_gripper=True, duration_s=0.3))
    data["parameters"].update(bad)
    with pytest.raises(ValueError):
        decode_command(data)


def test_ball_valve_setup_is_phase_tagged():
    commands, _ = validate_setup(load_setup(_BALL_VALVE_SETUP))
    phases = [c.phase for c in commands]
    assert phases == ["approach", "engage", "actuate", "release", "retreat"]
    # Phases never go backwards within one cycle.
    order = [PHASES.index(p) for p in phases]
    assert order == sorted(order)


def test_small_valve_setup_is_phase_tagged():
    commands, _ = validate_setup(load_setup(_SMALL_VALVE_SETUP))
    assert [c.phase for c in commands] == ["approach", "engage", "grip", "actuate", "release", "retreat"]
    # The approach runs into the engage without stopping.
    assert commands[0].chain_with_next and not commands[1].chain_with_next


@pytest.mark.parametrize("path", sorted((_REPO / "configs").glob("Isaac-HiveBoard-*.json")), ids=lambda p: p.stem)
def test_every_saved_setup_loads(path):
    validate_setup(load_setup(path))


def test_curobo_joint_speed_limits_lower_the_urdf_limits():
    import xml.etree.ElementTree as ET

    from isaaclab_hiveboard.assets import ASSET_DIR
    from isaaclab_hiveboard.mdp.curobo_robot_cfg import limit_joint_velocity, load_curobo_robot_cfg

    yaml_path, urdf_path = f"{ASSET_DIR}/anymal/cumotion/dynaarm.yaml", f"{ASSET_DIR}/anymal/cumotion/dynaarm.urdf"
    robot_cfg = load_curobo_robot_cfg(yaml_path, urdf_path)
    kinematics = robot_cfg["robot_cfg"]["kinematics"]
    assert kinematics["cspace"]["max_acceleration"] == 10.0
    names = kinematics["cspace"]["joint_names"]
    # The URDF allows 4 rad/s; joints left out keep it and a faster limit does not raise it.
    limits = {names[0]: 1.0, names[1]: 1.0, names[4]: 2.0, names[5]: 8.0}
    assert limit_joint_velocity(robot_cfg, limits) is robot_cfg
    assert kinematics["urdf_path"] != urdf_path and "velocity_scale" not in kinematics["cspace"]
    joints = {j.get("name"): j for j in ET.parse(kinematics["urdf_path"]).getroot().iter("joint")}
    speeds = [float(joints[n].find("limit").get("velocity")) for n in names]
    assert speeds == [1.0, 1.0, 4.0, 4.0, 2.0, 4.0]
    with pytest.raises(ValueError, match="no velocity limit"):
        limit_joint_velocity(load_curobo_robot_cfg(yaml_path, urdf_path), {"not_a_joint": 1.0})


def test_plans_are_retimed_to_the_joint_speed_limit():
    import torch

    from isaaclab_hiveboard.mdp.commands.sequential_pose_command import _limit_joint_speed

    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(6, 1)
    pos = torch.zeros(6, 3)
    slow = torch.linspace(0.0, 0.25, 6)[:, None].repeat(1, 6)
    assert _limit_joint_speed(pos, quat, slow, 0.1)[2] is slow
    # 0.05 rad steps, then two of 0.3 rad on one joint.
    joints = torch.zeros(6, 6)
    joints[:, 4] = torch.tensor([0.0, 0.05, 0.1, 0.4, 0.7, 0.75])
    pos, quat, retimed = _limit_joint_speed(pos, quat, joints, 0.1)
    assert len(retimed) == len(pos) == len(quat) == 10
    assert float((retimed[1:] - retimed[:-1]).abs().max()) <= 0.1 + 1.0e-6
    # The slow start keeps its timing and the plan still ends on its last waypoint.
    assert torch.allclose(retimed[:3], joints[:3]) and torch.allclose(retimed[-1], joints[-1])
    # Per-joint limits: the same 0.3 rad steps fit a 0.3 limit on joint 4 but not 0.1 on joint 0.
    per_joint = torch.tensor([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])
    assert _limit_joint_speed(pos[:6], quat[:6], joints, per_joint)[2] is joints
    joints[:, 0] = joints[:, 4]
    assert len(_limit_joint_speed(pos[:6], quat[:6], joints, per_joint)[2]) == 10
    assert CuroboPlannedGoToFrameCfg().max_joint_velocity is None
    assert CuroboPlannedGoToFrameCfg().plan_within_joint_velocity is False
    assert CuroboPlannedRotateFrameCfg().max_joint_velocity is None
