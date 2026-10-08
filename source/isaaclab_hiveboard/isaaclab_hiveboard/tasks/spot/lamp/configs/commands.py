from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import as_command_offset
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    ScrewFrameCfg,
    ScrewJointCouplingCfg,
    SequentialPoseCommandCfg,
)

from .scene import LAMP_UNSCREWED_POSITION, SPOT_LAMP_EE

LAMP_TURN_ANGLES = (180.0,) * 7 + (90.0,)
# Measured seating: 7.5 half-turns over the model's 24 mm travel.
LAMP_PITCH_M_PER_REVOLUTION = LAMP_UNSCREWED_POSITION / (sum(LAMP_TURN_ANGLES) / 360.0)


def lamp_commands(*, open_before_retreat: bool = False) -> list:
    """Seat the bulb with wrist resets in place and a single final retreat."""
    commands = [
        GoToFrameCfg(
            frame_name="target_frame",
            target_frame_name="approaching",
            gripper_open=True,
            distance_threshold=0.015,
            orientation_threshold_deg=10.0,
            canonicalize_upward=False,
        ),
        GoToFrameCfg(
            frame_name="target_frame",
            target_frame_name="lamp_grasp",
            gripper_open=True,
            distance_threshold=0.015,
            orientation_threshold_deg=10.0,
            hold_current_orientation=True,
        ),
        GripperCommand(open_gripper=False, duration_s=0.75),
    ]
    for turn, angle_deg in enumerate(LAMP_TURN_ANGLES):
        commands.append(ScrewFrameCfg(
            frame_name="target_frame",
            target_frame_name="screw_frame",
            axis=(1.0, 0.0, 0.0),
            angle_deg=angle_deg,
            axial_distance=LAMP_PITCH_M_PER_REVOLUTION * angle_deg / 360.0,
            angular_velocity=1.5,
            angle_threshold_deg=2.0,
            gripper_open=False,
        ))
        if turn < len(LAMP_TURN_ANGLES) - 1:
            commands.extend([
                GripperCommand(open_gripper=True, duration_s=0.5),
                RotateFrameCfg(
                    frame_name="target_frame",
                    target_frame_name="screw_frame",
                    axis=(1.0, 0.0, 0.0),
                    angle_deg=-angle_deg,
                    angular_velocity=1.5,
                    angle_threshold_deg=2.0,
                    gripper_open=True,
                    rotate_in_place=True,
                    use_valve_angle=False,
                ),
                GripperCommand(open_gripper=False, duration_s=0.75),
            ])
    retreat = GoToFrameCfg(
        frame_name="target_frame",
        target_frame_name="release_frame",
        gripper_open=open_before_retreat,
        distance_threshold=0.015,
        orientation_threshold_deg=10.0,
        hold_current_orientation=True,
    )
    release = GripperCommand(open_gripper=True, duration_s=0.5)
    commands.extend([release, retreat] if open_before_retreat else [retreat, release])
    return commands


@configclass
class FramePoseCommandsCfg:
    """Seat the lamp with seven half-turns, a final quarter-turn and wrist unwinds."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=SPOT_LAMP_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        commands=lamp_commands(),
        body_offset=as_command_offset(SPOT_LAMP_EE),
        screw_coupling=ScrewJointCouplingCfg(
            asset_name="lamp",
            pitch_m_per_revolution=LAMP_PITCH_M_PER_REVOLUTION,
            # The lamp joint axis points opposite Spot's TCP screw axis after
            # applying the lamp's spawn rotation.
            command_joint_angle_scale=-1.0,
            grasp_sensor_names=("finger_contact", "jaw_contact"),
            lower_limit=0.0,
            upper_limit=0.024,
            # Solver-level backstop for the per-step position target above —
            # see ScrewJointCouplingCfg's docstring for why this needs a
            # MODEL_INIT builder hook instead of a USD-authored <mimic>.
            use_native_mimic_constraint=True,
        ),
    )
