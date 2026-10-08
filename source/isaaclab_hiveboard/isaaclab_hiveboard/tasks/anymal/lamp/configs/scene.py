# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ANYMAL_EE, ANYMAL_WORKSPACE, make_ee_frame
from isaaclab_hiveboard.assets.anymal.anymal import ROBOTIQ_LEFT_PAD_PRIM, ROBOTIQ_RIGHT_PAD_PRIM
from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_NEWTON_CFG
from isaaclab_hiveboard.tasks.spot.lamp.configs.scene import LampSceneCfg


@configclass
class AnymalLampSceneCfg(LampSceneCfg):
    """ANYmal-D + DynaArm + 2F-140 facing the shared screw-in lamp."""

    robot: ArticulationCfg = ANYMAL_ARM_NEWTON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = make_ee_frame(ANYMAL_EE)
    finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + ROBOTIQ_LEFT_PAD_PRIM,
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Lamp/Geometry/World/rotation_pivot/lamp_pivot"],
    )
    jaw_contact = finger_contact.replace(prim_path="{ENV_REGEX_NS}/Robot/" + ROBOTIQ_RIGHT_PAD_PRIM)

    def __post_init__(self):
        self.lamp.init_state.pos = ANYMAL_WORKSPACE.object_pos
