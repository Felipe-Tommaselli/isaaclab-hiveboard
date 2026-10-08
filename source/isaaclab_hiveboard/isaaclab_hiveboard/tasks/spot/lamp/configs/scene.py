from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab_hiveboard.assets import (
    LAMP_NEWTON_USD,
    SPOT_EE,
    make_ee_frame,
)
from isaaclab_hiveboard.assets.spot.constants import ARM_JOINT_NAMES, SPOT_DEFAULT_JOINT_POS
from isaaclab_hiveboard.assets.spot.spot import (
    SPOT_ARM_NEWTON_CFG,
    SPOT_ARM_UUC_BODY_PRIM,
    SPOT_ARM_UUC_FNGR_PRIM,
    SPOT_ARM_UUC_JAW_PRIM,
    SPOT_ARM_UUC_SOURCE_PRIM,
)

from isaaclab_tasks.core.cabinet.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
)

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.scenes.mechanism import GROUND, LIGHT, honeycomb

# Turn the lamp's local +X axis toward Spot. The socket remains behind it.
LAMP_SPAWN_POS = (1.0, 0.0, 0.65)
# 180 degrees about Z, in Isaac Lab 3 xyzw order.
LAMP_SPAWN_QUAT = (0.0, 0.0, 1.0, 0.0)
LAMP_UNSCREWED_POSITION = 0.024
LAMP_SEATED_POSITION = 0.0

# The fixed jaw's inner surface is 32 mm below wr1. Centre the 56 mm bulb
# above it; the thinner valve TCP would drive that jaw through the bulb.
SPOT_LAMP_EE = SPOT_EE.replace(tcp_offset=SPOT_EE.tcp_offset.replace(pos=(0.21, 0.0, -0.006)))

# All offsets are expressed in the lamp's local frame. The lamp bulb extends
# along +X and the prismatic joint translates it by another 24 mm when open.
# The local 180-degree rotation cancels the articulation spawn rotation for
# orientation only, making canonical TCP +X point from Spot toward the lamp.
LAMP_TARGET_QUAT = LAMP_SPAWN_QUAT
# Offsets from ``lamp_pivot``. They follow the bulb inward as the screw
# coupling advances it; the grasp stays near the accessible front surface.
LAMP_APPROACH_OFFSET = OffsetCfg(pos=(0.126, 0.0, 0.0), rot=LAMP_TARGET_QUAT)
# Forty millimetres deeper than the original surface grasp, on the bulb axis.
LAMP_GRASP_OFFSET = OffsetCfg(pos=(0.076, 0.0, 0.0), rot=LAMP_TARGET_QUAT)
LAMP_ROTATION_OFFSET = OffsetCfg(pos=(0.076, 0.0, 0.0), rot=LAMP_TARGET_QUAT)
# A fixed socket-relative clearance, beyond the fully unscrewed bulb. The
# retreat must stay below the bulb even after it has rotated several turns.
LAMP_RELEASE_OFFSET = OffsetCfg(pos=(0.150, 0.0, -0.030), rot=LAMP_TARGET_QUAT)


@configclass
class LampSceneCfg(InteractiveSceneCfg):
    """Fixed-base Spot facing a screw-in lamp on the HiveBoard panel."""

    robot: ArticulationCfg = SPOT_ARM_NEWTON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ground: AssetBaseCfg = GROUND
    light: AssetBaseCfg = LIGHT

    lamp = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Lamp",
        spawn=sim_utils.UsdFileCfg(
            usd_path=LAMP_NEWTON_USD,
            activate_contact_sensors=True,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                linear_damping=2.0,
                angular_damping=1.0,
                max_linear_velocity=10.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=0.20,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
            semantic_tags=[("class", "lamp")],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=LAMP_SPAWN_POS,
            rot=LAMP_SPAWN_QUAT,
            joint_pos={
                "PrismaticJoint": LAMP_UNSCREWED_POSITION,
                "RevoluteJoint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "rotation": IdealPDActuatorCfg(
                joint_names_expr=["RevoluteJoint"],
                # Compliant PD so ScrewFrame can turn the bulb with the gripper
                # instead of sweeping the fingers through a stationary mesh.
                stiffness=80.0,
                # The explicit drive updates at 200 Hz. Damping 8.0 with this
                # inertia oscillates even at rest and rotates the approach
                # reference before grasping; solver substeps do not update PD.
                damping=1.0,
                armature=0.01,
                friction=0.0,
                dynamic_friction=0.0,
                viscous_friction=0.0,
                effort_limit=100.0,
            ),
            "insertion": IdealPDActuatorCfg(
                joint_names_expr=["PrismaticJoint"],
                # The environment updates this drive target from measured
                # revolute travel using the calibrated command coupling pitch. The
                # USD values (50 kN/m, 300 N-s/m) are too stiff for the URDF's
                # light bulb link at this timestep, so use a stable equivalent.
                stiffness=5000.0,
                damping=50.0,
                friction=0.0,
                dynamic_friction=0.0,
                viscous_friction=0.0,
                effort_limit=100.0,
            ),
        },
    )

    # The fixed socket's rear face is 44.7325 mm behind the lamp joint origin.
    # Parenting to World keeps the panel at the socket pose for every robot
    # workspace and reset, without following the bulb's screw motion.
    honeycomb = honeycomb("{ENV_REGEX_NS}/Lamp/Geometry/World", back_x=-0.0447325, collision_enabled=True)

    target_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Lamp/Geometry/World/rotation_pivot/lamp_pivot",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LampTransformers"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Lamp/Geometry/World/rotation_pivot/lamp_pivot",
                name="approaching",
                offset=LAMP_APPROACH_OFFSET,
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Lamp/Geometry/World/rotation_pivot/lamp_pivot",
                name="lamp_grasp",
                offset=LAMP_GRASP_OFFSET,
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Lamp/Geometry/World/rotation_pivot/lamp_pivot",
                name="screw_frame",
                offset=LAMP_ROTATION_OFFSET,
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Lamp/Geometry/World",
                name="release_frame",
                offset=LAMP_RELEASE_OFFSET,
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Lamp/Geometry/World",
                name="unwind_frame",
                offset=LAMP_RELEASE_OFFSET,
            ),
        ],
    )

    ee_frame: FrameTransformerCfg = make_ee_frame(SPOT_LAMP_EE)

    finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + SPOT_ARM_UUC_FNGR_PRIM,
        update_period=0.0,
        debug_vis=False,
        track_pose=True,
        force_threshold=0.1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Lamp/Geometry/World/rotation_pivot/lamp_pivot"],
    )

    jaw_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + SPOT_ARM_UUC_JAW_PRIM,
        update_period=0.0,
        debug_vis=False,
        track_pose=True,
        force_threshold=0.1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Lamp/Geometry/World/rotation_pivot/lamp_pivot"],
    )

    def __post_init__(self):
        # Starting folded over the back sends differential IK into a wrist
        # joint limit before it reaches the lamp. Start in the forward arm pose.
        self.robot.init_state.joint_pos.update({name: SPOT_DEFAULT_JOINT_POS[name] for name in ARM_JOINT_NAMES})
        self.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/" + SPOT_ARM_UUC_SOURCE_PRIM
        self.ee_frame.target_frames[0].prim_path = "{ENV_REGEX_NS}/Robot/" + SPOT_ARM_UUC_BODY_PRIM
