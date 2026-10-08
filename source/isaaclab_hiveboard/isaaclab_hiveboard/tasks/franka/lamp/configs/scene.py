from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import FRANKA_EE, FRANKA_FR3_HIGH_PD_CFG, FRANKA_WORKSPACE, make_ee_frame
from isaaclab_hiveboard.tasks.spot.lamp.configs.scene import LampSceneCfg


@configclass
class FrankaLampSceneCfg(LampSceneCfg):
    """Franka Research 3 facing the shared screw-in lamp."""

    robot: ArticulationCfg = FRANKA_FR3_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.30, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "fr3_joint1": 0.0,
                "fr3_joint2": -0.569,
                "fr3_joint3": 0.0,
                "fr3_joint4": -2.810,
                "fr3_joint5": 0.0,
                "fr3_joint6": 3.037,
                "fr3_joint7": 0.741,
                "fr3_finger_joint.*": 0.04,
            },
        ),
    )
    ee_frame: FrameTransformerCfg = make_ee_frame(FRANKA_EE)
    finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr3_leftfinger",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Lamp/Geometry/World/rotation_pivot/lamp_pivot"],
    )
    jaw_contact = finger_contact.replace(prim_path="{ENV_REGEX_NS}/Robot/fr3_rightfinger")

    def __post_init__(self):
        self.lamp.init_state.pos = FRANKA_WORKSPACE.object_pos
        # Integrate the stiff drives in MJWarp's implicit solver. Explicit
        # PD at this task's 5 ms timestep saturates the arm torques and makes
        # the fingers oscillate even under a constant open command.
        self.robot.actuators = {
            "fr3_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=400.0,
                damping=80.0,
                armature=0.1,
            ),
            "fr3_forearm": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=400.0,
                damping=80.0,
                armature=0.05,
            ),
            "fr3_hand": ImplicitActuatorCfg(
                joint_names_expr=["fr3_finger_joint.*"],
                effort_limit_sim=20.0,
                stiffness=200.0,
                damping=20.0,
                armature=0.01,
            ),
        }
