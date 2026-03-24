
#what data looks like in robosuite env

env_meta:  {'env_name': 'Lift', 'env_version': '1.5.1', 'type': 1, 'env_kwargs': {'has_renderer': False, 'has_offscreen_renderer': False, 'ignore_done': True, 'use_object_obs': True, 'use_camera_obs': False, 'control_freq': 20, 'controller_configs': {'type': 'BASIC', 'body_parts': {'right': {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_limits': [0, 10], 'position_limits': None, 'orientation_limits': None, 'uncouple_pos_ori': True, 'control_delta': True, 'interpolation': None, 'ramp_ratio': 0.2, 'input_ref_frame': 'world', 'gripper': {'type': 'GRIP'}}}}, 'robots': ['Panda'], 'camera_depths': False, 'camera_heights': 84, 'camera_widths': 84, 'lite_physics': False, 'reward_shaping': False}}

 Robosuite observation keys: odict_keys(['robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_joint_acc', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_quat_site', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'cube_pos', 'cube_quat', 'gripper_to_cube_pos', 'robot0_proprio-state', 'object-state'])

 Action space: (array([-1., -1., -1., -1., -1., -1., -1.]), array([1., 1., 1., 1., 1., 1., 1.]))
DataCollectionWrapper: making folder at Diffusion_Policy/videos/ep_1771972343_2502685

 obs:  OrderedDict([('robot0_joint_pos', array([-0.01697571,  0.19283784, -0.02274446, -2.65940527, -0.06687063,
        2.96784836,  0.78485425])), ('robot0_joint_pos_cos', array([ 0.99985592,  0.98146433,  0.99974136, -0.88598271,  0.99776499,
       -0.98494439,  0.70749128])), ('robot0_joint_pos_sin', array([-0.0169749 ,  0.1916449 , -0.0227425 , -0.46371827, -0.0668208 ,
        0.17287147,  0.70672207])), ('robot0_joint_vel', array([-0.03277907, -0.77296252, -1.57065171, -0.71834096, -2.99935884,
       -0.53969942, -0.05334823])), ('robot0_joint_acc', array([  5.26751117,  -8.31320171, -19.14403617,   2.02926726,
       -46.43530685, -13.14185354,   9.81203575])), ('robot0_eef_pos', array([-0.11863418, -0.0137289 ,  1.0026901 ])), ('robot0_eef_quat', array([0.99819219, 0.0121609 , 0.05801556, 0.00993256])), ('robot0_eef_quat_site', array([0.6972294 , 0.7144275 , 0.03399982, 0.04804658], dtype=float32)), ('robot0_gripper_qpos', array([ 0.02092817, -0.02222375])), ('robot0_gripper_qvel', array([ 0.00125286, -0.03028341])), ('cube_pos', array([-0.01966099, -0.00489296,  0.81981015])), ('cube_quat', array([-8.13413818e-19,  9.29256688e-19,  9.31361461e-01,  3.64095907e-01])), ('gripper_to_cube_pos', array([ 0.09897319,  0.00883594, -0.18287995])), ('robot0_proprio-state', array([-1.69757120e-02,  1.92837841e-01, -2.27444627e-02, -2.65940527e+00,
       -6.68706284e-02,  2.96784836e+00,  7.84854249e-01,  9.99855916e-01,
        9.81464330e-01,  9.99741356e-01, -8.85982712e-01,  9.97764993e-01,
       -9.84944391e-01,  7.07491282e-01, -1.69748967e-02,  1.91644902e-01,
       -2.27425018e-02, -4.63718270e-01, -6.68208022e-02,  1.72871474e-01,
        7.06722071e-01, -3.27790693e-02, -7.72962522e-01, -1.57065171e+00,
       -7.18340960e-01, -2.99935884e+00, -5.39699420e-01, -5.33482254e-02,
        5.26751117e+00, -8.31320171e+00, -1.91440362e+01,  2.02926726e+00,
       -4.64353068e+01, -1.31418535e+01,  9.81203575e+00, -1.18634181e-01,
       -1.37288989e-02,  1.00269010e+00,  9.98192191e-01,  1.21609046e-02,
        5.80155630e-02,  9.93256098e-03,  6.97229385e-01,  7.14427471e-01,
        3.39998193e-02,  4.80465777e-02,  2.09281739e-02, -2.22237451e-02,
        1.25285672e-03, -3.02834121e-02])), ('object-state', array([-1.96609948e-02, -4.89295785e-03,  8.19810146e-01, -8.13413818e-19,
        9.29256688e-19,  9.31361461e-01,  3.64095907e-01,  9.89731860e-02,
        8.83594104e-03, -1.82879954e-01]))])

 reward:  0.0

 done:  False

 info:  {}

 action:  [-0.9763439   0.33165688  0.94155646  0.12221566 -0.14534915 -0.3673559
  0.8000048 ]


  --------------------------------------------------------------------------------------------------------------


Ai breakdown, edited and formatted by me for more depth. 

Observations: how they map to robot parts
From debug_stuff.md, you’re using the Lift task in robosuite with a Panda arm and a BASIC → OSC_POSE controller. 

BASIC is a high-level controller configuration in robosuite. It’s a convenience preset that says: “Use a standard control setup for this robot,” and then internally picks a specific low-level controller type per body part (arm, gripper, etc.) with some default gains, limits, etc.

OSC_POSE
This is the actual low-level arm controller being used:
       OSC = Operational Space Control → control in task space (EEF pose) rather than directly in joint space.
       POSE = the command is a 6D pose (position + orientation), typically as deltas when control_delta=True.

So “BASIC → OSC_POSE” means:
You’re using the BASIC controller preset, which for the right arm selects an operational-space pose controller (OSC_POSE). Your 6D action (first 6 dims) is interpreted as desired changes in the end-effector’s position and orientation, and the controller computes the joint torques/velocities needed to achieve that.

The observation dict you printed is:

robot0_joint_pos
Joint angles of the 7 DoF Panda arm (shoulder, elbow, wrist, etc.). Each element = angle of one joint.

robot0_joint_pos_cos, robot0_joint_pos_sin
Cosine and sine of each joint angle. These are just a different encoding of robot0_joint_pos to avoid angle wraparound issues when learning.

robot0_joint_vel
Joint velocities of the 7 arm joints (how fast each joint angle is changing).

robot0_joint_acc
Joint accelerations of the 7 joints (rate of change of velocity).

robot0_eef_pos
3D position of the end-effector (the gripper “hand”) in world coordinates: $[x, y, z]$.

robot0_eef_quat
Orientation of the end-effector as a quaternion (4 numbers).

robot0_eef_quat_site
Similar orientation info, but taken from a specific MuJoCo “site” attached to the EEF (slightly different frame / reference point, used internally by robosuite).

robot0_gripper_qpos
Gripper finger joint positions (for Panda: 2 numbers, left/right finger opening).

robot0_gripper_qvel
Velocities of those gripper finger joints.

cube_pos
3D position of the cube in world coordinates.

cube_quat
Orientation of the cube as a quaternion.

gripper_to_cube_pos
Vector from gripper to cube: cube_pos - eef_pos. This is a task-relevant relative position (how far and in what direction the gripper is from the cube).

robot0_proprio-state
A concatenated proprioceptive vector for the robot. In your printout, it’s just all of the robot-related pieces flattened together (joint positions, cos/sin of those, velocities, accelerations, EEF pose, gripper state, etc.). It’s basically a ready-made “robot state” feature vector.

object-state
A concatenated object-related vector: cube position, cube orientation, and relative position from gripper to cube (exactly those 10 numbers you see at the end).

Together, robot0_proprio-state + object-state are typically what a state-based policy uses as its low-dimensional observation.

-----------------------------------------------------------------------------------------------------------------------
Actions: how they affect the robot parts


Action space: [-1, …, -1] to [1, …, 1] with shape (7,)

From your env metadata in debug_stuff.md:

Controller config:
       Type: OSC_POSE:
       body_parts → right → output_max: [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
       output_min: [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]
       gripper: {'type': 'GRIP'}
       control_delta: True

So your 7D action is:
       First 6 elements:
       Scaled deltas for the end-effector pose:
              3 elements → $\Delta x, \Delta y, \Delta z$ (meters per step, limited by ±0.05)
              3 elements → orientation deltas (rotation in some axis-angle or equivalent representation), scaled by ±0.5

Internally, robosuite maps the normalized actions in [-1, 1] into those physical ranges using output_min / output_max. So e.g. an action of [1, 0, 0, 0, 0, 0, *] will try to move the EEF forward by +0.05 m in x this step.
this is implemented as an operational-space position/orientation controller that tries to move the EEF to a new target pose each step, offset from the current pose by your action. So it’s not a pure joint-velocity controller; it’s closer to position control in task space, where each action sets a small desired position/orientation change.

       7th element:
       The gripper command (via the GRIP controller).
              Positive values → close gripper
              Negative values → open gripper
       Magnitude controls how fast/strongly it moves within each step.
       The GRIP controller interprets this as a command to open/close the fingers, again more like a position / opening-width command than raw joint velocity.

So:actions are conceptually position-controlled (EEF pose deltas), not direct velocity control, even though you’re sending per-step increments.

Putting it together:
The observations tell you:
       Where the arm joints are, how they’re moving (robot0_joint_*)
       Where the hand is and which way it’s pointing (robot0_eef_*)
       How open the gripper is (robot0_gripper_*)
       Where the cube is and how it’s oriented (cube_*, gripper_to_cube_pos, object-state)

The actions are normalized commands that:
       Move the EEF in 3D and adjust its orientation, indirectly changing joint positions/velocities in the next step.
       Open/close the gripper fingers, changing robot0_gripper_qpos and eventually whether the cube is grasped.

------------------------------------------------------------------------------------------------------------------

from explore dataset

Top-level keys: ['data', 'mask']

Episodes in dataset: 200

Keys in demo_0: ['actions', 'dones', 'next_obs', 'obs', 'rewards', 'states']
Actions shape: (59, 7)
Actions: [-0.          0.          0.          0.00381497  0.14820713  0.01447902
 -1.        ]

 Observation keys: ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_quat_site', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel']

 Next observation keys: ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_quat_site', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel']

 States shape: (59, 32)
States: [ 0.         -0.04141039  0.21736869  0.00753974 -2.5898454  -0.00784382
  2.95545758  0.77382831  0.020833   -0.020833    0.02644941  0.02698126
  0.83142407  0.2466312   0.          0.          0.96910941  0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.        ]



action_min:  tensor([-1.0000, -0.5600, -1.0000, -0.1507, -1.0000, -0.5180, -1.0000])
action_max:  tensor([1.0000, 0.6520, 1.0000, 0.1186, 0.3051, 0.4782, 1.0000]

keys for dataset
    obs_keys = [
    "object",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_eef_quat_site",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
    ]

keys for simulation:
    obs_keys = [
    "object-state",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_eef_quat_site",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
    ]

both are 53 dim

--------------------------------------

How to load

ckpt = torch.load("Diffusion_Policy/model.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
action_min = ckpt["action_min"]
action_max = ckpt["action_max"]
state_mean = ckpt["state_mean"]
state_std = ckpt["state_std"]