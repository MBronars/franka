#!/usr/bin/env python
"""
Keyboard teleoperation for the Franka arm.

The end-effector is kept pointed straight down (the reset orientation) and is
driven around a horizontal XY plane.  Other keys raise / lower that plane.

Safety design (read this before running on hardware!)
-----------------------------------------------------
1. Velocity-limited:   keys command a *velocity*, not a jump.  The commanded
                       target only ever moves SPEED_* * dt per control tick,
                       so the arm can never lurch, no matter how you mash keys.
2. Bounding box:       the target is hard-clamped to SAFE_BOX (and to the
                       frankapy virtual walls).  A command that would leave
                       the box is simply ignored on that axis.
3. Joint-limit guard:  every tick the actual joint angles are checked.  If any
                       joint gets within JOINT_LIMIT_MARGIN of its limit, the
                       target is reverted to the last known-good pose -- i.e.
                       the offending command is *not followed*.

This script was written on a server and has NOT been tested on the robot.
Tune SAFE_BOX and the speed constants for your cell before trusting it.

Controls
--------
  w / s  or  up / down arrow   : +x / -x   (away from / toward the base)
  a / d  or  left / right arrow: +y / -y   (robot's left / right)
  r / f                        : raise / lower the XY plane (+z / -z)
  + / =                        : take the speed up a notch (1.25x, capped)
  -                            : take the speed down a notch (1/1.25x)
  o / c                        : open / close the gripper (non-blocking)
  space                        : freeze -- zero all motion, hold position
  q  (or Ctrl-C)               : quit -- stops the skill, restores terminal
"""

import os
import sys
import select
import termios
import tty
import time

import numpy as np
import rospy

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

# ----------------------------------------------------------------------------
# Tunable constants -- REVIEW THESE ON THE ROBOT.
# ----------------------------------------------------------------------------
CONTROL_HZ = 50                 # control / publish loop rate
DT = 1.0 / CONTROL_HZ

SPEED_XY = 0.025                # m/s, base end-effector speed in the XY plane
SPEED_Z = 0.02                  # m/s, base speed when raising / lowering

# Live speed control: `+`/`=` scale up, `-` scales down.  speed_mult starts at
# 1.0 and is clamped to [SPEED_MULT_MIN, SPEED_MULT_MAX] so the top speed stays
# safe (SPEED_XY * SPEED_MULT_MAX defines the worst-case command velocity).
SPEED_STEP = 1.25
SPEED_MULT_MIN = 0.2
SPEED_MULT_MAX = 4.0

# How long a key counts as "held" after the last keystroke.  Terminals send
# repeated characters while a key is held; this must be a bit longer than the
# key-repeat interval so motion is smooth.  It is also the max coast time after
# release (<= SPEED * KEY_TIMEOUT of overshoot).  Tune with `xset r rate`.
KEY_TIMEOUT = 0.30              # s

# Safe bounding box for the end-effector, in the 'world' frame (meters).
# The target is hard-clamped to this box.  Keep it well inside what the arm
# can reach with the gripper pointing straight down.
SAFE_BOX = {
    'x': (0.30, 0.60),
    'y': (-0.25, 0.25),
    'z': (0.05, 0.45),
}
WALL_MARGIN = 0.03              # extra margin kept inside the virtual walls

# Pose to move to (blocking) before teleop starts -- the center of the box.
START_X = 0.5 * (SAFE_BOX['x'][0] + SAFE_BOX['x'][1])
START_Y = 0.5 * (SAFE_BOX['y'][0] + SAFE_BOX['y'][1])
START_Z = 0.30

# Joint-limit guard: if any joint gets this close (rad) to its limit, the
# last commanded motion is reverted.
JOINT_LIMIT_MARGIN = 0.10

# A dynamic skill runs for a fixed duration; we transparently re-arm it before
# it expires so teleop can run indefinitely.
SKILL_SEGMENT_DURATION = 60.0   # s
SKILL_REARM_BEFORE = 5.0        # s, re-arm this long before expiry

GRIPPER_STEP_OPEN = FC.GRIPPER_WIDTH_MAX
GRIPPER_STEP_CLOSE = FC.GRIPPER_WIDTH_MIN


# ----------------------------------------------------------------------------
# Workspace bounds: intersect SAFE_BOX with the frankapy virtual walls.
# ----------------------------------------------------------------------------
def compute_bounds():
    """Return (lo, hi) xyz arrays clamped to both SAFE_BOX and the walls."""
    wall_lo = FC.WORKSPACE_WALLS[:, :3].min(axis=0) + WALL_MARGIN
    wall_hi = FC.WORKSPACE_WALLS[:, :3].max(axis=0) - WALL_MARGIN
    lo = np.array([SAFE_BOX['x'][0], SAFE_BOX['y'][0], SAFE_BOX['z'][0]])
    hi = np.array([SAFE_BOX['x'][1], SAFE_BOX['y'][1], SAFE_BOX['z'][1]])
    lo = np.maximum(lo, wall_lo)
    hi = np.minimum(hi, wall_hi)
    if np.any(lo >= hi):
        raise ValueError('SAFE_BOX is empty after intersecting virtual walls: '
                         'lo={} hi={}'.format(lo, hi))
    return lo, hi


# ----------------------------------------------------------------------------
# Non-blocking keyboard reader.
# ----------------------------------------------------------------------------
class KeyboardReader(object):
    """Puts the terminal into cbreak mode and reads keys without blocking.

    cbreak (not raw) is used so Ctrl-C still raises KeyboardInterrupt.
    """

    # Arrow-key escape sequences -> logical key names.
    ARROWS = {'A': 'up', 'B': 'down', 'C': 'right', 'D': 'left'}

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_keys(self):
        """Return the list of keys pressed since the last call (may be empty)."""
        keys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            data = os.read(self.fd, 1024).decode('utf-8', errors='ignore')
            if not data:
                break
            i = 0
            while i < len(data):
                c = data[i]
                if c == '\x1b' and data[i + 1:i + 2] == '[' \
                        and data[i + 2:i + 3] in self.ARROWS:
                    keys.append(self.ARROWS[data[i + 2]])
                    i += 3
                else:
                    keys.append(c)
                    i += 1
        return keys


# ----------------------------------------------------------------------------
# Dynamic-skill helper: publishes target poses to the impedance controller.
# ----------------------------------------------------------------------------
class DynamicPosePublisher(object):
    """Wraps a re-armable dynamic goto_pose skill."""

    def __init__(self, fa):
        self.fa = fa
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC,
                                   SensorDataGroup, queue_size=100)
        self.msg_id = 0
        self.segment_start = 0.0
        self.init_time = 0.0

    def arm(self, pose):
        """(Re)start a dynamic skill with `pose` (RigidTransform) as the seed."""
        self.fa.goto_pose(pose,
                          duration=SKILL_SEGMENT_DURATION,
                          dynamic=True,
                          buffer_time=SKILL_SEGMENT_DURATION + 10,
                          ignore_errors=False)
        self.msg_id = 1
        self.segment_start = time.time()
        self.init_time = rospy.Time.now().to_time()

    def needs_rearm(self):
        return (time.time() - self.segment_start) > \
               (SKILL_SEGMENT_DURATION - SKILL_REARM_BEFORE)

    def publish(self, translation, quaternion):
        """Send one target pose to the controller."""
        self.msg_id += 1
        timestamp = rospy.Time.now().to_time() - self.init_time
        proto_msg = PosePositionSensorMessage(
            id=self.msg_id, timestamp=timestamp,
            position=translation, quaternion=quaternion)
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                proto_msg, SensorDataMessageType.POSE_POSITION))
        self.pub.publish(ros_msg)

    def terminate(self):
        """Tell the controller to stop the skill cleanly."""
        timestamp = rospy.Time.now().to_time() - self.init_time
        term_msg = ShouldTerminateSensorMessage(timestamp=timestamp,
                                                should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_msg, SensorDataMessageType.SHOULD_TERMINATE))
        self.pub.publish(ros_msg)


# ----------------------------------------------------------------------------
# Joint-limit check.
# ----------------------------------------------------------------------------
def joint_margin(joints):
    """Return the smallest distance (rad) of any joint to its nearest limit."""
    joints = np.asarray(joints)
    lo = np.asarray(FC.JOINT_LIMITS_MIN)
    hi = np.asarray(FC.JOINT_LIMITS_MAX)
    return float(np.min(np.minimum(joints - lo, hi - joints)))


def main():
    print('Connecting to the robot...')
    fa = FrankaArm()

    lo, hi = compute_bounds()
    print('Teleop bounding box (world frame):')
    print('  x: [{:.3f}, {:.3f}]  y: [{:.3f}, {:.3f}]  z: [{:.3f}, {:.3f}]'
          .format(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))

    # Reset to the known downward-facing home configuration, then capture the
    # downward orientation we will hold for the entire session.
    print('Resetting joints (gripper opens)...')
    fa.reset_joints()
    fa.open_gripper(block=True)
    home_pose = fa.get_pose()
    down_quat = home_pose.quaternion          # [w, x, y, z], stays constant

    # Move (blocking) to a defined, in-box start pose.
    start_pose = home_pose.copy()
    start_pose.translation = np.clip([START_X, START_Y, START_Z], lo, hi)
    print('Moving to start pose {}...'.format(np.round(start_pose.translation, 3)))
    fa.goto_pose(start_pose, duration=5)

    # `target` is the pose we command; `last_safe` is the last target that the
    # joint-limit guard considered good.
    target = np.array(start_pose.translation, dtype=float)
    last_safe = target.copy()

    publisher = DynamicPosePublisher(fa)
    publisher.arm(start_pose)

    # Per-key timestamp of the most recent press; a key is "held" if pressed
    # within the last KEY_TIMEOUT seconds.
    last_press = {k: -1e9 for k in ('xp', 'xn', 'yp', 'yn', 'zp', 'zn')}
    KEYMAP = {
        'w': 'xp', 'up': 'xp', 's': 'xn', 'down': 'xn',
        'a': 'yp', 'left': 'yp', 'd': 'yn', 'right': 'yn',
        'r': 'zp', 'f': 'zn',
    }

    speed_mult = 1.0

    rate = rospy.Rate(CONTROL_HZ)
    last_hud = 0.0
    print('\nTeleop active. Controls: w/a/s/d or arrows = XY, r/f = height, '
          '+/- = speed, o/c = gripper, space = freeze, q = quit.\n')

    try:
        with KeyboardReader() as kb:
            while not rospy.is_shutdown():
                now = time.time()

                # --- read keyboard --------------------------------------
                for key in kb.get_keys():
                    if key in ('q', '\x03'):          # q or Ctrl-C
                        raise KeyboardInterrupt
                    if key in KEYMAP:
                        last_press[KEYMAP[key]] = now
                    elif key == ' ':                 # freeze
                        for k in last_press:
                            last_press[k] = -1e9
                    elif key == 'o':
                        fa.goto_gripper(GRIPPER_STEP_OPEN, block=False)
                    elif key == 'c':
                        fa.goto_gripper(GRIPPER_STEP_CLOSE, grasp=True,
                                        block=False)
                    elif key in ('+', '='):
                        speed_mult = min(speed_mult * SPEED_STEP,
                                         SPEED_MULT_MAX)
                    elif key == '-':
                        speed_mult = max(speed_mult / SPEED_STEP,
                                         SPEED_MULT_MIN)

                # --- a key is "held" if pressed recently ----------------
                def held(name):
                    return (now - last_press[name]) < KEY_TIMEOUT

                vx = SPEED_XY * speed_mult * (held('xp') - held('xn'))
                vy = SPEED_XY * speed_mult * (held('yp') - held('yn'))
                vz = SPEED_Z * speed_mult * (held('zp') - held('zn'))

                # --- integrate velocity, then clamp to the safe box -----
                candidate = target + np.array([vx, vy, vz]) * DT
                candidate = np.clip(candidate, lo, hi)

                # --- joint-limit guard ----------------------------------
                # If the arm is near a joint limit, do NOT follow the new
                # command -- revert to the last pose that was known good.
                margin = joint_margin(fa.get_joints())
                if margin < JOINT_LIMIT_MARGIN:
                    target = last_safe.copy()
                    if now - last_hud > 0.5:
                        sys.stdout.write(
                            '\r[joint limit guard] command ignored, '
                            'margin={:.3f} rad   \n'.format(margin))
                        sys.stdout.flush()
                else:
                    target = candidate
                    last_safe = target.copy()

                # --- re-arm the dynamic skill if it is about to expire --
                if publisher.needs_rearm():
                    publisher.terminate()
                    rate.sleep()
                    pose = home_pose.copy()
                    pose.translation = target
                    fa.stop_skill()
                    publisher.arm(pose)

                # --- send the target to the controller ------------------
                publisher.publish(target, down_quat)

                # --- heads-up display -----------------------------------
                if now - last_hud > 0.25:
                    sys.stdout.write(
                        '\rtarget  x={:+.3f}  y={:+.3f}  z={:+.3f}   '
                        'speed={:.2f}x   jmargin={:.2f}   '.format(
                            target[0], target[1], target[2],
                            speed_mult, margin))
                    sys.stdout.flush()
                    last_hud = now

                rate.sleep()
    except KeyboardInterrupt:
        print('\nQuit requested.')
    finally:
        print('Stopping skill...')
        try:
            publisher.terminate()
        except Exception:
            pass
        fa.stop_skill()
        print('Done. Robot is holding its last position.')


if __name__ == '__main__':
    main()
