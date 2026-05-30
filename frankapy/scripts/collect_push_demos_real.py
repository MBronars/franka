#!/usr/bin/env python
"""
Real-world scripted push policy on the Franka.

Holds the end-effector pointing straight down at a fixed pushing plane,
then runs a simple state machine:
   1) pick a face of the cube,
   2) move to a standoff point behind that face,
   3) push along the face normal for PUSH_DISTANCE,
   4) pick a new face, repeat.

Cube state is supplied by an external tracker via the `CubeStateProvider`
interface -- swap in your CoTracker pipeline by subclassing it.

Standalone: depends only on numpy, frankapy, rospy, and stdlib.  Nothing
imported from the sim repo or from other scripts.

Controls
--------
  space  : start the next episode (after you've reset the cube)
  s      : end the current episode + save it
  p      : pause / resume
  q      : quit cleanly
"""

import argparse
import os
import select
import sys
import termios
import time
import tty
from datetime import datetime

import numpy as np
import rospy

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup


# ============================================================================
# Tunable constants -- review and tune for your cell.
# ============================================================================
CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ

MAX_SPEED = 0.025                 # m/s, EE speed cap (safety)
PUSH_Z_DEFAULT = 0.08             # m, height of the pushing plane

# Safe XY/Z bounding box (same convention as keyboard_teleop.py).  Tune!
SAFE_BOX = {
    'x': (0.30, 0.60),
    'y': (-0.25, 0.25),
    'z': (0.05, 0.45),
}
WALL_MARGIN = 0.03
JOINT_LIMIT_MARGIN = 0.10         # rad

# Cube + push policy.
CUBE_HALF_EXTENT = 0.025          # m, half side length (5 cm cube)
STANDOFF_DIST = 0.05              # m, standoff distance behind the chosen face
ARRIVE_THRESH = 0.01              # m, "reached the standoff" tolerance
PUSH_DISTANCE = 0.08              # m, total push along the face normal

# Dynamic skill timing.
SKILL_SEGMENT_DURATION = 60.0
SKILL_REARM_BEFORE = 5.0


# ============================================================================
# Workspace + joint guard helpers.
# ============================================================================
def compute_bounds():
    wall_lo = FC.WORKSPACE_WALLS[:, :3].min(axis=0) + WALL_MARGIN
    wall_hi = FC.WORKSPACE_WALLS[:, :3].max(axis=0) - WALL_MARGIN
    lo = np.array([SAFE_BOX['x'][0], SAFE_BOX['y'][0], SAFE_BOX['z'][0]])
    hi = np.array([SAFE_BOX['x'][1], SAFE_BOX['y'][1], SAFE_BOX['z'][1]])
    lo = np.maximum(lo, wall_lo)
    hi = np.minimum(hi, wall_hi)
    if np.any(lo >= hi):
        raise ValueError('SAFE_BOX is empty after intersecting walls')
    return lo, hi


def joint_margin(joints):
    j = np.asarray(joints)
    return float(np.min(np.minimum(
        j - np.asarray(FC.JOINT_LIMITS_MIN),
        np.asarray(FC.JOINT_LIMITS_MAX) - j,
    )))


# ============================================================================
# Keyboard reader (cbreak mode; Ctrl-C still works).
# ============================================================================
class KeyboardReader(object):
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_keys(self):
        keys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            data = os.read(self.fd, 1024).decode('utf-8', errors='ignore')
            if not data:
                break
            for c in data:
                keys.append(c)
        return keys


# ============================================================================
# Re-armable dynamic pose publisher.
# ============================================================================
class DynamicPosePublisher(object):
    def __init__(self, fa):
        self.fa = fa
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC,
                                   SensorDataGroup, queue_size=100)
        self.msg_id = 0
        self.segment_start = 0.0
        self.init_time = 0.0

    def arm(self, pose):
        self.fa.goto_pose(pose, duration=SKILL_SEGMENT_DURATION,
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
        self.msg_id += 1
        timestamp = rospy.Time.now().to_time() - self.init_time
        m = PosePositionSensorMessage(
            id=self.msg_id, timestamp=timestamp,
            position=translation, quaternion=quaternion)
        self.pub.publish(make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                m, SensorDataMessageType.POSE_POSITION)))

    def terminate(self):
        timestamp = rospy.Time.now().to_time() - self.init_time
        t = ShouldTerminateSensorMessage(timestamp=timestamp,
                                         should_terminate=True)
        self.pub.publish(make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                t, SensorDataMessageType.SHOULD_TERMINATE)))


# ============================================================================
# Cube tracker -- plug your real tracker here.
# ============================================================================
class CubeStateProvider(object):
    """Override get_state() to return (xy_in_robot_base, yaw_rad)."""

    def get_state(self):
        raise NotImplementedError


class MockStationaryTracker(CubeStateProvider):
    """Returns a fixed pose -- lets you dry-run the script without a tracker."""

    def __init__(self, xy, yaw=0.0):
        self.xy = np.asarray(xy, dtype=np.float64)
        self.yaw = float(yaw)

    def get_state(self):
        return self.xy.copy(), self.yaw


# TODO: implement your real tracker:
#
#   class CoTrackerCubeProvider(CubeStateProvider):
#       def __init__(self, ...):
#           # start the CoTracker thread / open the topic / etc.
#           ...
#       def get_state(self):
#           # return (xy_in_robot_base_meters, yaw_radians)
#           ...
#
# Then in main(), replace
#       tracker = MockStationaryTracker(...)
# with
#       tracker = CoTrackerCubeProvider(...)


# ============================================================================
# Simple push policy: pick a face, approach standoff, push, repeat.
# ============================================================================
def face_normal(face_idx, yaw):
    """Outward face normal in world frame, for cube rotated by yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    return [np.array([c, s]),       # +x in cube frame
            np.array([-c, -s]),     # -x
            np.array([-s, c]),      # +y
            np.array([s, -c])][face_idx]  # -y


class SimplePushPolicy:

    def __init__(self, rng):
        self.rng = rng
        self.phase = 'approach'         # 'approach' or 'push'
        self.face = 0
        self.push_start = np.zeros(2)
        self.push_dir = np.zeros(2)
        self.push_traveled = 0.0

    def reset(self, cube_xy, cube_yaw):
        self.face = int(self.rng.integers(4))
        self.phase = 'approach'
        self.push_traveled = 0.0

    def _standoff(self, cube_xy, cube_yaw):
        normal = face_normal(self.face, cube_yaw)
        contact = cube_xy - normal * CUBE_HALF_EXTENT
        return contact - normal * STANDOFF_DIST

    def next_target(self, ee_xy, cube_xy, cube_yaw):
        """One control-tick step.  Returns the desired XY target this tick."""
        # 1) phase transitions
        if self.phase == 'push' and self.push_traveled >= PUSH_DISTANCE:
            self.face = int(self.rng.integers(4))
            self.phase = 'approach'
            self.push_traveled = 0.0

        if self.phase == 'approach':
            standoff = self._standoff(cube_xy, cube_yaw)
            if np.linalg.norm(ee_xy - standoff) < ARRIVE_THRESH:
                self.phase = 'push'
                self.push_start = ee_xy.copy()
                self.push_dir = face_normal(self.face, cube_yaw)
                self.push_traveled = 0.0

        # 2) produce a step toward the right thing for the current phase
        if self.phase == 'approach':
            standoff = self._standoff(cube_xy, cube_yaw)
            delta = standoff - ee_xy
            dist = float(np.linalg.norm(delta))
            step = min(dist, MAX_SPEED * DT)
            if dist < 1e-9:
                return ee_xy.copy()
            return ee_xy + delta / dist * step
        # push: walk forward along push_dir
        step_vec = self.push_dir * (MAX_SPEED * DT)
        self.push_traveled += float(np.linalg.norm(step_vec))
        return ee_xy + step_vec


# ============================================================================
# Episode saving.
# ============================================================================
def _save_episode(record_dir, ep_num, traj, end_reason):
    if not traj:
        return None
    os.makedirs(record_dir, exist_ok=True)
    out = os.path.join(
        record_dir,
        'episode_{:04d}_{}.npz'.format(
            ep_num, datetime.now().strftime('%Y%m%d_%H%M%S')))
    np.savez(out,
             t=np.array([s['t'] for s in traj]),
             ee_xy=np.array([s['ee_xy'] for s in traj]),
             cube_xy=np.array([s['cube_xy'] for s in traj]),
             cube_yaw=np.array([s['cube_yaw'] for s in traj]),
             phase=np.array([s['phase'] for s in traj]),
             face=np.array([s['face'] for s in traj]),
             end_reason=end_reason)
    print('  saved {}  ({} ticks, end={})'.format(out, len(traj), end_reason))
    return out


# ============================================================================
# Main.
# ============================================================================
def main():
    args = parse_args()

    print('Connecting to the robot...')
    fa = FrankaArm()
    lo, hi = compute_bounds()
    push_z = float(np.clip(args.push_z, lo[2], hi[2]))
    if push_z != args.push_z:
        print('  ! push_z clamped from {:.3f} to {:.3f} by SAFE_BOX z bounds'
              .format(args.push_z, push_z))
    lo2d, hi2d = lo[:2], hi[:2]
    print('Workspace XY: x={}, y={};  push_z={:.3f}'
          .format(tuple(np.round([lo2d[0], hi2d[0]], 3)),
                  tuple(np.round([lo2d[1], hi2d[1]], 3)), push_z))

    box_center = np.array([0.5 * (lo2d[0] + hi2d[0]),
                           0.5 * (lo2d[1] + hi2d[1])])
    tracker = MockStationaryTracker(xy=box_center, yaw=0.0)
    print('Tracker: {}   (REPLACE WITH YOUR REAL TRACKER)'
          .format(type(tracker).__name__))

    print('Resetting joints + closing gripper (closed gripper = push tip)...')
    fa.reset_joints()
    fa.close_gripper()
    home_pose = fa.get_pose()
    down_quat = home_pose.quaternion

    start_pose = home_pose.copy()
    start_pose.translation = np.array([box_center[0], box_center[1], push_z])
    print('Moving to start pose {}...'.format(np.round(start_pose.translation, 3)))
    fa.goto_pose(start_pose, duration=5)

    target = np.array(start_pose.translation, dtype=float)
    last_safe = target.copy()

    publisher = DynamicPosePublisher(fa)
    publisher.arm(start_pose)

    rng = np.random.default_rng(args.seed)
    policy = SimplePushPolicy(rng)

    rate = rospy.Rate(CONTROL_HZ)
    episode = 0
    in_episode = False
    paused = False
    episode_start_time = 0.0
    traj = []

    print('\nReady.  Place the cube, then press SPACE.')
    print('Controls: SPACE=start episode, s=stop+save, p=pause, q=quit')

    try:
        with KeyboardReader() as kb:
            while not rospy.is_shutdown() and episode < args.num_episodes:
                now = time.time()

                # --- keyboard ---------------------------------------------
                for key in kb.get_keys():
                    if key in ('q', '\x03'):
                        raise KeyboardInterrupt
                    if key == ' ' and not in_episode:
                        cube_xy, cube_yaw = tracker.get_state()
                        policy.reset(cube_xy, cube_yaw)
                        episode += 1
                        in_episode = True
                        traj = []
                        episode_start_time = now
                        print('\n=== Episode {} ==='.format(episode))
                    elif key == 's' and in_episode:
                        _save_episode(args.record_dir, episode, traj, 'manual_stop')
                        in_episode = False
                        print('Stopped.  Reset cube; SPACE for next.')
                    elif key == 'p':
                        paused = not paused
                        print('PAUSED' if paused else 'RESUMED')

                # --- one control tick -------------------------------------
                if in_episode and not paused:
                    ee_pose = fa.get_pose()
                    ee_xy = np.array(ee_pose.translation[:2])
                    cube_xy, cube_yaw = tracker.get_state()

                    cube_oob = (cube_xy[0] < lo2d[0] - 0.05
                                or cube_xy[0] > hi2d[0] + 0.05
                                or cube_xy[1] < lo2d[1] - 0.05
                                or cube_xy[1] > hi2d[1] + 0.05)
                    timed_out = (now - episode_start_time) > args.max_episode_seconds

                    if cube_oob:
                        _save_episode(args.record_dir, episode, traj, 'cube_oob')
                        in_episode = False
                        print('Cube out of workspace.  Reset; SPACE for next.')
                    elif timed_out:
                        _save_episode(args.record_dir, episode, traj, 'timeout')
                        in_episode = False
                        print('Timeout.  Reset; SPACE for next.')
                    else:
                        new_target_xy = policy.next_target(ee_xy, cube_xy, cube_yaw)
                        new_target_xy = np.clip(new_target_xy, lo2d, hi2d)
                        new_target = np.array([new_target_xy[0],
                                               new_target_xy[1], push_z])

                        margin = joint_margin(fa.get_joints())
                        if margin < JOINT_LIMIT_MARGIN:
                            # Joint guard: revert; don't follow the command.
                            target = last_safe.copy()
                        else:
                            target = new_target
                            last_safe = target.copy()

                        traj.append({
                            't': now - episode_start_time,
                            'ee_xy': ee_xy.copy(),
                            'cube_xy': cube_xy.copy(),
                            'cube_yaw': float(cube_yaw),
                            'phase': policy.phase,
                            'face': int(policy.face),
                        })

                # --- re-arm if expiring -----------------------------------
                if publisher.needs_rearm():
                    publisher.terminate()
                    rate.sleep()
                    pose = home_pose.copy()
                    pose.translation = target
                    fa.stop_skill()
                    publisher.arm(pose)

                publisher.publish(target, down_quat)
                rate.sleep()

    except KeyboardInterrupt:
        print('\nQuit.')
        if in_episode:
            _save_episode(args.record_dir, episode, traj, 'user_quit')
    finally:
        print('Stopping skill...')
        try:
            publisher.terminate()
        except Exception:
            pass
        fa.stop_skill()
        print('Done.  Robot holding last position.')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--num_episodes', type=int, default=5)
    p.add_argument('--max_episode_seconds', type=float, default=60.0)
    p.add_argument('--record_dir', type=str, default='real_demos/')
    p.add_argument('--push_z', type=float, default=PUSH_Z_DEFAULT,
                   help='Height of the pushing plane (m, robot base frame). '
                        'Tune by teleop-ing the EE to the desired height '
                        'and reading fa.get_pose().translation[2].')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


if __name__ == '__main__':
    main()
