#!/usr/bin/env python
"""
Probe the teleop control limits.

Re-uses SAFE_BOX, speeds, and the safety guards from keyboard_teleop.py so the
boundary it shows is exactly the boundary teleop enforces.  Nothing here moves
faster or further than keyboard_teleop does.

What it does
------------
- Resets joints, opens gripper, moves to the XY-box center at START_Z
  (end-effector pointing straight down).
- On SPACE  : traces the perimeter of the XY box at the *current* height --
              current_pos -> (xmin,ymin) -> (xmax,ymin) -> (xmax,ymax)
              -> (xmin,ymax) -> (xmin,ymin) -> center, at SPEED_XY * speed_mult.
              Lets you see (and feel) exactly where the boundary is.
- r / f     : raise / lower the height while idle (velocity-limited).
- + / =     : take the speed up a notch (1.25x, capped).
- -         : take the speed down a notch (1/1.25x).
- x         : abort an in-progress trace and return to center.
- q (Ctrl-C): quit cleanly.

Safety
------
Same three layers as teleop:
  1. Velocity-limited (SPEED_XY, SPEED_Z) -- never a position jump.
  2. Hard clamp to SAFE_BOX (intersected with the virtual walls).
  3. Joint-limit guard: if any joint comes within JOINT_LIMIT_MARGIN of its
     limit the command is *not followed*.  If a corner can't be reached, the
     trace skips it and reports it -- nothing is forced into a limit.

A live ASCII top-down map shows where the EE is inside the box.
"""

import os
import sys
import time

import numpy as np
import rospy

# Reuse everything we already wrote so the two scripts can't drift apart.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keyboard_teleop import (  # noqa: E402
    SAFE_BOX, SPEED_XY, SPEED_Z, CONTROL_HZ, DT,
    JOINT_LIMIT_MARGIN,
    SPEED_STEP, SPEED_MULT_MIN, SPEED_MULT_MAX,
    START_X, START_Y, START_Z,
    compute_bounds, joint_margin,
    KeyboardReader, DynamicPosePublisher,
)
from frankapy import FrankaArm  # noqa: E402

# How long the joint guard may block progress toward a waypoint before we give
# up on it and move on to the next one.
STUCK_TIMEOUT = 0.5             # seconds
WAYPOINT_REACHED_TOL = 1e-3     # meters (1 mm)

HUD_HZ = 4                      # ASCII map refresh rate
MAP_W, MAP_H = 41, 13           # ASCII map size in characters


# ----------------------------------------------------------------------------
# Trace path.
# ----------------------------------------------------------------------------
def build_waypoints(start_xyz, lo, hi):
    """Perimeter path at the current Z: 4 corners + closure + return to center."""
    z = start_xyz[2]
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    return [
        np.array([lo[0], lo[1], z]),    # back-right
        np.array([hi[0], lo[1], z]),    # front-right
        np.array([hi[0], hi[1], z]),    # front-left
        np.array([lo[0], hi[1], z]),    # back-left
        np.array([lo[0], lo[1], z]),    # close the rectangle
        np.array([cx,    cy,    z]),    # back to center
    ]


def step_toward(target, waypoint, speed, dt):
    """Move `target` toward `waypoint` by at most speed*dt. Returns (new, reached)."""
    delta = waypoint - target
    dist = float(np.linalg.norm(delta))
    if dist <= WAYPOINT_REACHED_TOL:
        return waypoint.copy(), True
    step = speed * dt
    if step >= dist:
        return waypoint.copy(), True
    return target + delta * (step / dist), False


# ----------------------------------------------------------------------------
# ASCII top-down map.
# ----------------------------------------------------------------------------
def render_map(target, lo, hi):
    """Top-down ASCII view of the XY box.

    Convention (bird's eye, looking down at the table):
      - rows  go top->bottom = +x (front, away from base) -> -x (back, base side)
      - cols  go left->right = +y (robot's left)          -> -y (robot's right)
    """
    grid = [[' '] * MAP_W for _ in range(MAP_H)]
    # border
    for c in range(MAP_W):
        grid[0][c] = grid[MAP_H - 1][c] = '-'
    for r in range(MAP_H):
        grid[r][0] = grid[r][MAP_W - 1] = '|'
    for (r, c) in [(0, 0), (0, MAP_W - 1),
                   (MAP_H - 1, 0), (MAP_H - 1, MAP_W - 1)]:
        grid[r][c] = '+'
    # center crosshair
    cr, cc = MAP_H // 2, MAP_W // 2
    grid[cr][cc] = '+'
    # end-effector position
    fx = (hi[0] - target[0]) / max(hi[0] - lo[0], 1e-6)  # 0 at front, 1 at back
    fy = (hi[1] - target[1]) / max(hi[1] - lo[1], 1e-6)  # 0 at +y (left), 1 at -y
    r = int(round(fx * (MAP_H - 1)))
    c = int(round(fy * (MAP_W - 1)))
    r = max(0, min(MAP_H - 1, r))
    c = max(0, min(MAP_W - 1, c))
    grid[r][c] = 'X'
    return [''.join(row) for row in grid]


def render(target, lo, hi, mode, wp_idx, n_waypoints, jmargin, speed_mult, notes):
    """Compose and print the screen (cursor-home + clear-to-eol per line)."""
    lines = []
    lines.append('Probe teleop limits  '
                 '(SPACE=trace,  r/f=height,  +/-=speed,  x=abort,  q=quit)')
    lines.append('-' * 70)
    lines.append('mode: {:<8s}   target  x={:+.3f}  y={:+.3f}  z={:+.3f}   '
                 'speed={:.2f}x'
                 .format(mode, target[0], target[1], target[2], speed_mult))
    if mode == 'TRACING':
        lines.append('waypoint {}/{}    joint margin={:.2f} rad'
                     .format(wp_idx + 1, n_waypoints, jmargin))
    else:
        lines.append('                  joint margin={:.2f} rad'.format(jmargin))
    lines.append('')
    lines.append('top-down view (X = end-effector,  + = corners/center)')
    lines.append('+x = forward (away from base)   +y = left')
    lines.extend(render_map(target, lo, hi))
    lines.append('box  x:[{:+.3f},{:+.3f}]  y:[{:+.3f},{:+.3f}]  z:[{:+.3f},{:+.3f}]'
                 .format(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    lines.append('')
    lines.append('recent events:')
    for line in (notes[-4:] if notes else ['(none)']):
        lines.append('  ' + line)

    out = '\033[H'  # cursor home
    for ln in lines:
        out += ln + '\033[K\n'
    out += '\033[J'  # clear to end of screen
    sys.stdout.write(out)
    sys.stdout.flush()


def main():
    print('Connecting to the robot...')
    fa = FrankaArm()
    lo, hi = compute_bounds()

    print('Resetting joints (gripper opens)...')
    fa.reset_joints()
    fa.open_gripper(block=True)
    home_pose = fa.get_pose()
    down_quat = home_pose.quaternion

    start_pose = home_pose.copy()
    start_pose.translation = np.clip([START_X, START_Y, START_Z], lo, hi)
    print('Moving to start pose {}...'.format(np.round(start_pose.translation, 3)))
    fa.goto_pose(start_pose, duration=5)

    target = np.array(start_pose.translation, dtype=float)
    last_safe = target.copy()

    publisher = DynamicPosePublisher(fa)
    publisher.arm(start_pose)

    last_press = {'zp': -1e9, 'zn': -1e9}
    KEYMAP_Z = {'r': 'zp', 'f': 'zn'}

    mode = 'IDLE'
    waypoints = []
    wp_idx = 0
    stuck_since = None
    speed_mult = 1.0
    notes = ['ready -- press SPACE to trace the box edges at the current height']

    rate = rospy.Rate(CONTROL_HZ)
    sys.stdout.write('\033[2J\033[H')  # clear screen once
    last_hud = 0.0

    try:
        with KeyboardReader() as kb:
            while not rospy.is_shutdown():
                now = time.time()
                start_trace = False
                abort_trace = False

                # --- read keyboard ---------------------------------------
                for key in kb.get_keys():
                    if key in ('q', '\x03'):
                        raise KeyboardInterrupt
                    if key == ' ':
                        start_trace = True
                    elif key == 'x':
                        abort_trace = True
                    elif key in KEYMAP_Z:
                        last_press[KEYMAP_Z[key]] = now
                    elif key in ('+', '='):
                        speed_mult = min(speed_mult * SPEED_STEP,
                                         SPEED_MULT_MAX)
                        notes.append('speed -> {:.2f}x'.format(speed_mult))
                    elif key == '-':
                        speed_mult = max(speed_mult / SPEED_STEP,
                                         SPEED_MULT_MIN)
                        notes.append('speed -> {:.2f}x'.format(speed_mult))

                # --- mode transitions ------------------------------------
                if mode == 'IDLE' and start_trace:
                    waypoints = build_waypoints(target, lo, hi)
                    wp_idx = 0
                    stuck_since = None
                    mode = 'TRACING'
                    notes.append('trace started at z={:+.3f}'.format(target[2]))
                elif mode == 'TRACING' and abort_trace:
                    mode = 'IDLE'
                    notes.append('trace aborted by user')

                # --- compute new target ----------------------------------
                if mode == 'TRACING':
                    wp = waypoints[wp_idx]
                    new_target, reached = step_toward(
                        target, wp, SPEED_XY * speed_mult, DT)
                else:  # IDLE -- r/f raise/lower height
                    held_zp = (now - last_press['zp']) < 0.30
                    held_zn = (now - last_press['zn']) < 0.30
                    vz = SPEED_Z * speed_mult * (int(held_zp) - int(held_zn))
                    new_target = target + np.array([0.0, 0.0, vz]) * DT
                    reached = False

                new_target = np.clip(new_target, lo, hi)

                # --- joint-limit guard -----------------------------------
                jmargin = joint_margin(fa.get_joints())
                if jmargin < JOINT_LIMIT_MARGIN:
                    # Revert: do not follow the command.
                    target = last_safe.copy()
                    if mode == 'TRACING':
                        if stuck_since is None:
                            stuck_since = now
                        elif (now - stuck_since) > STUCK_TIMEOUT:
                            notes.append(
                                'waypoint {} unreachable (joint guard) at '
                                'xyz=({:+.3f},{:+.3f},{:+.3f}); skipping'
                                .format(wp_idx + 1, *waypoints[wp_idx]))
                            wp_idx += 1
                            stuck_since = None
                            if wp_idx >= len(waypoints):
                                mode = 'IDLE'
                                notes.append('trace finished (some skips)')
                else:
                    target = new_target
                    last_safe = target.copy()
                    stuck_since = None
                    if mode == 'TRACING' and reached:
                        wp_idx += 1
                        if wp_idx >= len(waypoints):
                            mode = 'IDLE'
                            notes.append('trace complete')

                # --- re-arm if the dynamic skill is about to expire ------
                if publisher.needs_rearm():
                    publisher.terminate()
                    rate.sleep()
                    pose = home_pose.copy()
                    pose.translation = target
                    fa.stop_skill()
                    publisher.arm(pose)

                publisher.publish(target, down_quat)

                # --- HUD -------------------------------------------------
                if (now - last_hud) > (1.0 / HUD_HZ):
                    render(target, lo, hi, mode, wp_idx,
                           len(waypoints), jmargin, speed_mult, notes)
                    last_hud = now

                rate.sleep()
    except KeyboardInterrupt:
        sys.stdout.write('\n')
        print('Quit requested.')
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
