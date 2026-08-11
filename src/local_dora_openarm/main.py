# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Node to control OpenArm."""

import argparse
import dataclasses
import os
import pathlib
import time

import dora
import numpy as np
import pyarrow as pa

try:
    import openarm_driver
except ImportError:
    import local_openarm_driver as openarm_driver


@dataclasses.dataclass
class AlignState:
    """State for alignment."""

    align_target: np.ndarray | None = None
    startup_target: np.ndarray | None = None
    last_update: float | None = None
    settled_since: float | None = None
    last_progress_log: float | None = None


def _step_towards(
    current_command: np.ndarray,
    target: np.ndarray,
    max_joint_speed: float,
    dt: float,
) -> np.ndarray:
    """Move arm-joint commands toward target without exceeding rad/s."""
    command = current_command.copy()
    max_step = max_joint_speed * dt
    command[:7] += np.clip(
        target[:7] - current_command[:7],
        -max_step,
        max_step,
    )
    return command


def _align(
    arm,
    state,
    new_position,
    name,
    threshold,
    max_joint_speed,
    trigger=None,
):
    """Approach the first startup target at a bounded joint speed."""
    live_target = np.asarray(new_position, dtype=np.float32)

    if trigger == "gripper":
        # v1: prismatic finger (slide), range [0, 0.044] m
        # gripping = gripper near 0 (closed position, < 5 mm)
        is_gripping = live_target[-1] < 0.005
        if not is_gripping:
            return False

    current_position = np.array(arm.fetch_position(), dtype=np.float32)

    if state.align_target is None:
        state.align_target = current_position.copy()
        state.startup_target = live_target.copy()
        state.last_update = time.monotonic()
        state.last_progress_log = state.last_update
        print(
            f"[{name}] startup alignment: approaching captured target "
            f"at {max_joint_speed:.2f} rad/s"
        )

    target = state.startup_target
    assert target is not None

    # The gripper must always follow the current trigger target. Only the
    # first seven arm joints use the captured target during startup alignment.
    state.align_target[-1] = live_target[-1]
    
    # Prefer measured motor positions when deciding that alignment is complete.
    actual_error = float(np.max(np.abs(target[:7] - current_position[:7])))
    if actual_error <= threshold:
        arm.send_position(state.align_target)
        return True

    now = time.monotonic()
    dt = min(max(now - state.last_update, 0.0), 0.1)
    state.last_update = now
    state.align_target = _step_towards(
        state.align_target,
        target,
        max_joint_speed,
        dt,
    )
    arm.send_position(state.align_target)

    command_error = float(np.max(np.abs(target[:7] - state.align_target[:7])))
    if command_error <= threshold:
        if state.settled_since is None:
            state.settled_since = now
        elif now - state.settled_since >= 0.5:
            # Some adapters report a small persistent position offset even
            # after the bounded command trajectory has reached its target.
            print(
                f"[{name}] startup command settled "
                f"(measured max error {actual_error:.3f} rad)"
            )
            return True
    else:
        state.settled_since = None

    if (
        state.last_progress_log is not None
        and now - state.last_progress_log >= 1.0
    ):
        print(
            f"[{name}] aligning: command error {command_error:.3f} rad, "
            f"measured error {actual_error:.3f} rad"
        )
        state.last_progress_log = now

    return False


def _env_flag(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def main():
    """Move to the given position and output the current position."""
    parser = argparse.ArgumentParser(description="Control OpenArm")
    parser.add_argument(
        "--side",
        choices=["right", "left"],
        default="right",
        help="right or left",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="The configuration file for this OpenArm",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--align-trigger",
        choices=["gripper"],
        default=None,
        help="Alignment trigger: gripper (default: None)",
    )
    parser.add_argument(
        "--align-threshold",
        default=0.03,
        help="Startup alignment threshold [rad] (default: 0.03)",
        type=float,
    )
    parser.add_argument(
        "--align-speed",
        default=0.3,
        help="Startup alignment joint speed [rad/s] (default: 0.3)",
        type=float,
    )
    parser.add_argument(
        "--stop",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("STOP", True),
        help="Stop the arm on exit.",
    )
    parser.add_argument(
        "--refresh-every-request",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("REFRESH", True),
        help="Refresh OpenArm on every request to make it more accurate.",
    )
    args = parser.parse_args()
    if args.align_speed <= 0:
        parser.error("--align-speed must be greater than zero")
    if args.align_threshold <= 0:
        parser.error("--align-threshold must be greater than zero")

    node = dora.Node()
    name = f"{args.side}_arm"
    config = openarm_driver.Config(args.config)
    arm = openarm_driver.SingleArmDriver(name, config)
    arm.start()
    node.send_output("status", pa.array(["aligning"]))

    initialized = False
    last_sent = None  # track last position actually sent to hardware
    init_tick = 0     # count ticks since init for gradual acceleration
    rate_limit = 0.03  # current rate limit (rad/tick), gently ramps up
    align_state = AlignState()
    for event in node:
        if event["type"] != "INPUT":
            continue

        # Main process
        event_id = event["id"]
        if event_id == "request_position":
            current_position = arm.fetch_position(
                refresh=args.refresh_every_request,
            )

            node.send_output(
                "position",
                pa.array(current_position, type=pa.float32()),
            )
        elif event_id == "request_state":
            state = arm.fetch_state(refresh=args.refresh_every_request)
            node.send_output(
                "state",
                pa.StructArray.from_arrays(
                    [
                        pa.array(state["qpos"], type=pa.float32()),
                        pa.array(state["qvel"], type=pa.float32()),
                        pa.array(state["qtorque"], type=pa.float32()),
                    ],
                    names=["qpos", "qvel", "qtorque"],
                ),
            )
        elif event_id == "move_position":
            value = event["value"]
            if isinstance(value, pa.StructArray):
                new_position = value.field("new_position")
                # TODO: We use this for safety check later.
                # other_arm_position = value.field("other_arm_position")
            else:
                new_position = value
                # other_arm_position = None
            if not initialized:
                initialized = _align(
                    arm,
                    align_state,
                    new_position,
                    name,
                    args.align_threshold,
                    args.align_speed,
                    trigger=args.align_trigger,
                )
                if initialized:
                    init_tick = 0
                    rate_limit = 0.03
                    last_sent = np.array(arm.fetch_position(), dtype=np.float32)
                    print(f"[{name}] startup alignment complete; teleoperation ready")
                    node.send_output("status", pa.array(["ready"]))
                continue

            target = np.array(new_position, dtype=np.float32)

            # If any joint jumped > 0.3 rad (likely STALE→OK or intermittent tracking glitch),
            # reset the acceleration ramp to avoid a velocity spike.
            if last_sent is not None:
                jump = np.max(np.abs(target - last_sent))
                if jump > 0.3:
                    init_tick = 0
                    rate_limit = 0.03

            # Gradually ramp rate_limit from 0.03 → 0.15 rad/tick over 40 ticks (~0.8s at 50Hz)
            # This prevents a velocity jump on startup: if your hand is far from home,
            # the arm accelerates smoothly instead of lurching.
            if init_tick < 40:
                init_tick += 1
                target_limit = 0.03 + 0.12 * (init_tick / 40)  # 0.03 → 0.15
                if target_limit > rate_limit:
                    rate_limit = target_limit
            else:
                rate_limit = 0.15  # full speed after warmup

            # Proportional rate-limit with gradual velocity cap
            if last_sent is not None:
                diff = target - last_sent
                max_step = np.max(np.abs(diff))
                if max_step > rate_limit:
                    target = last_sent + diff * (rate_limit / max_step)

            arm.send_position(target)
            last_sent = target.copy()
    if args.stop:
        arm.stop()
    else:
        arm.on_start()


if __name__ == "__main__":
    main()
