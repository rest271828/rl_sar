#!/usr/bin/env python3
"""Jetson-side depth TCP bridge for rl_real_go2 (do NOT reopen RealSense).

Subscribes to the existing realsense2_camera topic
  /camera/depth/image_rect_raw  (16UC1, millimeters, typically 848x480)
and serves raw uint16 frames to the laptop over TCP.

Must run on Jetson with the SAME CycloneDDS stack as the camera node:
  source /opt/ros/foxy/setup.bash
  source ~/cyclonedds_ws/install/setup.bash   # if present
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI=file:///home/unitree/cyclonedds_ws/cyclonedds.xml

Example:
  python3 jetson_depth_tcp_bridge.py --bind 0.0.0.0 --port 9876

Laptop:
  export RL_SAR_EP_DEPTH_HOST=192.168.123.18
  export RL_SAR_EP_DEPTH_PORT=9876
"""

from __future__ import annotations

import argparse
import socket
import struct
import threading
import time
from typing import Optional, Tuple

import numpy as np


MAGIC = b"EPDF"
VERSION = 1


class LatestFrame:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buf: Optional[bytes] = None
        self._wh: Tuple[int, int] = (0, 0)
        self._stamp_ns: int = 0
        self._seq: int = 0

    def set(self, arr: np.ndarray, stamp_ns: int) -> None:
        if arr.dtype != np.uint16:
            arr = arr.astype(np.uint16, copy=False)
        payload = arr.tobytes(order="C")
        with self._lock:
            self._buf = payload
            self._wh = (int(arr.shape[1]), int(arr.shape[0]))  # w, h
            self._stamp_ns = int(stamp_ns)
            self._seq += 1

    def get(self) -> Optional[Tuple[bytes, int, int, int, int]]:
        with self._lock:
            if self._buf is None:
                return None
            w, h = self._wh
            return self._buf, w, h, self._stamp_ns, self._seq


def pack_frame(payload: bytes, w: int, h: int, stamp_ns: int) -> bytes:
    header = MAGIC + struct.pack("<IIIQ", VERSION, w, h, stamp_ns)
    return header + payload


def serve(bind: str, port: int, latest: LatestFrame, min_period_s: float) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind, port))
    srv.listen(1)
    print(f"[bridge] listening on {bind}:{port}", flush=True)
    while True:
        conn, addr = srv.accept()
        print(f"[bridge] client {addr}", flush=True)
        last_seq = -1
        try:
            with conn:
                while True:
                    item = latest.get()
                    if item is None:
                        time.sleep(0.01)
                        continue
                    payload, w, h, stamp_ns, seq = item
                    if seq == last_seq:
                        time.sleep(0.005)
                        continue
                    last_seq = seq
                    conn.sendall(pack_frame(payload, w, h, stamp_ns))
                    if min_period_s > 0:
                        time.sleep(min_period_s)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"[bridge] client gone: {e}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9876)
    ap.add_argument("--topic", default="/camera/depth/image_rect_raw")
    ap.add_argument("--hz", type=float, default=30.0, help="max send rate to laptop")
    args = ap.parse_args()

    latest = LatestFrame()
    min_period = 1.0 / args.hz if args.hz > 0 else 0.0

    thr = threading.Thread(
        target=serve, args=(args.bind, args.port, latest, min_period), daemon=True
    )
    thr.start()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    class DepthNode(Node):
        def __init__(self) -> None:
            super().__init__("ep_depth_tcp_bridge")
            self.create_subscription(
                Image, args.topic, self._cb, qos_profile_sensor_data
            )
            self.get_logger().info(f"subscribing {args.topic}")

        def _cb(self, msg: Image) -> None:
            if msg.encoding not in ("16UC1", "mono16"):
                self.get_logger().warn(f"unexpected encoding {msg.encoding}")
                return
            h, w = int(msg.height), int(msg.width)
            arr = np.frombuffer(msg.data, dtype=np.uint16)
            if arr.size < w * h:
                self.get_logger().warn(f"short image {arr.size} < {w * h}")
                return
            arr = np.asarray(arr[: w * h]).reshape(h, w)
            stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(
                msg.header.stamp.nanosec
            )
            latest.set(arr, stamp_ns)

    rclpy.init()
    node = DepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
