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

    def seq(self) -> int:
        with self._lock:
            return self._seq


def pack_frame(payload: bytes, w: int, h: int, stamp_ns: int) -> bytes:
    header = MAGIC + struct.pack("<IIIQ", VERSION, w, h, stamp_ns)
    return header + payload


def serve(bind: str, port: int, latest: LatestFrame, min_period_s: float) -> None:
    """Send a depth frame only when ROS stamp advances (no same-stamp keepalive).

    If the camera stalls, we intentionally send nothing rather than replaying an
    old stamp — the laptop must treat missing updates as stale. TCP keepalive
    keeps the socket alive; a log heartbeat reports idle time.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind, port))
    srv.listen(4)
    print(f"[bridge] listening on {bind}:{port} (send only on advancing stamp)", flush=True)
    while True:
        conn, addr = srv.accept()
        print(f"[bridge] client {addr}", flush=True)
        last_sent_stamp = -1
        last_send = 0.0
        last_hb = time.time()
        frames_sent = 0
        skipped_same = 0
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # Detect dead peers without replaying depth frames.
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            with conn:
                while True:
                    item = latest.get()
                    now = time.time()
                    if now - last_hb >= 2.0:
                        print(
                            f"[bridge] heartbeat sent={frames_sent} skip_same_stamp={skipped_same} "
                            f"last_stamp={last_sent_stamp} idle_s={now - last_send:.2f}",
                            flush=True,
                        )
                        last_hb = now
                    if item is None:
                        time.sleep(0.01)
                        continue
                    payload, w, h, stamp_ns, seq = item
                    if stamp_ns <= last_sent_stamp:
                        skipped_same += 1
                        time.sleep(0.005)
                        continue
                    if min_period_s > 0 and (now - last_send) < min_period_s:
                        time.sleep(0.001)
                        continue
                    last_sent_stamp = stamp_ns
                    last_send = now
                    conn.sendall(pack_frame(payload, w, h, stamp_ns))
                    frames_sent += 1
                    if frames_sent <= 3 or frames_sent % 30 == 0:
                        print(
                            f"[bridge] sent frame#{frames_sent} {w}x{h} seq={seq} stamp_ns={stamp_ns}",
                            flush=True,
                        )
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"[bridge] client gone after {frames_sent} frames: {e}", flush=True)


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
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import Image

    class DepthNode(Node):
        def __init__(self) -> None:
            super().__init__("ep_depth_tcp_bridge")
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
                durability=DurabilityPolicy.VOLATILE,
            )
            self._n = 0
            self.create_subscription(Image, args.topic, self._cb, qos)
            self.create_timer(2.0, self._heartbeat)
            self.get_logger().info(f"subscribing {args.topic} BEST_EFFORT keep_last=5")

        def _cb(self, msg: Image) -> None:
            if msg.encoding not in ("16UC1", "mono16"):
                self.get_logger().warn(f"unexpected encoding {msg.encoding}")
                return
            h, w = int(msg.height), int(msg.width)
            arr = np.frombuffer(msg.data, dtype=np.uint16)
            if arr.size < w * h:
                self.get_logger().warn(f"short image {arr.size} < {w * h}")
                return
            arr = np.asarray(arr[: w * h]).reshape(h, w).copy()
            stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(
                msg.header.stamp.nanosec
            )
            latest.set(arr, stamp_ns)
            self._n += 1
            if self._n <= 5 or self._n % 60 == 0:
                self.get_logger().info(f"depth cb #{self._n} {w}x{h}")

        def _heartbeat(self) -> None:
            self.get_logger().info(f"heartbeat cbs={self._n} seq={latest.seq()}")

    rclpy.init()
    node = DepthNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
