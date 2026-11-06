#!/usr/bin/env python3
"""Jetson ROS2 → TCP depth bridge (EPDF) for rl_real_go2.

Protocol (little-endian):
  magic "EPDF" + ver u32=1 + w u32 + h u32 + stamp_ns u64 + uint16[w*h] millimeters

Why this revision (frozen-ROS-stamp fix)
----------------------------------------
RealSense/ROS on the Jetson often delivers new Image callbacks whose
header.stamp does NOT advance (lab log: identical stamp_ns, e.g.
1793877449746194176). The freshness patch then deadlocked:

  * this bridge sent only when stamp strictly increased → after the first
    frame, NOTHING went out;
  * the laptop rejected non-increasing stamps and waited forever for a
    newer stamp before the first Forward.

This script sends a frame when EITHER:
  1) ROS header stamp increases, OR
  2) host monotonic elapsed ≥ 30 ms AND a new ROS callback arrived
     (payload may be identical — a static scene is still a live camera).

On the wire, stamp_ns is ALWAYS strictly increasing:
  stamp_ns = max(ros_stamp_ns, last_sent_stamp_ns + 1, last_sent_stamp_ns + elapsed_ns)
so the laptop sees a new stamp even when ROS stamp is frozen.

Never timer-resend the last image when ROS has gone silent (no fake
0.2 s keepalive of a stale picture). Heartbeat to stderr if no callback
for >1 s.

Do NOT open the camera. Subscribe to the existing realsense2_camera node.

Typical launch (Jetson, same DDS as the camera node):
  source /opt/ros/foxy/setup.bash
  source ~/cyclonedds_ws/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI=file:///home/unitree/cyclonedds_ws/cyclonedds.xml
  python3 -u ~/jetson_depth_tcp_bridge.py --bind 0.0.0.0 --port 9876
"""

from __future__ import print_function

import argparse
import signal
import socket
import struct
import sys
import threading
import time
import zlib

try:
    import numpy as np
except ImportError:
    np = None

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

MAGIC = b"EPDF"
PROTO_VER = 1
HDR_FMT = "<4sIIIQ"  # magic, ver, w, h, stamp_ns
HDR_SIZE = struct.calcsize(HDR_FMT)
MIN_SEND_INTERVAL_S = 0.030
HEARTBEAT_S = 1.0
TOPIC_DEFAULT = "/camera/depth/image_rect_raw"


def eprint(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def ros_stamp_ns(msg):
    stamp = msg.header.stamp
    return int(stamp.sec) * 1000000000 + int(stamp.nanosec)


def image_to_u16(msg):
    """Return (w, h, payload_bytes_le, crc32, encoding)."""
    w = int(msg.width)
    h = int(msg.height)
    enc = (msg.encoding or "").strip()
    if w <= 0 or h <= 0:
        raise ValueError("invalid image size %sx%s" % (w, h))
    n = w * h
    data = bytes(msg.data)

    if enc in ("16UC1", "mono16", "16SC1"):
        if np is not None:
            dt = ">u2" if msg.is_bigendian else "<u2"
            arr = np.frombuffer(data, dtype=dt, count=n)
            le = np.ascontiguousarray(arr, dtype="<u2")
            payload = le.tobytes()
        else:
            if msg.is_bigendian:
                # swap to little-endian uint16
                raw = bytearray(data[: n * 2])
                raw[0::2], raw[1::2] = raw[1::2], raw[0::2]
                payload = bytes(raw)
            else:
                payload = data[: n * 2]
        if len(payload) < n * 2:
            raise ValueError("short 16-bit payload %d < %d" % (len(payload), n * 2))
        payload = payload[: n * 2]
    elif enc in ("32FC1",):
        if np is None:
            raise RuntimeError("numpy required to convert 32FC1 depth")
        dt = ">f4" if msg.is_bigendian else "<f4"
        meters = np.frombuffer(data, dtype=dt, count=n)
        mm = np.clip(np.nan_to_num(meters, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0, 0, 65535)
        payload = np.ascontiguousarray(mm.astype("<u2")).tobytes()
    else:
        raise ValueError("unsupported encoding %r" % enc)

    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return w, h, payload, crc, enc


def set_tcp_opts(sock):
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except (OSError, AttributeError):
        pass
    # Linux TCP keepalive: detect a dead laptop fast.
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
    except (OSError, AttributeError):
        pass


class DepthBridge(Node):
    def __init__(self, topic):
        super().__init__("jetson_depth_tcp_bridge")
        self._lock = threading.Lock()
        self.cb_count = 0
        self.last_cb_mono = 0.0
        self.latest = None  # dict
        # Publisher is RELIABLE (realsense2_camera default). BEST_EFFORT
        # is DDS-compatible on paper but after reboot this node delivered
        # one late-join sample then went silent; RELIABLE keeps the stream.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(Image, topic, self._on_depth, qos)
        eprint(
            "[bridge] subscribe %s RELIABLE keep_last=5; "
            "send on stamp advance OR (dt>=30ms AND new callback); "
            "rewrite frozen ROS stamp to strictly increasing stamp_ns"
            % topic
        )

    def _on_depth(self, msg):
        try:
            w, h, payload, crc, enc = image_to_u16(msg)
        except Exception as ex:
            # rate-limit decode errors
            now = time.monotonic()
            last = getattr(self, "_last_err_mono", 0.0)
            if now - last >= 1.0:
                self._last_err_mono = now
                eprint("[bridge] drop frame: %s" % ex)
            return
        rec = {
            "w": w,
            "h": h,
            "payload": payload,
            "crc": crc,
            "enc": enc,
            "ros_stamp_ns": ros_stamp_ns(msg),
            "recv_mono_ns": time.monotonic_ns(),
            "cb_seq": 0,
        }
        with self._lock:
            self.cb_count += 1
            rec["cb_seq"] = self.cb_count
            self.last_cb_mono = time.monotonic()
            self.latest = rec

    def snapshot(self):
        with self._lock:
            return self.latest, self.cb_count, self.last_cb_mono


def should_send(frame, cb_count, last_sent, min_interval_s):
    """Return True iff this frame should go on the wire.

    last_sent keys: cb_seq, ros_stamp_ns, crc, mono
    """
    if frame is None:
        return False
    new_cb = frame["cb_seq"] > last_sent["cb_seq"]
    if not new_cb:
        # No new ROS callback → never keepalive-resend the old image.
        return False
    stamp_up = frame["ros_stamp_ns"] > last_sent["ros_stamp_ns"]
    now = time.monotonic()
    elapsed = now - last_sent["mono"] if last_sent["mono"] > 0.0 else 1e9
    if stamp_up:
        return True
    if elapsed >= min_interval_s:
        # Frozen ROS stamp: new callback (payload may be unchanged) is enough.
        return True
    return False


def make_wire_stamp(frame, last_sent_stamp_ns, last_sent_mono_ns):
    """Strictly increasing on-wire stamp_ns even if ROS stamp is frozen."""
    ros = int(frame["ros_stamp_ns"])
    recv_mono = int(frame["recv_mono_ns"])
    now_mono = time.monotonic_ns()
    if last_sent_stamp_ns <= 0:
        # First frame: prefer ROS stamp when it looks like a real clock,
        # otherwise host receive monotonic.
        return ros if ros > 0 else recv_mono
    elapsed = max(1, now_mono - last_sent_mono_ns) if last_sent_mono_ns > 0 else 1
    bumped = last_sent_stamp_ns + elapsed
    # Always at least last+1; never go backwards if ROS stamp is frozen or jumps back.
    return max(ros, last_sent_stamp_ns + 1, bumped)


def pack_frame(w, h, stamp_ns, payload):
    return struct.pack(HDR_FMT, MAGIC, PROTO_VER, int(w), int(h), int(stamp_ns)) + payload


def serve(node, bind, port, min_interval_s):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    set_tcp_opts(srv)
    srv.bind((bind, port))
    srv.listen(1)
    srv.settimeout(0.2)
    eprint("[bridge] listen %s:%d  header=%dB  min_interval=%.0fms" % (bind, port, HDR_SIZE, min_interval_s * 1000.0))

    client = None
    client_addr = None
    last_sent = {
        "cb_seq": 0,
        "ros_stamp_ns": -1,
        "crc": None,
        "mono": 0.0,
        "stamp_ns": 0,
        "mono_ns": 0,
    }
    sent_count = 0
    skip_no_cb = 0
    skip_interval = 0
    last_hb_mono = 0.0
    last_hb_no_cb = 0.0
    last_log_sent = 0.0
    stop = {"flag": False}

    def _stop(*_a):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        while not stop["flag"] and rclpy.ok():
            frame, cb_count, last_cb_mono = node.snapshot()
            now = time.monotonic()

            # Heartbeat if ROS callbacks stall >1 s.
            if last_cb_mono <= 0.0:
                if now - last_hb_no_cb >= HEARTBEAT_S:
                    last_hb_no_cb = now
                    eprint(
                        "[heartbeat] waiting for first depth callback  cbs=%d sent=%d"
                        % (cb_count, sent_count)
                    )
            elif (now - last_cb_mono) > HEARTBEAT_S and (now - last_hb_mono) >= HEARTBEAT_S:
                last_hb_mono = now
                eprint(
                    "[heartbeat] no depth callback for %.2fs  cbs=%d sent=%d "
                    "skip_no_cb=%d skip_interval=%d last_ros_stamp=%s last_sent_stamp=%s"
                    % (
                        now - last_cb_mono,
                        cb_count,
                        sent_count,
                        skip_no_cb,
                        skip_interval,
                        frame["ros_stamp_ns"] if frame else None,
                        last_sent["stamp_ns"],
                    )
                )

            if client is None:
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                set_tcp_opts(conn)
                conn.settimeout(2.0)
                client = conn
                client_addr = addr
                eprint("[bridge] client %s:%d connected" % (addr[0], addr[1]))
                continue

            if frame is None:
                time.sleep(0.005)
                continue

            if not should_send(frame, cb_count, last_sent, min_interval_s):
                if frame["cb_seq"] <= last_sent["cb_seq"]:
                    skip_no_cb += 1
                else:
                    skip_interval += 1
                time.sleep(0.002)
                continue

            wire_stamp = make_wire_stamp(frame, last_sent["stamp_ns"], last_sent["mono_ns"])
            frozen = frame["ros_stamp_ns"] <= last_sent["ros_stamp_ns"]
            blob = pack_frame(frame["w"], frame["h"], wire_stamp, frame["payload"])
            try:
                client.sendall(blob)
            except (OSError, socket.timeout) as ex:
                eprint("[bridge] client %s dropped (%s)" % (client_addr, ex))
                try:
                    client.close()
                except OSError:
                    pass
                client = None
                client_addr = None
                continue

            sent_count += 1
            last_sent = {
                "cb_seq": frame["cb_seq"],
                "ros_stamp_ns": frame["ros_stamp_ns"],
                "crc": frame["crc"],
                "mono": time.monotonic(),
                "stamp_ns": wire_stamp,
                "mono_ns": time.monotonic_ns(),
            }
            if sent_count <= 8 or (now - last_log_sent) >= 1.0:
                last_log_sent = now
                eprint(
                    "sent frame %dx%d stamp_ns=%d ros_stamp_ns=%d cb=%d seq_sent=%d "
                    "frozen_ros=%d crc=%08x bytes=%d"
                    % (
                        frame["w"],
                        frame["h"],
                        wire_stamp,
                        frame["ros_stamp_ns"],
                        frame["cb_seq"],
                        sent_count,
                        int(frozen),
                        frame["crc"],
                        len(blob),
                    )
                )
    finally:
        if client is not None:
            try:
                client.close()
            except OSError:
                pass
        srv.close()
        eprint("[bridge] listen socket closed  sent=%d" % sent_count)


def main():
    ap = argparse.ArgumentParser(description="Jetson ROS depth → TCP EPDF bridge")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9876)
    ap.add_argument("--topic", default=TOPIC_DEFAULT)
    ap.add_argument(
        "--min-interval-ms",
        type=float,
        default=MIN_SEND_INTERVAL_S * 1000.0,
        help="min dt between sends when ROS stamp is frozen (default 30)",
    )
    args = ap.parse_args()
    min_interval_s = max(0.005, args.min_interval_ms / 1000.0)

    rclpy.init(args=None)
    node = DepthBridge(args.topic)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, name="ros-spin", daemon=True)
    spin_thread.start()
    eprint("[bridge] MultiThreadedExecutor spinning")
    try:
        serve(node, args.bind, args.port, min_interval_s)
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        eprint("[bridge] shutdown")


if __name__ == "__main__":
    main()
