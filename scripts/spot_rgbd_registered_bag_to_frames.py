#!/usr/bin/env python3
"""Extract RGB + REGISTERED depth + poses from the 2026-08-11 depth-walk bag.

Combines the two existing adapters:

  * ``data_tag42/spot_rgb_bag_to_frames.py`` — compressed RGB decode + the
    ``vision`` world frame + fiducial dump (this bag's conventions);
  * ``scripts/spot_rgbd_bag_to_frames.py`` — TF/quaternion helpers and the
    RGB<->depth pairing pattern;

and adds the step neither has: **depth registration**. The bag carries only
unregistered ``/depth/<cam>/image`` (240x424, its own K, frame ``<cam>``)
while RGB K is expressed in ``<cam>_fisheye``, so each depth frame is
unprojected with the depth K, moved through the static ``<cam> ->
<cam>_fisheye`` extrinsic, reprojected with the RGB K into the 480x640 RGB
grid, and z-buffered (nearest wins). Output satisfies the ``frames-json``
one-K/co-registered contract (frames_json.py): ``depth_size == rgb_size``,
single ``K`` = RGB K, uint16-mm PNGs with ``scale_to_metres = 0.001``.

    python scripts/spot_rgbd_registered_bag_to_frames.py \
        --bag-dir ~/bags/2026-08-11_depth_walk_tag42 \
        --out-dir ~/bags/farm_depth_tag42/frames --period 0.4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_rgbd = _load("_spot_rgbd", "scripts/spot_rgbd_bag_to_frames.py")
_rgb = _load("_spot_rgb_tag42", "data_tag42/spot_rgb_bag_to_frames.py")

CAMERAS = _rgbd.CAMERAS
TS = _rgbd.TS
Reader = _rgbd.Reader
interp_pose = _rgbd.interp_pose
stamp_s = _rgbd.stamp_s
compose_static = _rgbd.compose_static
decode_depth_u16 = _rgbd.decode_depth_u16
read_tf = _rgb.read_tf          # vision->body + static edges + fiducials
decode_compressed = _rgb.decode_compressed

WORLD_FRAME = "vision"


def register_depth(
    depth_u16: np.ndarray,
    K_depth: np.ndarray,
    K_rgb: np.ndarray,
    T_fisheye_from_cam: np.ndarray,
    out_hw: Tuple[int, int],
) -> np.ndarray:
    """Warp raw depth (frame <cam>, its own K) into the RGB grid (<cam>_fisheye).

    Returns uint16 mm image of shape out_hw; 0 = no data. Nearest-depth wins
    on collisions (z-buffer via lexsort trick: write farthest first).
    """
    h, w = depth_u16.shape
    v, u = np.mgrid[0:h, 0:w]
    z = depth_u16.astype(np.float32) * 0.001
    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]
    if z.size == 0:
        return np.zeros(out_hw, dtype=np.uint16)

    fx_d, fy_d = K_depth[0, 0], K_depth[1, 1]
    cx_d, cy_d = K_depth[0, 2], K_depth[1, 2]
    x = (u - cx_d) / fx_d * z
    y = (v - cy_d) / fy_d * z
    pts = np.stack([x, y, z, np.ones_like(z)], axis=0)  # 4xN, frame <cam>

    p = T_fisheye_from_cam @ pts                          # frame <cam>_fisheye
    zf = p[2]
    front = zf > 1e-3
    p, zf = p[:, front], zf[front]

    fx_r, fy_r = K_rgb[0, 0], K_rgb[1, 1]
    cx_r, cy_r = K_rgb[0, 2], K_rgb[1, 2]
    uu = np.round(fx_r * p[0] / zf + cx_r).astype(np.int32)
    vv = np.round(fy_r * p[1] / zf + cy_r).astype(np.int32)
    oh, ow = out_hw
    inb = (uu >= 0) & (uu < ow) & (vv >= 0) & (vv < oh)
    uu, vv, zf = uu[inb], vv[inb], zf[inb]

    order = np.argsort(-zf)  # farthest first; nearest overwrites last
    out = np.zeros(out_hw, dtype=np.uint16)
    out[vv[order], uu[order]] = np.clip(zf[order] * 1000.0, 1, 65535).astype(np.uint16)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bag-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--period", type=float, default=0.4)
    ap.add_argument("--match-tol", type=float, default=0.08,
                    help="max |rgb-depth| stamp diff (s); depth is 15 Hz")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--max-frames-per-cam", type=int, default=None)
    args = ap.parse_args(argv)

    bag = args.bag_dir.expanduser().resolve()
    out = args.out_dir.expanduser().resolve()
    for cam in CAMERAS:
        (out / "rgb" / cam).mkdir(parents=True, exist_ok=True)
        (out / "depth" / cam).mkdir(parents=True, exist_ok=True)

    print(f"[reg2frames] reading TF (world={WORLD_FRAME}) ...", flush=True)
    static_edges, (wb_t, wb_p, wb_q), fid_world, fid_cam = read_tf(bag)
    if not wb_t:
        raise SystemExit(f"no dynamic {WORLD_FRAME}->body transforms in {bag}")
    print(f"[reg2frames] {len(wb_t)} {WORLD_FRAME}->body samples, "
          f"{len(static_edges)} static edges", flush=True)
    T_body_cam = {cam: compose_static(static_edges, cam) for cam in CAMERAS}
    T_fisheye_from_cam = {
        cam: np.linalg.inv(static_edges[(cam, f"{cam}_fisheye")]) for cam in CAMERAS
    }

    rgb_topic = {f"/camera/{c}/compressed": c for c in CAMERAS}
    rgb_info = {f"/camera/{c}/camera_info": c for c in CAMERAS}
    d_topic = {f"/depth/{c}/image": c for c in CAMERAS}
    d_info = {f"/depth/{c}/camera_info": c for c in CAMERAS}

    K_rgb: Dict[str, np.ndarray] = {}
    K_depth: Dict[str, np.ndarray] = {}
    depth_buf: Dict[str, Deque[Tuple[float, np.ndarray]]] = {c: deque(maxlen=40) for c in CAMERAS}
    last_kept: Dict[str, float] = {c: -1e18 for c in CAMERAS}
    kept = {c: 0 for c in CAMERAS}
    dropped_no_depth = 0
    records: List[dict] = []

    picks_topics = set(rgb_topic) | set(rgb_info) | set(d_topic) | set(d_info)
    with Reader(str(bag)) as r:
        picks = [c for c in r.connections if c.topic in picks_topics]
        for conn, _bt, raw in r.messages(connections=picks):
            topic = conn.topic
            if topic in rgb_info:
                cam = rgb_info[topic]
                if cam not in K_rgb:
                    m = TS.deserialize_cdr(raw, conn.msgtype)
                    K_rgb[cam] = np.array(m.k, dtype=np.float64).reshape(3, 3)
                continue
            if topic in d_info:
                cam = d_info[topic]
                if cam not in K_depth:
                    m = TS.deserialize_cdr(raw, conn.msgtype)
                    K_depth[cam] = np.array(m.k, dtype=np.float64).reshape(3, 3)
                continue
            if topic in d_topic:
                cam = d_topic[topic]
                m = TS.deserialize_cdr(raw, conn.msgtype)
                depth_buf[cam].append((stamp_s(m.header), decode_depth_u16(m)))
                continue

            cam = rgb_topic[topic]
            if cam not in K_rgb or cam not in K_depth or not depth_buf[cam]:
                continue
            if args.max_frames_per_cam and kept[cam] >= args.max_frames_per_cam:
                continue
            m = TS.deserialize_cdr(raw, conn.msgtype)
            st = stamp_s(m.header)
            if st - last_kept[cam] < args.period:
                continue
            d_st, d_img = min(depth_buf[cam], key=lambda p: abs(p[0] - st))
            if abs(d_st - st) > args.match_tol:
                dropped_no_depth += 1
                continue
            last_kept[cam] = st
            bgr = decode_compressed(m)
            reg = register_depth(
                d_img, K_depth[cam], K_rgb[cam], T_fisheye_from_cam[cam], bgr.shape[:2]
            )
            T = interp_pose(wb_t, wb_p, wb_q, st) @ T_body_cam[cam]
            records.append({"stamp": st, "cam": cam, "T": T, "K": K_rgb[cam],
                            "rgb": bgr, "depth": reg})
            kept[cam] += 1
            if sum(kept.values()) % 200 == 0:
                print(f"[reg2frames] kept {sum(kept.values())} {dict(kept)}", flush=True)

    records.sort(key=lambda rec: rec["stamp"])
    if not records:
        raise SystemExit("no frames extracted")

    frames = []
    for i, rec in enumerate(records):
        fid = f"{i:06d}"
        cam = rec["cam"]
        cv2.imwrite(str(out / "rgb" / cam / f"{fid}.jpg"), rec["rgb"],
                    [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        cv2.imwrite(str(out / "depth" / cam / f"{fid}.png"), rec["depth"])
        h, w = rec["rgb"].shape[:2]
        frames.append({
            "camera": cam,
            "frame_id": fid,
            "rgb_path": f"rgb/{cam}/{fid}.jpg",
            "depth_path": f"depth/{cam}/{fid}.png",
            "K": rec["K"].tolist(),
            "T_world_cam": rec["T"].tolist(),
            "rgb_size": [h, w],
            "depth_size": [h, w],
            "timestamp_ns": int(round(rec["stamp"] * 1e9)),
        })

    index = {
        "schema_version": 1,
        "dataset": "spot_depth_walk_tag42",
        "scene_id": out.name,
        "global_frame": WORLD_FRAME,
        "cameras": list(CAMERAS),
        "depth_status": "registered-offline",
        "depth_encoding": {"format": "png_u16", "scale_to_metres": 0.001},
        "frames": frames,
        "notes": (
            "bag=2026-08-11_depth_walk_tag42; rgb=/camera/<cam>/compressed; "
            "depth=/depth/<cam>/image reprojected <cam> -> <cam>_fisheye into "
            "the RGB grid (z-buffered); single K = RGB K; "
            f"pose={WORLD_FRAME}->body@body->head@head->cam@cam->fisheye"
        ),
    }
    (out / "frames.json").write_text(json.dumps(index, indent=1))
    (out / "fiducial_42.json").write_text(json.dumps({
        "world_frame": WORLD_FRAME,
        "filtered_fiducial_42": fid_world,
        "raw_fiducial_42_per_camera": fid_cam,
    }, indent=1))

    print(f"[reg2frames] wrote {len(frames)} frames "
          f"({dict(kept)}), dropped_no_depth={dropped_no_depth}", flush=True)
    print(f"[reg2frames] span {records[-1]['stamp'] - records[0]['stamp']:.1f}s; "
          f"fiducial: {len(fid_world)} filtered / {len(fid_cam)} raw", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
