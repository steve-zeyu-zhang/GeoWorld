"""Timestamp-aware causal RGB clip decoding from original video files."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import torch


def _decode_times(path, timestamps, image_size=None):
    """Choose the last decoded presentation timestamp at or before each query."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Original video is missing: {path}")
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if not np.isfinite(timestamps).all() or np.any(timestamps < 0):
        raise ValueError("Video timestamps must be finite and nonnegative")
    if image_size is not None and image_size < 1:
        raise ValueError("image_size must be a positive optional spatial-size cap")
    order = np.argsort(timestamps, kind="stable")
    query = timestamps[order]
    selected = [None] * len(query)
    if not len(query):
        raise ValueError("Cannot decode zero video timestamps")
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        start_offset = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
        seek_seconds = max(0.0, float(query[0]) - 1.0) + start_offset
        if seek_seconds > 0:
            container.seek(int(seek_seconds * av.time_base), backward=True, any_frame=False)
        cursor, previous, previous_time = 0, None, None
        frame_number = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                if seek_seconds > 0:
                    raise ValueError("Video lacks presentation timestamps needed for accurate seeking")
                time = frame_number / fps
            else:
                time = float(frame.pts * frame.time_base) - start_offset
            frame_number += 1
            if previous_time is not None and time < previous_time:
                raise ValueError("Decoded video presentation timestamps are not monotonic")
            if image_size is None:
                current = frame.to_ndarray(format="rgb24")
            else:
                # Optional explicit memory cap, separate from encoder preprocessing.
                im = frame.to_image()
                im.thumbnail((image_size, image_size))
                current = np.asarray(im).copy()
            while cursor < len(query) and query[cursor] < time - 1e-9:
                selected[cursor] = previous if previous is not None else current
                cursor += 1
            while cursor < len(query) and abs(query[cursor] - time) <= 1e-9:
                selected[cursor] = current
                cursor += 1
            previous, previous_time = current, time
            if cursor == len(query):
                break
        if previous is None:
            raise ValueError(f"Video contains no decodable frames: {path}")
        # Annotation endpoints may extend a fraction of one second beyond the
        # final presentation timestamp. Larger truncation is an explicit error.
        tolerance = max(0.25, 1.0 / max(fps, 1e-6))
        if cursor < len(query) and query[-1] - previous_time > tolerance:
            raise ValueError(f"{path}: annotation extends beyond decoded video ({query[-1]:.3f}s > {previous_time:.3f}s)")
        while cursor < len(query):
            selected[cursor] = previous
            cursor += 1
    restored = [None] * len(order)
    for position, original in enumerate(order):
        restored[original] = selected[position]
    return np.stack(restored)


def load_video_clips(path, end_times, frames_per_clip=16, clip_seconds=1.0, image_size=None):
    """Return [states,3,frames,H,W] RGB floats in [0,1], with no normalization.

    Clips cover [end_time-clip_seconds,end_time]. At the beginning of a video,
    negative times repeat the first frame. Causal selection never chooses a
    future frame for the initial observation. For image planning use one frame.
    """
    if frames_per_clip < 1 or not np.isfinite(clip_seconds) or clip_seconds <= 0:
        raise ValueError("frames_per_clip and clip_seconds must be positive")
    end_times = np.asarray(end_times, dtype=np.float64)
    if end_times.ndim != 1 or not len(end_times) or not np.isfinite(end_times).all() or np.any(end_times < 0):
        raise ValueError("State endpoints must be a nonempty vector of finite nonnegative seconds")
    offsets = np.linspace(-clip_seconds, 0, frames_per_clip) if frames_per_clip > 1 else np.array([0.0])
    times = np.maximum(0.0, end_times[:, None] + offsets[None])
    decoded = _decode_times(path, times.reshape(-1), image_size=image_size)
    decoded = decoded.reshape(len(end_times), frames_per_clip, *decoded.shape[1:])
    return torch.from_numpy(decoded).permute(0, 4, 1, 2, 3).contiguous().float().div_(255.0)


def load_video_clip(path, end_time, frames_per_clip=16, clip_seconds=1.0):
    return load_video_clips(path, [end_time], frames_per_clip, clip_seconds)[0]
