"""Frame-index compatible reader with FFmpeg pixel decoding.

Decord supplies the existing index/fps/PTS metadata only. Its random pixel
access can return a different frame while reporting the requested index.
Keep sampling policy unchanged and seek FFmpeg by presentation timestamp.
"""
from collections import OrderedDict
import json
import operator
from pathlib import Path
import shutil
import subprocess
import sys

from decord import VideoReader as MetadataReader
import numpy as np


class _Array:
    def __init__(self, value):
        self.value = value

    def asnumpy(self):
        return self.value


def _binary(name):
    bundled = Path(sys.executable).parent / name
    path = str(bundled) if bundled.is_file() else shutil.which(name)
    if not path:
        raise RuntimeError(f'{name} is required for accurate video reading')
    return path


class VideoReader:
    """The reader API used by CoSeek; no model or sampling configuration.

    Cached RGB pixels are bounded to 128 MiB. No faulty-reader pixel fallback
    is allowed: decode errors remain explicit failures of the tool request.
    """
    def __init__(self, uri, *, cache_bytes=128 * 1024 * 1024):
        self.uri = str(uri)
        self._metadata = MetadataReader(self.uri, num_threads=1)
        self._ffmpeg = _binary('ffmpeg')
        probe = subprocess.run(
            [_binary('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=width,height,start_time', '-of', 'json', self.uri],
            capture_output=True, check=True, timeout=90)
        stream = json.loads(probe.stdout)['streams'][0]
        self._shape = (int(stream['height']), int(stream['width']), 3)
        # Decord timestamps are relative to this stream's first frame.
        self._pts_offset = float(stream.get('start_time', 0))
        self._frame_bytes = int(np.prod(self._shape))
        self._cache = OrderedDict()
        self._capacity = max(0, int(cache_bytes) // self._frame_bytes)

    def __len__(self):
        return len(self._metadata)

    def get_avg_fps(self):
        return self._metadata.get_avg_fps()

    def get_frame_timestamp(self, indices):
        return self._metadata.get_frame_timestamp(indices)

    def _index(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f'Frame index {index} outside video of {len(self)} frames')
        return index

    def _read(self, index):
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        # Metadata PTS are float32. Seek halfway after the previous frame,
        # avoiding rounding just beyond the requested frame's timestamp.
        # This uses real PTS rather than index / average_fps (also for VFR).
        pts = np.asarray(self.get_frame_timestamp([max(0, index - 1), index]),
                         dtype=np.float64)[:, 0]
        if not np.isfinite(pts).all() or (index and pts[1] <= pts[0]):
            raise ValueError(f'Invalid presentation timestamps for frame {index}')
        command = [self._ffmpeg, '-nostdin', '-v', 'error', '-threads', '1']
        if index:
            command += ['-seek_timestamp', '1', '-ss',
                        f'{self._pts_offset + float(pts.mean()):.9f}']
        command += ['-noautorotate', '-i', self.uri, '-map', '0:v:0',
                    '-frames:v', '1', '-an', '-sn', '-dn', '-pix_fmt', 'rgb24',
                    '-threads', '1', '-fps_mode', 'passthrough', '-f', 'rawvideo', 'pipe:1']
        result = subprocess.run(command, capture_output=True, check=True, timeout=90)
        if len(result.stdout) != self._frame_bytes:
            raise RuntimeError(f'Incomplete decoded frame {index}: '
                               f'{len(result.stdout)} / {self._frame_bytes} bytes')
        pixels = np.frombuffer(result.stdout, dtype=np.uint8).reshape(self._shape)
        if self._capacity:
            self._cache[index] = pixels
            while len(self._cache) > self._capacity:
                self._cache.popitem(last=False)
        return pixels

    def get_batch(self, indices):
        indices = [self._index(i) for i in indices]
        if not indices:
            return _Array(np.empty((0, *self._shape), dtype=np.uint8))
        # Preserve duplicates and caller order, including nonmonotonic batches.
        frames = {i: self._read(i) for i in dict.fromkeys(indices)}
        return _Array(np.stack([frames[i] for i in indices]))

    def __getitem__(self, index):
        return _Array(self._read(self._index(index)).copy())
