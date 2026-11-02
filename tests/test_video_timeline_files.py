"""Real codec/file regressions; run with the release's av==13.0.0."""

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest

import av
import numpy as np

from hmr4d.utils.video_io_utils import probe_video_timeline, resample_video_to_fps


def make_video(path, pts, durations, rate=25, b_frames=0, audio_seconds=0):
    time_base = Fraction(1, 6000)
    ticks = [round(float(t) / time_base) for t in pts]
    packet_durations = {t: round(float(d) / time_base) for t, d in zip(ticks, durations)}
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=rate)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = time_base
        stream.options = {"bf": str(b_frames), "crf": "18" if b_frames else "0"}
        if b_frames:
            stream.options["x264-params"] = "b-adapt=0:scenecut=0:keyint=250"
        audio = output.add_stream("aac", rate=48000) if audio_seconds else None
        packets = []
        for i, timestamp in enumerate(ticks):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 30 + 3 * i, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = timestamp, time_base
            packets.extend(stream.encode(frame))
        packets.extend(stream.encode())
        for packet in packets:
            packet.duration = packet_durations[packet.pts]
            output.mux(packet)
        if audio:
            samples = np.zeros((1, int(audio_seconds * 48000)), dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
            frame.sample_rate, frame.pts, frame.time_base = 48000, 0, Fraction(1, 48000)
            for packet in list(audio.encode(frame)) + list(audio.encode()):
                output.mux(packet)


def read_frames(path):
    with av.open(str(path)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


class RealVideoTimelineTests(unittest.TestCase):
    def check_resample(self, source, expected_duration, expected_indices=None):
        timeline = probe_video_timeline(source)
        self.assertAlmostEqual(timeline["duration"], expected_duration, places=5)
        dest = source.with_name("output.mp4")
        metadata = resample_video_to_fps(source, dest, crf=0)
        frames = read_frames(dest)
        originals = read_frames(source)
        self.assertEqual(len(frames), metadata["target_frames"])
        self.assertLessEqual(abs(len(frames) / 30 - expected_duration), 1 / 30)
        self.assertAlmostEqual(probe_video_timeline(dest)["duration"], len(frames) / 30, places=5)
        if expected_indices is not None:
            np.testing.assert_allclose(frames.astype(float), originals[expected_indices].astype(float), atol=3)
        return timeline, frames

    def test_long_final_frame_preserves_content_and_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "long_last.mp4"
            make_video(source, [0, .04, .08, .12, .16], [.04] * 4 + [.50])
            timeline, frames = self.check_resample(source, .66, [0, 1, 2, 3, 3] + [4] * 15)
            self.assertEqual(len(frames), 20)
            self.assertTrue(timeline["is_vfr"])

    def test_cfr_files_24_25_30_60(self):
        with tempfile.TemporaryDirectory() as tmp:
            for fps in (24, 25, 30, 60):
                with self.subTest(fps=fps):
                    source = Path(tmp) / f"cfr_{fps}.mp4"
                    make_video(source, np.arange(fps) / fps, [1 / fps] * fps, rate=fps)
                    timeline, frames = self.check_resample(source, 1.0)
                    self.assertFalse(timeline["is_vfr"])
                    self.assertEqual(len(frames), 30)
                    # Each output image must come from a nearest source PTS;
                    # permit either source at exact midpoint ties.
                    originals = read_frames(source).astype(float).mean(axis=(1, 2, 3))
                    for i, frame in enumerate(frames):
                        distances = abs(np.arange(fps) / fps - i / 30)
                        candidates = originals[np.isclose(distances, distances.min())]
                        self.assertLessEqual(np.min(abs(candidates - frame.mean())), 3)

    def test_vfr_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "vfr.mp4"
            make_video(source, [0, .04, .11, .25, .28], [.04, .07, .14, .03, .20])
            timeline, _ = self.check_resample(source, .48, [0, 1, 1, 2, 2, 2, 3, 3] + [4] * 7)
            self.assertTrue(timeline["is_vfr"])

    def test_nonzero_start_and_longer_audio_do_not_extend_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "offset.mp4"
            make_video(source, [2, 2.04, 2.08, 2.12, 2.16], [.04] * 4 + [.50], audio_seconds=4)
            timeline, frames = self.check_resample(source, .66, [0, 1, 2, 3, 3] + [4] * 15)
            self.assertAlmostEqual(timeline["start_seconds"], 2.0, places=5)
            self.assertEqual(len(frames), 20)

    def test_reordered_b_frames_use_presentation_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "b_frames.mp4"
            make_video(source, np.arange(25) / 25, [.04] * 25, b_frames=2)
            with av.open(str(source)) as container:
                pts = [p.pts for p in container.demux(video=0) if p.pts is not None]
            self.assertTrue(any(a > b for a, b in zip(pts, pts[1:])), "Fixture must contain reordered packets")
            self.check_resample(source, 1.0)

    @staticmethod
    def fake_container(pts, stream_start=None, stream_duration=None, frame_durations=None):
        tb = Fraction(1, 1000)
        stream = SimpleNamespace(width=32, height=32, average_rate=25, time_base=tb,
                                 start_time=stream_start, duration=stream_duration)
        frames = []
        for i, timestamp in enumerate(pts):
            frame = SimpleNamespace(pts=timestamp, time_base=tb)
            if frame_durations is not None:
                frame.duration = frame_durations[i]
            frames.append(frame)
        packets = [SimpleNamespace(pts=f.pts, time_base=tb, duration=0, decode=lambda f=f: [f]) for f in frames]

        class Container:
            streams = SimpleNamespace(video=[stream])
            duration = 100000000  # Must never be used as video duration.
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def demux(self, stream): return iter(packets)

        return Container()

    def test_stream_end_fallback_handles_nonzero_start(self):
        fake = self.fake_container([2000, 2040], stream_start=2000, stream_duration=540)
        with patch.object(av, "open", return_value=fake):
            timeline = probe_video_timeline("fake.mp4")
        self.assertAlmostEqual(timeline["duration"], .54)
        self.assertEqual(timeline["duration_source"], "video_stream")

    def test_missing_final_duration_fails_even_with_fps_or_previous_duration(self):
        for pts, durations in (([0], [None]), ([0, 40], [40, None])):
            with self.subTest(pts=pts), patch.object(av, "open", return_value=self.fake_container(pts, frame_durations=durations)):
                with self.assertRaisesRegex(ValueError, "Cannot reliably determine"):
                    probe_video_timeline("missing.mp4")


if __name__ == "__main__":
    unittest.main()
