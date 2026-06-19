from pathlib import Path

import pytest

from simple_clipper.media import (
    FFmpegRunner,
    build_clip_output_path,
    output_suffix_for_media,
    _is_real_video_stream,
)
from simple_clipper.models import ClipSelection, MediaInfo


def test_build_video_clip_command_uses_accurate_reencode_options() -> None:
    runner = FFmpegRunner(ffmpeg_path="ffmpeg-test", ffprobe_path="ffprobe-test")

    command = runner.build_clip_command(
        input_path="input.mov",
        output_path="out.mp4",
        start=12.3456,
        end=17.9,
        has_video=True,
    )

    assert command[:5] == ["ffmpeg-test", "-hide_banner", "-y", "-i", "input.mov"]
    assert command[command.index("-ss") + 1] == "12.346"
    assert command[command.index("-t") + 1] == "5.554"
    assert "-c:v" in command
    assert "libx264" in command
    assert "-c:a" in command
    assert "aac" in command
    assert command[-1] == "out.mp4"


def test_build_video_clip_command_can_map_absolute_stream_index() -> None:
    runner = FFmpegRunner(ffmpeg_path="ffmpeg-test", ffprobe_path="ffprobe-test")

    command = runner.build_clip_command(
        input_path="input.mov",
        output_path="out.mp4",
        start=12.0,
        end=17.0,
        has_video=True,
        video_stream_index=2,
    )

    assert command[command.index("-map") + 1] == "0:2"


def test_build_audio_clip_command_uses_mp3_reencode_options() -> None:
    runner = FFmpegRunner(ffmpeg_path="ffmpeg-test", ffprobe_path="ffprobe-test")

    command = runner.build_clip_command(
        input_path="input.wav",
        output_path="out.mp3",
        start=1.0,
        end=2.25,
        has_video=False,
    )

    assert "-vn" in command
    assert "libmp3lame" in command
    assert command[command.index("-t") + 1] == "1.250"


def test_build_clip_command_rejects_empty_range() -> None:
    runner = FFmpegRunner()

    with pytest.raises(ValueError):
        runner.build_clip_command("input.mp4", "out.mp4", start=2.0, end=2.0, has_video=True)


def test_output_suffix_matches_media_kind() -> None:
    video = MediaInfo(Path("x.mov"), duration=1.0, has_audio=True, has_video=True)
    audio = MediaInfo(Path("x.wav"), duration=1.0, has_audio=True, has_video=False)

    assert output_suffix_for_media(video) == ".mp4"
    assert output_suffix_for_media(audio) == ".mp3"


def test_build_clip_output_path_is_stable() -> None:
    selection = ClipSelection(start=1.2, end=3.4, text="hello", source_indices=(0,))

    path = build_clip_output_path("My Source.mov", "/tmp/out", 7, selection, ".mp4")

    assert path == Path("/tmp/out/My_Source_clip_007_00-01-200-00-03-400.mp4")


def test_attached_picture_stream_is_not_real_video() -> None:
    stream = {
        "codec_type": "video",
        "disposition": {
            "attached_pic": 1,
            "still_image": 0,
        },
    }

    assert not _is_real_video_stream(stream)


def test_normal_video_stream_is_real_video() -> None:
    stream = {
        "codec_type": "video",
        "disposition": {
            "attached_pic": 0,
            "still_image": 0,
        },
    }

    assert _is_real_video_stream(stream)
