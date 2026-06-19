from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .models import ClipSelection, MediaInfo, format_timestamp, safe_filename_part


class FFmpegError(RuntimeError):
    pass


class FFmpegNotFoundError(FFmpegError):
    pass


class MediaProbeError(FFmpegError):
    pass


class ClipExportError(FFmpegError):
    pass


class FFmpegRunner:
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe") -> None:
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

    def check_available(self) -> None:
        if shutil.which(self.ffmpeg_path) is None:
            raise FFmpegNotFoundError("ffmpeg was not found. Install it with: sudo apt install ffmpeg")
        if shutil.which(self.ffprobe_path) is None:
            raise FFmpegNotFoundError("ffprobe was not found. Install it with: sudo apt install ffmpeg")

    def probe(self, path: str | Path) -> MediaInfo:
        self.check_available()
        media_path = Path(path)
        command = [
            self.ffprobe_path,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(media_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise MediaProbeError(completed.stderr.strip() or "ffprobe failed to read the media file.")

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise MediaProbeError("ffprobe returned invalid JSON.") from exc

        streams = payload.get("streams", [])
        has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
        real_video_streams = [stream for stream in streams if _is_real_video_stream(stream)]
        has_video = bool(real_video_streams)
        video_stream_index = _parse_optional_int(real_video_streams[0].get("index")) if real_video_streams else None
        if not has_audio and not has_video:
            raise MediaProbeError("The selected file does not contain audio or video streams.")

        format_payload = payload.get("format", {})
        duration = _parse_optional_float(format_payload.get("duration"))
        return MediaInfo(
            path=media_path,
            duration=duration,
            has_audio=has_audio,
            has_video=has_video,
            format_name=format_payload.get("format_name"),
            video_stream_index=video_stream_index,
        )

    def build_clip_command(
        self,
        input_path: str | Path,
        output_path: str | Path,
        start: float,
        end: float,
        has_video: bool,
        video_stream_index: int | None = None,
    ) -> list[str]:
        if end <= start:
            raise ValueError("Clip end timestamp must be greater than start timestamp.")

        duration = end - start
        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-y",
            "-i",
            str(input_path),
            "-ss",
            _format_seconds(start),
            "-t",
            _format_seconds(duration),
        ]

        if has_video:
            video_map = f"0:{video_stream_index}" if video_stream_index is not None else "0:v:0"
            command.extend(
                [
                    "-map",
                    video_map,
                    "-map",
                    "0:a?",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-c:a",
                    "aac",
                    "-movflags",
                    "+faststart",
                ]
            )
        else:
            command.extend(["-vn", "-c:a", "libmp3lame", "-q:a", "2"])

        command.append(str(output_path))
        return command

    def export_clip(
        self,
        input_path: str | Path,
        output_path: str | Path,
        selection: ClipSelection,
        has_video: bool,
        video_stream_index: int | None = None,
    ) -> None:
        self.check_available()
        command = self.build_clip_command(
            input_path=input_path,
            output_path=output_path,
            start=selection.start,
            end=selection.end,
            has_video=has_video,
            video_stream_index=video_stream_index,
        )
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise ClipExportError(completed.stderr.strip() or "ffmpeg failed to export the clip.")


def output_suffix_for_media(info: MediaInfo) -> str:
    return ".mp4" if info.has_video else ".mp3"


def build_clip_output_path(
    source_path: str | Path,
    output_dir: str | Path,
    index: int,
    selection: ClipSelection,
    suffix: str,
) -> Path:
    source = Path(source_path)
    start = safe_filename_part(format_timestamp(selection.start).replace(":", "-").replace(".", "-"))
    end = safe_filename_part(format_timestamp(selection.end).replace(":", "-").replace(".", "-"))
    return Path(output_dir) / f"{safe_filename_part(source.stem)}_clip_{index:03d}_{start}-{end}{suffix}"


def _format_seconds(seconds: float) -> str:
    return f"{max(0.0, seconds):.3f}"


def _parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_real_video_stream(stream: dict[str, object]) -> bool:
    if stream.get("codec_type") != "video":
        return False
    disposition = stream.get("disposition")
    if isinstance(disposition, dict):
        if disposition.get("attached_pic") == 1:
            return False
        if disposition.get("still_image") == 1:
            return False
    return True
