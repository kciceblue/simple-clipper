from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TranscriptWord:
    start: float
    end: float
    text: str
    segment_id: int


@dataclass(frozen=True)
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    words: tuple[TranscriptWord, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TranscriptionResult:
    segments: tuple[TranscriptSegment, ...]
    language: str | None = None
    duration: float | None = None
    word_timestamps: bool = False


@dataclass(frozen=True)
class ClipSelection:
    start: float
    end: float
    text: str
    source_indices: tuple[int, ...]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration: float | None
    has_audio: bool
    has_video: bool
    format_name: str | None = None
    video_stream_index: int | None = None

    @property
    def media_kind(self) -> str:
        if self.has_video:
            return "video"
        if self.has_audio:
            return "audio"
        return "unknown"


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "--:--.---"
    total_millis = int(round(max(0.0, float(seconds)) * 1000))
    hours, remainder = divmod(total_millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"
    return f"{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def safe_filename_part(value: str, limit: int = 80) -> str:
    allowed = []
    for char in value.strip().replace(" ", "_"):
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
    cleaned = "".join(allowed).strip("_")
    return (cleaned or "clip")[:limit]
