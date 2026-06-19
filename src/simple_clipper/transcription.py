from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .models import TranscriptSegment, TranscriptionResult, TranscriptWord

ProgressCallback = Callable[[int, str], None]

VOICE_DENOISE_FILTER = (
    "highpass=f=80,"
    "lowpass=f=8000,"
    "afftdn=nf=-25,"
    "anlmdn=s=1:p=0.002:r=0.002,"
    "speechnorm=e=6.25:r=0.00001:l=1"
)


@dataclass(frozen=True)
class TranscriptionConfig:
    model_size: str = "base"
    word_timestamps: bool = False
    language: str | None = None
    device: str = "auto"
    compute_type: str = "auto"
    voice_denoise: bool = True


class TranscriptionUnavailableError(RuntimeError):
    pass


class AudioPreprocessError(RuntimeError):
    pass


class Transcriber:
    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config

    def transcribe(
        self,
        media_path: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionUnavailableError(
                "faster-whisper is not installed. Run: python -m pip install -e ."
            ) from exc

        with _transcription_audio_path(
            media_path=Path(media_path),
            voice_denoise=self.config.voice_denoise,
            progress_callback=progress_callback,
        ) as transcription_path:
            device = _resolve_device(self.config.device, progress_callback)
            model = _load_model_with_fallback(
                WhisperModel=WhisperModel,
                model_size=self.config.model_size,
                device=device,
                compute_type=_resolve_compute_type(self.config.compute_type, device),
                progress_callback=progress_callback,
            )

            try:
                return _run_transcription(
                    model=model,
                    media_path=transcription_path,
                    word_timestamps=self.config.word_timestamps,
                    language=self.config.language,
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                if device == "cpu" or not _is_cuda_runtime_error(exc):
                    raise

                if progress_callback:
                    progress_callback(1, "CUDA failed during transcription; retrying on CPU")
                cpu_model = WhisperModel(self.config.model_size, device="cpu", compute_type="int8")
                return _run_transcription(
                    model=cpu_model,
                    media_path=transcription_path,
                    word_timestamps=self.config.word_timestamps,
                    language=self.config.language,
                    progress_callback=progress_callback,
                )


@contextlib.contextmanager
def _transcription_audio_path(
    media_path: Path,
    voice_denoise: bool,
    progress_callback: ProgressCallback | None,
):
    if not voice_denoise:
        yield media_path
        return

    with tempfile.TemporaryDirectory(prefix="simple_clipper_voice_") as temp_dir:
        denoised_path = Path(temp_dir) / "voice_denoised.wav"
        _create_voice_denoised_wav(media_path, denoised_path, progress_callback)
        yield denoised_path


def _create_voice_denoised_wav(
    input_path: Path,
    output_path: Path,
    progress_callback: ProgressCallback | None = None,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    if shutil.which(ffmpeg_path) is None:
        raise AudioPreprocessError("ffmpeg was not found. Install it with: sudo apt install ffmpeg")

    if progress_callback:
        progress_callback(1, "Preparing voice-focused audio")

    command = build_voice_denoise_command(input_path, output_path, ffmpeg_path=ffmpeg_path)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or "ffmpeg failed while preparing voice-focused audio."
        raise AudioPreprocessError(message)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise AudioPreprocessError("Voice-focused audio preprocessing produced an empty file.")

    if progress_callback:
        progress_callback(4, "Voice-focused audio ready")


def build_voice_denoise_command(
    input_path: str | Path,
    output_path: str | Path,
    ffmpeg_path: str = "ffmpeg",
    filter_chain: str = VOICE_DENOISE_FILTER,
) -> list[str]:
    return [
        ffmpeg_path,
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        filter_chain,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]


def _resolve_device(device: str, progress_callback: ProgressCallback | None = None) -> str:
    if device != "auto":
        return device
    if not shutil.which("nvidia-smi"):
        return "cpu"
    if _cuda_runtime_available():
        return "cuda"
    if progress_callback:
        progress_callback(1, "GPU found, but CUDA runtime libraries are unavailable; using CPU")
    return "cpu"


def _resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def _load_model_with_fallback(
    WhisperModel: object,
    model_size: str,
    device: str,
    compute_type: str,
    progress_callback: ProgressCallback | None,
) -> object:
    if progress_callback:
        progress_callback(1, f"Loading {model_size} model on {device}")
    try:
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as exc:
        if device == "cpu" or not _is_cuda_runtime_error(exc):
            raise
        if progress_callback:
            progress_callback(1, "CUDA model load failed; retrying on CPU")
        return WhisperModel(model_size, device="cpu", compute_type="int8")


def _run_transcription(
    model: object,
    media_path: str | Path,
    word_timestamps: bool,
    language: str | None,
    progress_callback: ProgressCallback | None,
) -> TranscriptionResult:
    if progress_callback:
        progress_callback(5, "Starting transcription")

    segments_iter, info = model.transcribe(
        str(media_path),
        beam_size=5,
        vad_filter=True,
        word_timestamps=word_timestamps,
        language=language,
    )

    duration = getattr(info, "duration", None)
    detected_language = getattr(info, "language", None)
    segments: list[TranscriptSegment] = []
    for index, segment in enumerate(segments_iter):
        words = tuple(
            TranscriptWord(
                start=float(word.start),
                end=float(word.end),
                text=str(word.word),
                segment_id=index,
            )
            for word in (segment.words or [])
        )
        segments.append(
            TranscriptSegment(
                id=index,
                start=float(segment.start),
                end=float(segment.end),
                text=str(segment.text).strip(),
                words=words,
            )
        )
        if progress_callback and duration:
            percent = min(99, max(5, int((float(segment.end) / float(duration)) * 100)))
            progress_callback(percent, f"Transcribed to {segment.end:.1f}s")

    if progress_callback:
        progress_callback(100, "Transcription complete")

    return TranscriptionResult(
        segments=tuple(segments),
        language=detected_language,
        duration=duration,
        word_timestamps=word_timestamps,
    )


def _cuda_runtime_available() -> bool:
    return _shared_library_loadable("libcublas.so.12") or _shared_library_loadable("cublas")


def _shared_library_loadable(name: str) -> bool:
    candidates = [name]
    found = ctypes.util.find_library(name)
    if found:
        candidates.append(found)

    for candidate in candidates:
        try:
            ctypes.CDLL(candidate)
            return True
        except OSError:
            continue
    return False


def _is_cuda_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        needle in message
        for needle in (
            "libcublas",
            "libcudnn",
            "cuda",
            "cublas",
            "cudnn",
        )
    )
