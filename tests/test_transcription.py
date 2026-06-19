from simple_clipper import transcription
from simple_clipper.transcription import TranscriptionConfig, build_voice_denoise_command


def test_auto_device_uses_cpu_when_nvidia_exists_but_cuda_runtime_is_missing(monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(transcription.shutil, "which", lambda command: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(transcription, "_cuda_runtime_available", lambda: False)

    device = transcription._resolve_device("auto", lambda percent, message: messages.append(message))

    assert device == "cpu"
    assert any("CUDA runtime libraries are unavailable" in message for message in messages)


def test_auto_device_uses_cuda_when_runtime_is_loadable(monkeypatch) -> None:
    monkeypatch.setattr(transcription.shutil, "which", lambda command: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(transcription, "_cuda_runtime_available", lambda: True)

    assert transcription._resolve_device("auto") == "cuda"


def test_cuda_runtime_error_detection_matches_missing_cublas() -> None:
    error = RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")

    assert transcription._is_cuda_runtime_error(error)


def test_voice_denoise_is_enabled_by_default() -> None:
    assert TranscriptionConfig().voice_denoise


def test_build_voice_denoise_command_creates_speech_focused_wav() -> None:
    command = build_voice_denoise_command("input.mp4", "clean.wav", ffmpeg_path="ffmpeg-test")

    assert command[:5] == ["ffmpeg-test", "-hide_banner", "-y", "-i", "input.mp4"]
    assert command[command.index("-map") + 1] == "0:a:0"
    assert "-vn" in command
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    filter_chain = command[command.index("-af") + 1]
    assert "highpass=f=80" in filter_chain
    assert "lowpass=f=8000" in filter_chain
    assert "afftdn" in filter_chain
    assert "anlmdn" in filter_chain
    assert "speechnorm" in filter_chain
    assert command[-1] == "clean.wav"
