# Simple Clipper

Simple Clipper is a local desktop app for turning audio/video files into transcript-based short clips.

It imports a media file, transcribes it with local Whisper via `faster-whisper`, shows timestamped transcript rows, lets you select transcript ranges, and exports accurate clips with FFmpeg.

## Features

- Local Whisper transcription through `faster-whisper`
- Segment-level and optional word-level transcript selection
- Voice-focused denoise preprocessing for clearer transcription input
- Transcript-based clip selection with editable merged clip text
- Accurate FFmpeg export from the original source media
- MP4 export for video sources and MP3 export for audio sources

## Requirements

- Ubuntu 24.04 or similar Linux desktop
- Python 3.12+
- FFmpeg and FFprobe
- `python3.12-venv` for the virtual environment workflow below
- Optional NVIDIA GPU for faster local transcription

Install system packages:

```bash
sudo apt update
sudo apt install -y ffmpeg python3.12-venv
```

Optional NVIDIA GPU runtime for current `faster-whisper` / `ctranslate2`:

```bash
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring_1.1-1_all.deb
sudo dpkg -i /tmp/cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y libcublas-12-9 libcudnn9-cuda-12
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

`socksio` is included because Hugging Face model downloads use `httpx`, and `httpx` requires this extra package when `ALL_PROXY` or `all_proxy` points to a SOCKS proxy.

## Run

```bash
simple-clipper
```

or:

```bash
python -m simple_clipper
```

The first transcription downloads the selected Whisper model. Segment-level transcription is the default. Enable word-level selection before transcribing when you need tighter text-based clips; it takes more processing time.

Voice-focused denoise is enabled by default. Before Whisper runs, the app creates a temporary mono 16 kHz WAV with speech-band filtering, FFT denoise, non-local-means denoise, and speech normalization. This cleaned temporary file is used only for transcription; exported clips are still cut from the original media.

After adding selected transcript rows as clips, use the merged clip review area to edit and copy the merged text before export. Editing text does not change the clip timestamps.

## Test

```bash
PYTHONPATH=src python -m pytest -q
```

## Development

The application code lives in `src/simple_clipper/`, with focused tests in `tests/`.

Useful checks before committing:

```bash
PYTHONPATH=src python -m pytest -q
python -m pip install -e ".[dev]"
```

## Notes

- Video inputs export MP4 clips with H.264 video and AAC audio.
- Audio inputs export MP3 clips.
- Export uses accurate re-encoding by default, so it favors timestamp precision over speed.
- If CUDA model loading fails, transcription falls back to CPU automatically.
- `nvidia-smi` alone is not enough for GPU transcription. `faster-whisper` also needs CUDA runtime libraries such as `libcublas.so.12`; when they are unavailable, the app uses CPU.
- Voice-focused denoise suppresses broad noise and non-speech frequencies, but it is not full source separation. Very loud music or overlapping speakers can still reduce accuracy.

## License

Simple Clipper is released under the MIT License. See [LICENSE](LICENSE).
