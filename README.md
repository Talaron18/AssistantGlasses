<div align="center">

# AssistantGlasses

**An open-source AI-powered wearable assistant for the visually impaired**

Real-time navigation · Obstacle detection · Voice interaction · Multimodal AI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://python.org)

</div>

---

## Overview

AssistantGlasses is a modular, glasses-form-factor device that helps visually impaired users navigate the world independently. It combines GPS navigation, depth-aware obstacle detection, wake-word speech recognition, and conversational AI into a single integrated system running on edge hardware.

**Key capabilities:**

- Turn-by-turn pedestrian navigation with Kalman-filtered GPS
- Real-time obstacle detection with metric depth estimation (YOLO + Depth-Anything-V2)
- Wake-word activated voice assistant with multimodal understanding
- Tool-calling AI agent that can capture and analyze photos on command
- Designed for low-latency, edge-based processing with graceful fallbacks

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                     src.py  (Orchestrator)                   │
├───────────┬─────────────┬────────────────┬───────────────────┤
│   Agent   │   Vision    │  Navigation    │  Speech & Voice   │
│           │             │                │                   │
│ LLM APIs  │ Camera      │ GNSS + Amap    │ Mic + TTS Engine  │
│ Tool Call │ YOLO+Depth  │ Kalman Filter  │ Whisper+Porcupine │
└───────────┴─────────────┴────────────────┴───────────────────┘
```

The system orchestrates four independent modules through queue-based communication. Each module can also run standalone for development and testing.

## Modules

| Module | Description | Key Tech |
|--------|-------------|----------|
| **Navigation** | GPS-based walking directions with route deviation detection | NMEA parser, WGS-84→GCJ-02, Kalman filter, Amap API |
| **Vision** | Obstacle detection with distance measurement | YOLOv8, Depth-Anything-V2, OpenCV |
| **Speech** | Wake-word detection + speech-to-text | Porcupine, Whisper-base, OpenVINO |
| **Voice** | Text-to-speech for guidance output | Kokoro ONNX, Edge TTS |
| **Agent** | Conversational AI with tool calling | GLM-4, Qwen3-VL, multimodal analysis |

## Hardware

| Component | Spec | Role |
|-----------|------|------|
| Camera (USB) | 720p+ | Object detection, depth estimation, photo capture |
| GNSS Receiver | Serial, 38400 baud | GPS positioning |
| Microphone | 16kHz mono | Voice input |
| Earphones | Bluetooth / wired | TTS output |
| Vibration Motors ×4 | — | Directional haptic feedback *(planned)* |

## Getting Started

### Prerequisites

- Python 3.13+
- CUDA-compatible GPU (recommended for real-time vision inference)
- API keys: [Amap](https://lbs.amap.com/) · [Picovoice](https://picovoice.ai/) · [Siliconflow](https://siliconflow.cn/) or OpenAI-compatible

### Installation

```bash
git clone https://github.com/Talaron18/AssistantGlasses.git
cd AssistantGlasses
pip install -r requirements.txt
```

### Configuration

Copy and edit the environment file:

```bash
cp .env.example .env
```

```ini
AMAP_API_KEY=your_amap_key
PICOVOICE_ACCESS_KEY=your_picovoice_key
SILICONFLOW_API_KEY=your_siliconflow_key

# Optional: Kokoro TTS model paths
ONNX-ZH=/path/to/chinese_model.onnx
ONNX-EN=/path/to/english_model.onnx
```

GNSS and navigation settings are in `navigation_module/config/system_config.yaml`.

### Usage

**Run the full system:**

```bash
python src.py
```

**Run modules independently:**

```bash
python navigation_module/core/main.py    # Navigation only
python vision_module/local_metric_depth.py  # Vision pipeline
```

## Project Structure

```
AssistantGlasses/
├── src.py                        # Main orchestrator
├── Agent/                        # Siliconflow-based AI agent
│   └── code/
│       ├── chat.py               # Conversation loop
│       ├── config.py             # Model & persona config
│       └── request.py            # API utilities
├── Gemma/                        # OpenAI-compatible agent (alternative)
├── navigation_module/
│   ├── core/                     # NavController + entry point
│   ├── algo/
│   │   ├── fusion/               # Kalman filter
│   │   └── geo/                  # Coordinate transforms
│   ├── sensors/gnss/             # Serial reader + NMEA parser
│   ├── services/                 # Amap API provider
│   └── config/                   # system_config.yaml
├── vision_module/
│   ├── local_metric_depth.py     # YOLO + depth fusion
│   ├── local_relative.py         # YOLO relative positioning
│   └── metric_depth/             # Depth-Anything-V2 (model + training)
├── speech_module/
│   ├── stream/                   # Wake-word + recording
│   └── tests/                    # ASR tests
├── voice_module/
│   ├── kokoro.py                 # Kokoro ONNX TTS
│   ├── edge_tts.py              # Edge TTS fallback
│   └── config.py                 # Voice settings
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Commit your changes
4. Open a Pull Request

## Acknowledgements

- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) — Metric depth estimation
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — Object detection
- [Kokoro](https://github.com/hexgrad/kokoro) — TTS engine
- [Porcupine](https://github.com/Picovoice/porcupine) — Wake-word detection
- [OpenAI Whisper](https://github.com/openai/whisper) — Speech recognition

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
