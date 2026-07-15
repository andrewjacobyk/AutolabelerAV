# VLA Pipeline

A **100% local, 100% open-weight** GUI application for building,
testing, fine-tuning and validating a Vision-Language Model pipeline
over video content.

The primary use case is:

> "Take a video, sample a few frames per minute, ask a VLM to describe
> each frame, and store the resulting captions with metadata as JSON."

Designed to run end-to-end on a single machine with **64 GB RAM +
NVIDIA RTX 4070 (8 GB VRAM)**. To scale to bigger models, run the same
pipeline on a bigger GPU — everything stays on your own hardware; no
paid APIs are used or supported.

---

## 1. Features

- **Modern GUI** (CustomTkinter) with 5 tabs mirroring the pipeline:
  1. **Extract** — pull *N* frames per minute from a video / folder.
  2. **Inference** — caption every frame with an open-weight VLM.
  3. **Fine-tune** — build a JSONL dataset and run LoRA training.
  4. **Validate** — compute BLEU, ROUGE-L and cosine similarity.
  5. **Settings** — paths, HF token (for gated models), GPU/RAM monitor.
- **13 open-weight VLM backends bundled** — Western + Chinese
  (see [§5](#5-model-catalog)).
- **Reproducible JSON output** with a versioned schema
  (see [§7](#7-output-json-schema)).
- **LoRA fine-tuning** targeting the language head, so training fits
  inside 8 GB VRAM.
- **Windows-first** installer (`install.bat`) with Linux/macOS
  companion (`install.sh`).
- **No API keys, no telemetry, no paid services.**

---

## 2. Requirements

| Item        | Recommended                                            |
|-------------|--------------------------------------------------------|
| OS          | Windows 10/11, Linux, macOS                             |
| Python      | 3.10, 3.11, 3.12 or 3.13                                |
| GPU         | NVIDIA GPU with CUDA 12.x drivers                       |
| VRAM        | 6 GB minimum, 8 GB comfortable (RTX 4070)               |
| RAM         | 16 GB minimum, 64 GB recommended                        |
| Disk        | 40 GB free (multiple model weights + frames cache)     |

CPU-only operation is supported as a fallback but will be very slow.

---

## 3. Install

### Windows

```powershell
cd C:\Users\<you>\Documents\CursorAI\VLA_Pipeline
.\install.bat
```

The installer:

1. Locates a Python 3.10 – 3.13 interpreter (prefers 3.12).
2. Creates a `.venv` virtual environment.
3. Installs PyTorch from the CUDA 12.6 wheel index (falls back to
   CUDA 11.8, then CPU-only).
4. Installs the rest of `requirements.txt`.
5. Creates `data/{videos,frames,outputs,datasets,models}` and `logs/`.
6. Writes a full transcript to `logs/install.log` (window stays open
   on failure so you can read the error).

### Linux / macOS

```bash
chmod +x install.sh run.sh
./install.sh
```

### Optional: Hugging Face token

Only needed to download **gated** open-weight models
(e.g. `google/paligemma-3b-mix-448` requires accepting Google's
license on the Hub):

```bash
cp .env.example .env
# then edit .env: HUGGINGFACE_TOKEN=hf_xxx
```

You can also enter the token temporarily from the **Settings** tab.

---

## 4. Launch

```powershell
.\run.bat            # Windows
./run.sh             # Linux / macOS
```

The GUI opens with a status bar showing the current CUDA device and
live CPU / RAM / VRAM utilisation.

---

## 5. Model catalog

Every model is open-weight (freely downloadable from Hugging Face) and
runs entirely on your GPU. Sizes assume fp16 unless noted.

### Western / global

| Model            | HF id                                     | Size    | Notes                                         |
|------------------|-------------------------------------------|---------|-----------------------------------------------|
| Moondream2       | `vikhyatk/moondream2`                     | 1.9B / ~4 GB  | Default. Best quality-per-size.         |
| Florence-2-base  | `microsoft/Florence-2-base`               | 232M / <1 GB  | Fastest local option.                   |
| Florence-2-large | `microsoft/Florence-2-large`              | 770M / ~2 GB  | Higher-quality captions.                |
| SmolVLM-256M     | `HuggingFaceTB/SmolVLM-256M-Instruct`     | 256M / <1 GB  | Tiny & fast, decent captions.           |
| SmolVLM-500M     | `HuggingFaceTB/SmolVLM-500M-Instruct`     | 500M / ~1.5 GB| Nice quality/speed trade-off.           |
| SmolVLM          | `HuggingFaceTB/SmolVLM-Instruct`          | 2.2B / ~5 GB  | Handles longer prompts.                 |
| BLIP-2 OPT-2.7B  | `Salesforce/blip2-opt-2.7b`               | 2.7B / ~6 GB (int4: ~3 GB) | Classic; use `int4`.       |
| PaliGemma-3B     | `google/paligemma-3b-mix-448`             | 3B / ~6 GB    | Google open-weight (gated, HF token).   |

### Chinese open-weight

| Model                    | HF id                                | Size / VRAM             | Origin                                |
|--------------------------|--------------------------------------|-------------------------|---------------------------------------|
| Qwen2-VL-2B-Instruct     | `Qwen/Qwen2-VL-2B-Instruct`          | 2B / ~5 GB              | Alibaba                               |
| Qwen2-VL-7B-Instruct     | `Qwen/Qwen2-VL-7B-Instruct`          | 7B / ~7 GB int4         | Alibaba                               |
| Qwen2.5-VL-3B-Instruct   | `Qwen/Qwen2.5-VL-3B-Instruct`        | 3B / ~7 GB              | Alibaba (newer)                       |
| Qwen2.5-VL-7B-Instruct   | `Qwen/Qwen2.5-VL-7B-Instruct`        | 7B / ~7 GB int4         | Alibaba (newer)                       |
| InternVL2.5-1B           | `OpenGVLab/InternVL2_5-1B`           | 1B / ~3 GB              | Shanghai AI Lab                       |
| InternVL2.5-2B           | `OpenGVLab/InternVL2_5-2B`           | 2B / ~5 GB              | Shanghai AI Lab                       |
| InternVL2.5-4B           | `OpenGVLab/InternVL2_5-4B`           | 4B / ~7 GB int4         | Shanghai AI Lab                       |
| InternVL3-2B             | `OpenGVLab/InternVL3-2B`             | 2B / ~5 GB              | Shanghai AI Lab (newest)              |
| MiniCPM-V-2.6            | `openbmb/MiniCPM-V-2_6`              | 8B / ~7 GB int4         | OpenBMB / Tsinghua                    |

**Precision tips for 8 GB VRAM:**

- ≤ 3B params: `fp16` (default) is fine.
- 4B – 7B params: use `int4` on the Inference tab.
- The bundled `bitsandbytes` handles Windows int4/int8 quantisation
  automatically.

### Estimated VRAM by model × precision

Weights + KV cache + CUDA overhead, batch = 1.  "n/a" means that
precision isn't wired up for that model family in this build.  The
same table is shown live inside the app under Settings.

| Model                    | Params | fp16 / bf16 | int8   | int4   |
|--------------------------|--------|-------------|--------|--------|
| moondream2               |  1.9B  |  4.9 GB     | 2.8 GB | 1.9 GB |
| florence2-base           |  0.2B  |  1.2 GB     |  n/a   |  n/a   |
| florence2-large          |  0.8B  |  2.4 GB     |  n/a   |  n/a   |
| smolvlm-256m             |  0.3B  |  1.3 GB     |  n/a   |  n/a   |
| smolvlm-500m             |  0.5B  |  1.8 GB     |  n/a   |  n/a   |
| smolvlm (2.2B)           |  2.2B  |  5.5 GB     |  n/a   |  n/a   |
| blip2-opt-2.7b           |  3.7B  |  8.8 GB     | 4.8 GB | 3.0 GB |
| paligemma-3b-mix-448     |  3.0B  |  7.2 GB     |  n/a   |  n/a   |
| qwen2-vl-2b-instruct     |  2.2B  |  5.5 GB     | 3.1 GB | 2.0 GB |
| qwen2-vl-7b-instruct     |  8.3B  | 18.5 GB     | 9.6 GB | 5.6 GB |
| qwen2.5-vl-3b-instruct   |  3.8B  |  8.8 GB     | 4.8 GB | 3.0 GB |
| qwen2.5-vl-7b-instruct   |  8.3B  | 18.5 GB     | 9.6 GB | 5.6 GB |
| internvl2.5-1b           |  0.9B  |  2.8 GB     | 1.8 GB | 1.3 GB |
| internvl2.5-2b           |  2.2B  |  5.5 GB     | 3.1 GB | 2.0 GB |
| internvl2.5-4b           |  3.7B  |  8.7 GB     | 4.7 GB | 2.9 GB |
| internvl3-2b             |  2.2B  |  5.5 GB     | 3.1 GB | 2.0 GB |
| minicpm-v-2.6            |  8.1B  | 18.1 GB     | 9.4 GB | 5.5 GB |

**Rule of thumb for an RTX 4070 8 GB card:**

- Green (fp16): moondream2 · florence2-\* · smolvlm-\* (any) · internvl2.5-1b · internvl2.5-2b · internvl3-2b · qwen2-vl-2b · paligemma-3b (tight)
- Requires int4: blip2-opt-2.7b · qwen2.5-vl-3b · internvl2.5-4b
- Requires int4 AND still ~5.5 GB VRAM: qwen*-vl-7b · minicpm-v-2.6

Disk footprint per model equals `params × 2 bytes` (fp16 weights are
always downloaded, then quantised on the fly for int8/int4).  Host RAM
peaks at ~1.5× that during load.  The Inference tab shows a live
warning **before** starting a run if the pick doesn't fit.

---

## 6. Typical local workflow (RTX 4070, 8 GB VRAM)

1. Drop one or more videos into `data/videos/`.
2. **Tab 1 — Extract**: keep the default `6 frames/minute`, hit
   *Extract*. Each video gets its own subfolder under `data/frames/`
   with a `manifest.json`.
3. **Tab 2 — Inference**: pick a model (start with `moondream2` or
   `qwen2-vl-2b-instruct`), choose precision (`fp16` for ≤ 3 B params,
   `int4` for larger models), then run on a single folder or *all
   subfolders*. Descriptions land in `data/outputs/<video-name>.json`.
4. (Optional) **Tab 3 — Fine-tune**: click *Build dataset* to convert
   the outputs into a JSONL, then *Start fine-tune*. LoRA adapters
   land in `data/models/finetuned/`.
5. (Optional) **Tab 4 — Validate**: pick a reference JSON and a
   hypothesis JSON to compare (BLEU / ROUGE-L / cosine).

---

## 7. Output JSON schema

Every video produces one JSON that looks like:

```json
{
  "schema_version": 1,
  "video": {
    "name": "vacation.mp4",
    "path": "data/videos/vacation.mp4",
    "duration_sec": 187.4,
    "fps": 29.97,
    "frame_count": 5615,
    "width": 1920,
    "height": 1080
  },
  "extraction": {
    "frames_per_minute": 6,
    "frames_dir": "data/frames/vacation",
    "num_frames": 19
  },
  "model": {
    "id": "qwen2-vl-2b-instruct",
    "hf_id": "Qwen/Qwen2-VL-2B-Instruct",
    "kind": "local",
    "precision": "fp16"
  },
  "prompt": "Describe this scene ...",
  "generated_at": "2026-07-14T22:41:03+00:00",
  "frames": [
    {
      "index": 0,
      "source_frame": 0,
      "timestamp_sec": 0.0,
      "timestamp_hhmmss": "00:00:00.000",
      "file": "vacation_000000.jpg",
      "description": "A wide beach at sunset with two people walking."
    }
  ]
}
```

The schema is stable within a `schema_version` and is the direct
input for the fine-tune / validation steps.

---

## 8. Project layout

```
VLA_Pipeline/
├── install.bat / install.sh       # Environment setup
├── run.bat     / run.sh           # Launcher
├── config.yaml                    # All defaults; editable in Settings
├── requirements.txt
├── src/
│   ├── main.py                    # Entrypoint
│   ├── core/
│   │   ├── config.py              # YAML config
│   │   ├── video.py               # Frame extraction (OpenCV)
│   │   ├── dataset.py             # JSON schema + JSONL builder
│   │   ├── inference.py           # Orchestrates VLM over frames
│   │   ├── finetune.py            # LoRA training (Florence-2)
│   │   ├── validate.py            # BLEU / ROUGE / cosine metrics
│   │   └── vlm/                   # Per-family model adapters
│   │       ├── moondream.py
│   │       ├── florence.py
│   │       ├── smolvlm.py
│   │       ├── blip.py
│   │       ├── paligemma.py
│   │       ├── qwenvl.py          # Qwen2-VL / Qwen2.5-VL
│   │       ├── internvl.py        # InternVL 2 / 2.5 / 3
│   │       └── minicpm.py         # MiniCPM-V
│   ├── gui/                       # CustomTkinter tabs & widgets
│   └── utils/                     # Logger, GPU monitor, paths
└── data/                          # Created at runtime
    ├── videos/  frames/  outputs/  datasets/  models/
```

---

## 9. Resource estimates (RTX 4070, 8 GB VRAM)

Run this any time to print the full matrix:

```powershell
.venv\Scripts\python.exe -m src.tools.print_resources
```

Or open **Inference → Resource table...** in the GUI.

Columns are **peak VRAM (GB)** at each precision. Add ~0.5 GB headroom
for the CUDA driver. Disk cache is always the fp16 shard size
(~2 bytes/param); int4/int8 still download fp16 weights first.

| Model | Params | fp16 | int4 | Fits 8 GB fp16? | Fits 8 GB int4? |
|-------|--------|------|------|-----------------|-----------------|
| florence2-base | 0.2B | 1.2 | n/a | yes | n/a |
| smolvlm-256m | 0.3B | 1.3 | n/a | yes | n/a |
| internvl2_5-1b | 0.9B | 2.8 | 1.3 | yes | yes |
| florence2-large | 0.8B | 2.4 | n/a | yes | n/a |
| smolvlm-500m | 0.5B | 1.8 | n/a | yes | n/a |
| **moondream2** | 1.9B | 4.9 | 1.9 | **yes** | yes |
| qwen2-vl-2b | 2.2B | 5.5 | 2.0 | yes | yes |
| internvl2_5-2b / internvl3-2b | 2.2B | 5.5 | 2.0 | yes | yes |
| smolvlm (2.2B) | 2.2B | 5.5 | n/a | yes | n/a |
| paligemma-3b | 3.0B | 7.2 | n/a | borderline | n/a |
| blip2-opt-2.7b | 3.7B | 8.8 | 3.0 | no | yes |
| internvl2_5-4b | 3.7B | 8.7 | 2.9 | no | yes |
| qwen2_5-vl-3b | 3.8B | 8.8 | 3.0 | no | yes |
| qwen2-vl-7b / qwen2_5-vl-7b | 8.3B | 18.5 | 5.6 | no | yes |
| minicpm-v-2_6 | 8.1B | 18.1 | 5.5 | no | yes |

**Recommended on your machine (8 GB):**
- Start with `moondream2` (fp16) or `florence2-base` (fastest).
- Chinese models ≤ 2B: `qwen2-vl-2b-instruct`, `internvl3-2b` (fp16).
- Models ≥ 3B at fp16 will OOM — use **int4** or pick a smaller variant.

---

## 10. Troubleshooting

- **Download looks stuck / shows "Reconstructing".** This was the
  Hugging Face *xet* backend showing two overlapping progress bars.
  `run.bat` now forces the classic HTTP downloader
  (`HF_HUB_DISABLE_XET=1`). Use **Download weights only** in the
  Inference tab, or pre-download from a terminal:

  ```powershell
  download_model.bat moondream2
  ```

  Let large models finish — closing the app mid-download restarts the
  transfer. Partial files resume automatically on the next attempt.

- **Symlink warning on Windows.** Enable **Developer Mode** in Windows
  Settings → System → For developers. Without it, the HF cache uses
  extra disk space but still works.

- **Unauthenticated HF requests / slow downloads.** Set
  `HUGGINGFACE_TOKEN` in Settings or `.env` for higher rate limits.

- **`No manifest.json in data/frames`.** For a single-video run, pick
  the per-video subfolder (e.g. `data/frames/m2-res_480p_dashcam`), not
  the parent `data/frames` directory.

- **`torch.cuda.is_available()` is False after install.** Reboot after
  installing the NVIDIA driver, then re-run `install.bat`. Consult
  `logs/install.log` for the full pip transcript.

- **Out of memory.** Switch precision to `int4` on the Inference tab.
  The GUI now warns before loading an oversized model.

- **Fine-tune says "unsupported family".** Only *florence* has a
  bundled LoRA recipe in this build.

---

## 11. License

Code in this repository is released under the MIT License. Model
weights retain their own licenses (see each model card on
Hugging Face).
