# Changes in This Fork vs Original Sammie-Roto 2

This document describes all modifications and new features in this fork compared to the original [Zarxrax/Sammie-Roto-2](https://github.com/Zarxrax/Sammie-Roto-2) codebase. It is intended to support review and potential merge into the upstream repository.

---

## Summary

| Area | Original | This Fork |
|------|----------|-----------|
| **Matting engines** | MatAnyone, MatAnyone 2 | + **VideoMaMa** (diffusion-based, optional) |
| **Tabs** | Segmentation, Matting, Object Removal | + **Corridor Key** (AI green screen keying) |
| **View modes** | 6 modes | + CK-Alpha, CK-FG, CK-Comp |
| **Export output types** | Segmentation/Matting/Removal | + **CK-Alpha**, **CK-FG**, **CK-Comp** (16-bit PNG/EXR) |
| **Dependencies** | requirements.txt (no antlr4 pin) | + `antlr4-python3-runtime==4.9.3`, `einops`, `huggingface_hub`, `timm`, `transformers` |
| **Launch (Linux)** | `run_sammie.command` basic | + `QT_QPA_PLATFORM=xcb`, `QT_X11_NO_MITSHM=1`, `CORRIDORKEY_OPT_MODE=lowvram` |

New models (VideoMaMa, CorridorKey) are **downloaded on first use**, so installation size and time are unchanged until the user runs those features.

---

## 1. VideoMaMa (Mask-Guided Video Matting)

### What It Is

- **VideoMaMa** is a diffusion-based video matting model (Stable Video Diffusion fine-tuned for matting).  
- **Source:** [cvlab-kaist/VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa) (CVPR 2026).  
- It produces more temporally consistent mattes than frame-by-frame MatAnyone, at the cost of higher VRAM and longer processing.  
- It requires a **segmentation mask on all frames** as guidance (run Segmentation → Propagate first).

### How It Works in the App

- In the **Matting** tab, the user can choose the engine: **MatAnyone**, **MatAnyone 2**, or **VideoMaMa**.
- When VideoMaMa is selected and the user clicks **Run Matting**, the app:
  1. Ensures models are downloaded (SVD base + VideoMaMa UNet, ~19 GB with `allow_patterns` to avoid redundant files).
  2. Runs the VideoMaMa pipeline (batch of 16 frames, overlap/blending for longer videos).
  3. Writes mattes under `temp/matting/` per frame/object; the viewer shows **Matting-Matte** as today.

### Main Files for VideoMaMa

| File | Role |
|------|------|
| **`videomama/`** (new directory) | VideoMaMa integration |
| `videomama/__init__.py` | Package init |
| `videomama/videomama_manager.py` | Manager: download (HuggingFace), run pipeline, progress UI, VRAM cleanup |
| `videomama/pipeline_svd_mask.py` | SVD + mask pipeline (diffusers), batch inference |

### Files Modified to Support VideoMaMa

| File | Changes |
|------|---------|
| `sammie_main.py` | MattingTab: engine combo (MatAnyone / MatAnyone 2 / VideoMaMa), `run_matting()` branches to VideoMaMaManager, VRAM cleanup before load, warning for low VRAM |
| `sammie/settings_manager.py` | `default_matting_engine`, `default_videomama_overlap`, `default_videomama_resolution`, `default_videomama_vae_tiling`; SessionSettings + persistence for VideoMaMa |
| `sammie/sammie.py` | View/export handling for matting output (unchanged for existing MatAnyone; VideoMaMa writes same matting paths) |
| `requirements.txt` | `diffusers`, `accelerate`, `huggingface_hub`, etc. (already in original); no extra package beyond existing stack |

---

## 2. Corridor Key (New Tab and Pipeline)

### What It Is

- **CorridorKey** is an AI green screen keyer: it refines a coarse matte (from Segmentation or Matting), outputs **refined alpha**, **despilled foreground**, and **composite**.  
- **Source:** [EZ-CorridorKey](https://github.com/edenaion/EZ-CorridorKey) (GUI by Ed Zisk; upstream [nikopueringer/CorridorKey](https://github.com/nikopueringer/CorridorKey)).  
- Inference logic and low-VRAM behavior (tiling, quality presets) follow the EZ-CorridorKey approach, integrated as a dedicated tab and export types.

### Tab: Corridor Key

- New sidebar tab **Corridor Key** with:
  - **Run CorridorKey** / **Clear CorridorKey**
  - **Mask Source:** Segmentation | Matting (which mask to refine).
  - **Quality:** Low (1024) | Medium (1536) | High (2048) — internal resolution; auto by VRAM if desired.
  - **Refiner Scale:** 0–2 (edge refinement strength).
  - **Despill:** 0–1 (green spill removal on FG).
  - **Auto Despeckle** + **Despeckle Size** (pixel threshold).
  - **Tiling:** optional tile-based refiner for low VRAM.

### View Modes and Export

- New view modes: **CK-Alpha**, **CK-FG**, **CK-Comp** (viewer reads from `corridorkey/` outputs).
- **File Export → Output Type:** added **CK-Alpha**, **CK-FG**, **CK-Comp** for all relevant formats (video and image sequences, including EXR and 16-bit PNG where applicable).

### Outputs and Quality

- Alpha, FG, and Comp are saved as **16-bit PNG** in a `corridorkey/` subfolder; an 8-bit alpha copy is still written for the existing matting viewer path if needed.
- EXR sequence export supports CK-Alpha, CK-FG, CK-Comp (float/RGB layers as appropriate).

### Main Files for Corridor Key

| File | Role |
|------|------|
| **`corridorkey/`** (new directory) | CorridorKey integration |
| `corridorkey/__init__.py` | Package init |
| `corridorkey/corridorkey_manager.py` | Manager: HuggingFace download, load/unload engine, run per-frame, save 16-bit PNG + 8-bit alpha copy |
| `corridorkey/inference_engine.py` | Inference: load checkpoint, refiner (full-frame or tiled), despill, despeckle; `torch.compile` disabled on Linux+Qt to avoid segfaults |
| `corridorkey/core/model_transformer.py` | Model (Hiera) and refiner architecture |
| `corridorkey/core/color_utils.py` | Color/despill utilities |

### Files Modified to Support Corridor Key

| File | Changes |
|------|---------|
| `sammie_main.py` | New **CorridorKeyTab** and Sidebar tab "Corridor Key"; `run_corridorkey()` / `clear_corridorkey()`; view combo + CK-Alpha/CK-FG/CK-Comp; tab change switches view to CK-Alpha when Corridor Key tab selected; Quality/Tiling from settings; VRAM cleanup before load; `clear_corridorkey` resets view to Segmentation-Edit if current view was CK-* |
| `sammie/settings_manager.py` | `default_corridorkey_*` (mask_source, refiner_scale, despill, despeckle, despeckle_size, quality, tiling); SessionSettings + persistence |
| `sammie/sammie.py` | `_load_ck_output()`, `_handle_ck_alpha_view`, `_handle_ck_fg_view`, `_handle_ck_comp_view` (16-bit PNG read, normalize for display/export); object_id_filter for single-object export; DeviceManager `clear_cache()` with `gc.collect()` for VRAM |
| `sammie/export_formats.py` | `get_available_output_types()` extended with **CK-Alpha**, **CK-FG**, **CK-Comp** for all video and sequence formats (ProRes, FFV1, H.264, H.265, VP9, PNG seq, EXR seq) |
| `sammie/export_workers.py` | Export for CK-*: 16-bit PNG handling, EXR layers for CK-FG/CK-Comp; `has_alpha` limited to Segmentation-Alpha/Matting-Alpha so CK-Alpha is not treated as RGBA alpha channel |

---

## 3. Other Modifications (Stability, UX, Install)

### Requirements and Install

| File | Change |
|------|--------|
| `requirements.txt` | Pinned `antlr4-python3-runtime==4.9.3` for Hydra/SAM2 compatibility (avoids ATN deserialization segfault); added `einops`, `huggingface_hub`, `timm`, `transformers` as needed by new models. |

### Launch Script (Linux)

| File | Change |
|------|--------|
| `run_sammie.command` | On Linux: `QT_QPA_PLATFORM=xcb`, `QT_X11_NO_MITSHM=1` to reduce Qt/OpenGL segfaults; `CORRIDORKEY_OPT_MODE=lowvram` so CorridorKey does not enable torch.compile (which can segfault with Qt on Linux). |

### UI / Behavior

- **Sidebar:** minimum width increased; tab content wrapped in `QScrollArea` so Corridor Key (and other tabs) are scrollable when the window is small; `on_tab_changed` unwraps scroll area to identify active tab.
- **Main window:** default size and main splitter proportions adjusted so the Corridor Key tab is not clipped; optional centering on screen; minimum window size set.
- **VRAM:** `gc.collect()` before `torch.cuda.empty_cache()` in DeviceManager, CorridorKey and VideoMaMa load/unload paths to reduce fragmentation and OOM on heavy use.

### Optional / Not in Repo

- `install_dependencies_flame.sh`: custom installer for Autodesk Flame (Python path, CUDA choice). Can stay local or be contributed separately.
- `clean_for_git.sh`, `GIT_SUBMIT.md`: ignored via `.gitignore` (local workflow only).
- `.gitignore`: `checkpoints/` fully ignored; `clean_for_git.sh`, `GIT_SUBMIT.md` ignored.

---

## 4. File Tree Summary

```
Sammie-Roto-2/
├── sammie_main.py          # +622 lines (VideoMaMa + Corridor Key UI, signals, run/clear)
├── run_sammie.command      # + Linux Qt env vars and CORRIDORKEY_OPT_MODE
├── requirements.txt        # + antlr4-python3-runtime, einops, huggingface_hub, timm, transformers
├── README.md               # Updated: VideoMaMa + CorridorKey, acknowledgements, links
├── videomama/              # NEW
│   ├── __init__.py
│   ├── videomama_manager.py
│   └── pipeline_svd_mask.py
├── corridorkey/            # NEW
│   ├── __init__.py
│   ├── corridorkey_manager.py
│   ├── inference_engine.py
│   └── core/
│       ├── __init__.py
│       ├── color_utils.py
│       └── model_transformer.py
└── sammie/
    ├── sammie.py           # + CK view handlers, 16-bit load, DeviceManager gc
    ├── settings_manager.py # + VideoMaMa and CorridorKey session/app defaults
    ├── export_formats.py   # + CK-Alpha, CK-FG, CK-Comp in output types
    └── export_workers.py   # + 16-bit PNG/EXR for CK-*; has_alpha fix
```

---

## 5. References

- **VideoMaMa:** [https://github.com/cvlab-kaist/VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)  
- **EZ-CorridorKey:** [https://github.com/edenaion/EZ-CorridorKey](https://github.com/edenaion/EZ-CorridorKey)  
- **CorridorKey (upstream):** [https://github.com/nikopueringer/CorridorKey](https://github.com/nikopueringer/CorridorKey)  

---

This document can be shared with the upstream maintainer for review. If a Pull Request is not possible, it can be sent as a summary of changes for manual consideration.
