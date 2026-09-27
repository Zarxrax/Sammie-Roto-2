# Segmentation engines

Add a package folder under `segmentation/` with `__init__.py` and `plugin.py`.
The application discovers its `ENGINES` sequence of `segmentation.registry.EngineSpec`
objects at startup and adds them to the Segmentation model selector. Each spec
provides a stable ID, label, and `load_predictor(device, parent_window)` callable.
The callable returns a predictor compatible with the shared prompt and tracking
operations in `segmentation.base.SamManager`, or `None` if loading is cancelled.

Checkpoint downloads belong in an optional `downloads.py` module exporting a
`DOWNLOADS` mapping of keys to `DownloadSpec` values. The shared downloader
discovers these mappings automatically.

`segmentation/sam2` contains the SAM2 and EfficientTAM variants and their
upstream implementation and licenses in `vendor/`. Existing checkpoint paths
under `checkpoints/` are preserved.
