# Object removal engines

Add a package folder under `object_removal/` with `__init__.py` and `plugin.py`.
The application discovers its `ENGINES` sequence of
`object_removal.registry.EngineSpec` objects at startup and adds them to the
Object Removal method selector. A spec supplies a stable ID, label, manager
factory, settings widget factory, settings loader, and preview signal connector.
It can also declare unsupported device types and a callback that saves defaults.

Managers derive from `object_removal.base.RemovalManager` and implement
`run(points, parent_window)` and `unload()`. Plugin settings widgets read and
write session settings themselves. Optional `downloads.py` files export a
`DOWNLOADS` mapping discovered by the shared downloader.

`minimax/` owns the MiniMax-Remover implementation, controls, downloads, and
upstream code and license in `vendor/`. `opencv/` owns the OpenCV implementation
and controls. Existing checkpoint paths under `checkpoints/` are preserved.
