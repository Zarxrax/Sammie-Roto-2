# Matting engines

Each engine lives in its own Python package folder under `matting/` (include
`__init__.py`). The application discovers folders with a `plugin.py` module at
startup and adds their `ENGINES` entries to
the model selector. No change to `sammie_main.py` is needed.

An engine folder supplies:

- `plugin.py`: an `ENGINES` sequence of `EngineSpec` values from
  `matting.registry`. Each spec has a stable `id`, a UI `label`, a
  `manager_factory`, and `instructions_html`. Optional fields define an
  engine settings widget, unsupported device types, display order, and a
  callback for saving engine-specific defaults.
- A manager derived from `matting.base.MattingManager`, with `BACKEND` equal to
  the spec ID and implementations of `load_matting_model()` and `run_matting()`.
  `run_matting()` returns `1` on success and `0` on cancellation or failure.
- If extra controls are needed, a QWidget factory whose widget implements
  `load_settings()`. The widget reads and writes session settings itself.
- If checkpoints must be downloaded, a `downloads.py` module exposing a
  `DOWNLOADS` mapping of keys to `DownloadSpec` values. The shared downloader
  discovers these mappings for its on-demand and all-model commands.

A minimal `plugin.py` looks like this:

```python
from matting.registry import EngineSpec
from .engine import MyManager

ENGINES = (
    EngineSpec(
        id="MyEngine",
        label="My Engine",
        manager_factory=MyManager,
        instructions_html="Add points, then run matting.",
    ),
)
```

`matting/matanyone` registers both MatAnyone variants from one folder because
they share inference code. Its upstream implementation and license are in
`matting/matanyone/vendor`. `matting/videomama` owns its pipeline, tiling code,
controls, and engine integration. Model weights stay in the existing
`checkpoints/` paths so downloaded files remain valid. Shared frame, mask, and progress helpers are
in `matting/base.py`. The `sammie/matting.py` module remains as a compatibility
import for existing application code.
