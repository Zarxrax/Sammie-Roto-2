"""Model downloads and gated-model authentication for LTX 2.5 Alpha Gen."""

from html import escape
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from sammie.model_downloader import DownloadSpec, ensure_models as ensure_downloads


MODEL_DIR = Path("checkpoints/ltx2.5")
VIDEO_VAE = MODEL_DIR / "ltx-2.5-video-vae-bf16.safetensors"
ALPHA_LORA = MODEL_DIR / "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors"
EMPTY_CONTEXT = MODEL_DIR / "ltx25_empty_lossless.safetensors"
ALPHA_REPO_URL = "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen"
HF_TOKEN_URL = "https://huggingface.co/settings/tokens/new?tokenType=read"
GGUF_REPO_URL = "https://huggingface.co/realrebelai/LTX-2.5_GGUFs"

# File sizes are those published by the repository. Memory estimates include
# the selected transformer, Alpha Gen LoRA, VAE stage, and normal model
# overhead, but exclude resolution/chunk-dependent activations.
QUANTS = {
    "Q2_K": {
        "filename": "LTX-2.5-Distilled-Q2_K.gguf", "size": "8.83 GB / 8.2 GiB",
        "base_peak": "~12 GiB", "recommended": "16+ GiB",
        "quality": "Smallest. Strongest quality loss; fine edges and stable alpha detail may degrade.",
    },
    "Q3_K_M": {
        "filename": "LTX-2.5-Distilled-Q3_K_M.gguf", "size": "11.5 GB / 10.7 GiB",
        "base_peak": "~15 GiB", "recommended": "20+ GiB",
        "quality": "Low memory, with visible precision loss compared with Q4 and higher.",
    },
    "Q4_K_S": {
        "filename": "LTX-2.5-Distilled-Q4_K_S.gguf", "size": "13.9 GB / 12.9 GiB",
        "base_peak": "~17 GiB", "recommended": "24+ GiB",
        "quality": "Good compromise when Q4_K_M is slightly too large.",
    },
    "Q4_K_M": {
        "filename": "LTX-2.5-Distilled-Q4_K_M.gguf", "size": "15.1 GB / 14.1 GiB",
        "base_peak": "~18 GiB", "recommended": "24+ GiB",
        "quality": "Recommended balance of alpha quality, speed, and memory.",
    },
    "Q5_K_M": {
        "filename": "LTX-2.5-Distilled-Q5_K_M.gguf", "size": "16.8 GB / 15.6 GiB",
        "base_peak": "~20 GiB", "recommended": "28+ GiB",
        "quality": "Better weight precision, with a smaller quality gain than the added memory cost.",
    },
    "Q6_K": {
        "filename": "LTX-2.5-Distilled-Q6_K.gguf", "size": "18.7 GB / 17.4 GiB",
        "base_peak": "~22 GiB", "recommended": "32+ GiB",
        "quality": "High precision and close to Q8, but substantially larger than Q4.",
    },
    "Q8_0": {
        "filename": "LTX-2.5-Distilled-Q8_0.gguf", "size": "23.6 GB / 22.0 GiB",
        "base_peak": "~27 GiB", "recommended": "36+ GiB",
        "quality": "Highest precision here; largest download and often little visible gain over Q6.",
    },
}

DOWNLOADS = {
    "ltx_alpha_video_vae": DownloadSpec(
        url=("https://huggingface.co/comfyicu/LTX-2.5/resolve/main/"
             "vae/ltx-2.5-video-vae-bf16.safetensors"),
        md5=None, dest_dir=str(MODEL_DIR)),
    "ltx_alpha_lora": DownloadSpec(
        url=("https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen/resolve/main/"
             "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors"),
        md5=None, dest_dir=str(MODEL_DIR), gated=True),
}


def _quant_path(quant):
    return MODEL_DIR / QUANTS[quant]["filename"]


def _selected_transformer():
    from sammie.settings_manager import get_settings_manager

    selected = get_settings_manager().get_app_setting("ltx_alpha_transformer_quant", "Q4_K_M")
    if selected in QUANTS and _quant_path(selected).is_file():
        return _quant_path(selected)
    # Preserve an already downloaded model after upgrades, even if no preference
    # was previously recorded. Prefer the old Q4_K_M default when it exists.
    for quant in ("Q4_K_M", *QUANTS.keys()):
        if quant in QUANTS and _quant_path(quant).is_file():
            get_settings_manager().set_app_setting("ltx_alpha_transformer_quant", quant)
            return _quant_path(quant)
    return None


def _request_transformer_quant(parent=None):
    """Let the user select one public GGUF and return its quant key."""
    from sammie.settings_manager import get_settings_manager

    dialog = QDialog(parent)
    dialog.setWindowTitle("Choose LTX 2.5 transformer quantization")
    dialog.setMinimumWidth(680)
    layout = QVBoxLayout(dialog)

    intro = QLabel(
        "<p>Choose the transformer quantization that fits your GPU VRAM or Apple unified memory. "
        "Lower quants reduce the download and model memory, but discard more weight precision and "
        "can produce less stable mattes, weaker fine edges, or more artifacts.</p>"
        "<p><b>Memory figures are estimates.</b> Base peak includes the GGUF, Alpha Gen LoRA, VAE "
        "stage, and model overhead. Resolution and frames per chunk add activation memory; 1080p "
        "and long chunks may require considerably more than the recommendation.</p>"
        f"<p>Files are downloaded from <a href='{GGUF_REPO_URL}'>realrebelai/LTX-2.5_GGUFs</a>.</p>"
    )
    intro.setWordWrap(True)
    intro.setOpenExternalLinks(True)
    layout.addWidget(intro)

    choices = QComboBox()
    for quant, info in QUANTS.items():
        suffix = " — Recommended" if quant == "Q4_K_M" else ""
        choices.addItem(
            f"{quant}: {info['size']}, base peak {info['base_peak']}, "
            f"recommended {info['recommended']}{suffix}", quant)
    saved = get_settings_manager().get_app_setting("ltx_alpha_transformer_quant", "Q4_K_M")
    saved_index = choices.findData(saved)
    choices.setCurrentIndex(saved_index if saved_index >= 0 else choices.findData("Q4_K_M"))
    layout.addWidget(choices)

    details = QLabel()
    details.setWordWrap(True)
    layout.addWidget(details)

    def update_details(_index):
        quant = choices.currentData()
        info = QUANTS[quant]
        details.setText(
            f"<b>{quant}</b> — {escape(info['quality'])}<br>"
            f"Download: {info['size']} &nbsp; | &nbsp; Estimated base peak: {info['base_peak']} "
            f"&nbsp; | &nbsp; Suggested available memory: {info['recommended']}"
        )

    choices.currentIndexChanged.connect(update_details)
    update_details(choices.currentIndex())

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.button(QDialogButtonBox.Ok).setText("Download selected GGUF")
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec() != QDialog.Accepted:
        return None
    quant = choices.currentData()
    get_settings_manager().set_app_setting("ltx_alpha_transformer_quant", quant)
    return quant


def _request_alpha_download(parent=None):
    """Explain the gated-model choices and return whether to download it."""
    from huggingface_hub import get_token, login

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    destination = ALPHA_LORA.resolve()
    saved_login = get_token() is not None

    dialog = QDialog(parent)
    dialog.setWindowTitle("LTX Alpha Gen model access")
    dialog.setMinimumWidth(620)
    layout = QVBoxLayout(dialog)

    explanation = QLabel(
        "<p>The Alpha Gen IC-LoRA is gated by Lightricks and cannot be downloaded "
        "until your Hugging Face account has accepted its terms.</p>"
        "<p><b>Option 1 — Manual installation</b><br>"
        f"Open the <a href='{ALPHA_REPO_URL}'>Alpha Gen model page</a>, accept the terms, "
        f"download <code>{escape(ALPHA_LORA.name)}</code>, and place it at:</p>"
        f"<p><code>{escape(str(destination))}</code></p>"
        "<p><b>Option 2 — Automatic download</b><br>"
        "Accept the terms on the same model page, then create a "
        f"<a href='{HF_TOKEN_URL}'>read-only Hugging Face token</a> and paste it below. "
        "Sammie validates and saves it using Hugging Face's standard user login cache. "
        "The token is not saved in Sammie settings or in this project.</p>"
    )
    explanation.setWordWrap(True)
    explanation.setOpenExternalLinks(True)
    explanation.setTextInteractionFlags(explanation.textInteractionFlags())
    layout.addWidget(explanation)

    token_input = QLineEdit()
    token_input.setEchoMode(QLineEdit.Password)
    token_input.setClearButtonEnabled(True)
    token_input.setPlaceholderText(
        "Saved Hugging Face login found — leave blank to use it"
        if saved_login else "Paste a read-only Hugging Face token (hf_…)"
    )
    layout.addWidget(token_input)

    buttons = QDialogButtonBox()
    open_page = buttons.addButton("Open model page", QDialogButtonBox.ActionRole)
    open_folder = buttons.addButton("Open destination folder", QDialogButtonBox.ActionRole)
    download = buttons.addButton("Download automatically", QDialogButtonBox.AcceptRole)
    buttons.addButton(QDialogButtonBox.Cancel)
    open_page.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(ALPHA_REPO_URL)))
    open_folder.clicked.connect(
        lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(MODEL_DIR.resolve()))))
    download.clicked.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    while dialog.exec() == QDialog.Accepted:
        entered_token = token_input.text().strip()
        try:
            if entered_token:
                login(token=entered_token, add_to_git_credential=False)
                token_input.clear()
            elif get_token() is None:
                QMessageBox.warning(
                    dialog,
                    "Hugging Face login required",
                    "Paste a read-only Hugging Face token, or cancel and install the file manually.",
                )
                continue
            return True
        except Exception as exc:
            QMessageBox.warning(dialog, "Hugging Face login failed", str(exc))
    return False


def ensure_models(parent=None):
    """Download only the transformer, video VAE, and Alpha Gen adapter.

    A manually installed Alpha Gen adapter is reused. Authentication is needed
    only when that gated file is absent and must be downloaded.
    """
    if not EMPTY_CONTEXT.is_file():
        raise RuntimeError(
            f"Missing cached empty-prompt conditioning: {EMPTY_CONTEXT}. "
            "Copy the small ComfyUI lossless empty conditioning file here; it replaces the 12B Gemma text encoder."
        )
    # Keep the order explicit: small public VAE, user-selected public GGUF,
    # then the gated adapter.
    if not ensure_downloads(DOWNLOADS["ltx_alpha_video_vae"], parent=parent,
                            title="Downloading LTX 2.5 video VAE"):
        return False

    transformer = _selected_transformer()
    if transformer is None:
        quant = _request_transformer_quant(parent)
        if quant is None:
            return False
        info = QUANTS[quant]
        transformer_spec = DownloadSpec(
            url=f"{GGUF_REPO_URL}/resolve/main/{info['filename']}",
            md5=None,
            dest_dir=str(MODEL_DIR),
        )
        if not ensure_downloads(transformer_spec, parent=parent,
                                title=f"Downloading LTX 2.5 {quant} transformer"):
            return False

    if not ALPHA_LORA.is_file():
        if not _request_alpha_download(parent):
            return False
        if not ensure_downloads(DOWNLOADS["ltx_alpha_lora"], parent=parent,
                                title="Downloading LTX 2.5 Alpha Gen IC-LoRA"):
            return False
    return True


def paths():
    transformer = _selected_transformer()
    if transformer is None:
        raise FileNotFoundError("No supported LTX 2.5 transformer GGUF is installed")
    return transformer, VIDEO_VAE, ALPHA_LORA, EMPTY_CONTEXT
