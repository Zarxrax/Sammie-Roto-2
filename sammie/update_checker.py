# sammie/update_checker.py
import logging
import re
from typing import Optional, Dict, Any, List
import requests
from packaging.version import parse, InvalidVersion
from PySide6.QtCore import QThread, Signal, QObject

logger = logging.getLogger(__name__)

GITHUB_REPO = "Zarxrax/Sammie-Roto-2"
API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases"


def extract_image_url(text: str) -> str:
    """Extracts the first markdown or HTML image URL from the text."""
    if not text:
        return ""
    md_match = re.search(r'!\[.*?\]\((.*?)\)', text)
    if md_match:
        return md_match.group(1)

    html_match = re.search(r'<img[^>]+src=["\'](.*?)["\']', text, flags=re.IGNORECASE)
    if html_match:
        return html_match.group(1)

    return ""


def clean_release_notes(text: str) -> str:
    """Strips HTML comments, image tags, markdown images, and excessive blank lines from release notes."""
    if not text:
        return "No release notes available."

    # Strip HTML comments (e.g., <!-- ... -->)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Strip HTML image tags (e.g., <img ... />)
    text = re.sub(r'<img[^>]*>', '', text, flags=re.IGNORECASE)

    # Strip standard Markdown images: ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Clean up excessive empty lines left behind by stripped images
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def check_for_updates(current_version: str, repo: str = GITHUB_REPO, timeout: int = 5) -> Dict[str, Any]:
    """Check GitHub releases API and return aggregated update details if newer versions exist."""
    try:
        url = f"https://api.github.com/repos/{repo}/releases"
        headers = {"Accept": "application/vnd.github+json"}

        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        data = resp.json()
        if not data or not isinstance(data, list):
            return {"update_available": False, "current_version": current_version}

        current_clean = str(current_version).lstrip("v").strip()
        try:
            current_parsed = parse(current_clean)
        except InvalidVersion as exc:
            logger.error("Invalid current version format: %s (%s)", current_clean, exc)
            return {"update_available": False, "current_version": current_version, "error": f"Invalid current version: {exc}"}

        # Find all valid, non-draft releases newer than current_version
        newer_releases: List[Dict[str, Any]] = []
        for rel in data:
            if rel.get("draft", False):
                continue
            tag_raw = rel.get("tag_name", "").strip()
            tag_clean = tag_raw.lstrip("v").strip()
            if not tag_clean:
                continue
            try:
                rel_parsed = parse(tag_clean)
                if rel_parsed > current_parsed:
                    newer_releases.append(rel)
            except (InvalidVersion, TypeError):
                continue

        if not newer_releases:
            return {
                "update_available": False,
                "current_version": current_version,
                "latest_version": current_clean,
            }

        # The first release in the list is the latest
        latest_release = newer_releases[0]
        latest_tag = latest_release.get("tag_name", "").strip()
        latest_version_clean = latest_tag.lstrip("v").strip()

        # Aggregate release notes for all uninstalled releases newer than current
        aggregated_notes = []
        for rel in newer_releases:
            tag = rel.get("tag_name", "")
            raw_body = rel.get("body", "")
            body = clean_release_notes(raw_body)
            aggregated_notes.append(f"## Version {tag}\n\n{body}")

        final_notes = "\n\n---\n\n".join(aggregated_notes)

        # Extract image from the latest release body if present
        image_bytes: Optional[bytes] = None
        img_url = extract_image_url(latest_release.get("body", ""))
        if img_url:
            try:
                img_resp = requests.get(img_url, timeout=timeout)
                if img_resp.status_code == 200:
                    image_bytes = img_resp.content
            except Exception as e:
                logger.warning("Failed to fetch release image: %s", e)

        return {
            "update_available": True,
            "current_version": current_version,
            "latest_version": latest_version_clean,
            "latest_tag": latest_tag,
            "url": f"https://github.com/{repo}/releases",
            "notes": final_notes,
            "image_bytes": image_bytes,
            "uninstalled_count": len(newer_releases),
        }

    except requests.RequestException as exc:
        logger.warning("Update check network error: %s", exc)
        return {"update_available": False, "current_version": current_version, "error": str(exc)}
    except Exception as exc:
        logger.exception("Unexpected error during update check.")
        return {"update_available": False, "current_version": current_version, "error": str(exc)}


class UpdateCheckWorker(QThread):
    """Thread-safe background worker for checking GitHub updates."""
    result = Signal(dict)

    def __init__(self, current_version: str, silent: bool = True, repo: str = GITHUB_REPO, timeout: int = 5):
        super().__init__()
        self.current_version = current_version
        self.silent = silent
        self.repo = repo
        self.timeout = timeout

    def run(self):
        update_info = check_for_updates(
            current_version=self.current_version,
            repo=self.repo,
            timeout=self.timeout
        )
        self.result.emit({"data": update_info, "silent": self.silent})


class UpdateChecker(QObject):
    """Backward-compatible wrapper for checking updates."""
    update_available = Signal(str, str)  # current_version, latest_version

    def __init__(self):
        super().__init__()
        self._worker: Optional[UpdateCheckWorker] = None

    def check_for_updates(self, repo: str = GITHUB_REPO, timeout: int = 5, current_version: Optional[str] = None):
        if current_version is None:
            try:
                from sammie_main import __version__ as current_version
            except Exception:
                current_version = "0.0.0"

        self._worker = UpdateCheckWorker(
            current_version=current_version,
            silent=True,
            repo=repo,
            timeout=timeout
        )
        self._worker.result.connect(self._on_worker_result)
        self._worker.start()

    def _on_worker_result(self, payload: dict):
        data = payload.get("data", {})
        if data.get("update_available"):
            self.update_available.emit(data.get("current_version", ""), data.get("latest_version", ""))
