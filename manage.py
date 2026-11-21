import os, sys, subprocess, platform, tomllib, shutil, shlex, re, time
import urllib.request

# ===== CONFIG =====
PYTHON_VERSION = "3.12"
REPO_URL = "https://github.com/Zarxrax/Sammie-Roto-2.git"
REPO_RAW_BASE = "https://raw.githubusercontent.com/Zarxrax/Sammie-Roto-2"
DEFAULT_BRANCH = "main"

def get_uv_exe():
    """Absolute path to the uv executable this installer bootstrapped with."""
    app_dir = os.path.abspath(os.path.dirname(__file__))
    exe_name = "uv.exe" if platform.system() == "Windows" else "uv"
    return os.path.join(app_dir, ".uv", exe_name)

def get_uv_env():
    """Environment for every uv subprocess manage.py invokes itself.
    Clears VIRTUAL_ENV so uv's internal build-isolation environment
    doesn't leak into child processes."""
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    return env

def fix_permissions():
    """Make launch scripts executable and clear download quarantine (macOS)."""
    if platform.system() == "Windows":
        return
    app_dir = os.path.abspath(os.path.dirname(__file__))

    for name in ("run_sammie.sh", "install.sh"):
        path = os.path.join(app_dir, name)
        if os.path.exists(path):
            try:
                os.chmod(path, os.stat(path).st_mode | 0o755)
            except OSError as e:
                print(f"[Warning: could not chmod {name}: {e}]")

    if platform.system() == "Darwin":
        # Skip the big generated folders; only the extracted download can be quarantined.
        skip = {".venv", ".uv", ".git"}
        for entry in os.listdir(app_dir):
            if entry in skip:
                continue
            subprocess.run(
                ["xattr", "-dr", "com.apple.quarantine", os.path.join(app_dir, entry)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )

def cleanup_cache():
    """After a successful install/reinstall/update, prunes the project cache."""
    run_command([get_uv_exe(), "cache", "prune"])

# ===== UTILS =====
def ask(prompt, default="", valid=None):
    """input() that re-prompts until the answer is one of `valid` (if given).
    Pressing Enter returns `default`. Answers are stripped and lowercased."""
    while True:
        answer = input(prompt).strip().lower() or default
        if valid is None or answer in valid:
            return answer
        print(f"Please enter one of: {', '.join(valid)}")

def ask_yes_no(prompt, default):
    """Yes/no question. Accepts y/yes/n/no; Enter returns `default` (True/False)."""
    hint = "Y/n" if default else "y/N"
    answer = ask(f"{prompt} ({hint}): ", "y" if default else "n", ["y", "yes", "n", "no"])
    return answer in ("y", "yes")

def run_command(cmd, extra_env=None):
    """Wrapper to handle uv commands. `extra_env` adds/overrides environment variables."""
    print(">", " ".join(cmd))
    env = get_uv_env()
    env.update(extra_env or {})
    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\nError executing command: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: 'uv' command not found. Please execute install.bat/install.sh")
        sys.exit(1)

def get_local_version():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
        return data["project"]["version"]

def get_remote_version(branch):
    raw_pyproject_url = f"{REPO_RAW_BASE}/{branch}/pyproject.toml"
    try:
        with urllib.request.urlopen(raw_pyproject_url) as response:
            data = tomllib.loads(response.read().decode())
            return data["project"]["version"]
    except Exception as e:
        print(f"[Warning: Could not check remote version: {e}]")
        return None

def parse_version(v):
    """Splits a dotted version string into a tuple of ints,
    e.g. '1.10.2' -> (1, 10, 2)."""
    parts = []
    for segment in v.strip().split("."):
        digits = ""
        for ch in segment:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits) if digits else 0)
    return tuple(parts)

def is_newer_version(remote_v, local_v):
    """Returns True if remote_v is a newer release than local_v."""
    r, l = parse_version(remote_v), parse_version(local_v)
    length = max(len(r), len(l))
    r += (0,) * (length - len(r))
    l += (0,) * (length - len(l))
    return r > l

def get_installed_backend():
    """Detects which torch extra is currently installed (used for updates)."""
    if platform.system() == "Darwin":
        return "macos"

    if not os.path.exists(".venv"):
        return None

    try:
        result = subprocess.check_output(
            [get_uv_exe(), "pip", "show", "torch", "--python", ".venv"], 
            encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL, env=get_uv_env()
        )
        version_line = next((l for l in result.splitlines() if l.startswith("Version:")), "").lower()
        for backend in ["cu130", "cu126", "rocm", "xpu", "cpu"]:
            if backend in version_line:
                return installed_rocm_backend() if backend == "rocm" else backend
    except (subprocess.CalledProcessError, StopIteration):
        print("[Warning: Could not detect installed backend]")
        pass
    
    return None

def installed_rocm_backend():
    """'rocm-gfxNNNN' for the GPU installed in .venv, read from its amd-torch-device-* package."""
    try:
        out = subprocess.check_output(
            [get_uv_exe(), "pip", "freeze", "--python", ".venv"],
            encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL, env=get_uv_env())
    except (subprocess.CalledProcessError, OSError):
        out = ""
    for line in out.lower().replace("_", "-").splitlines():
        match = re.match(r"amd-torch-device-(gfx[0-9a-f]+)==", line)
        if match:
            return f"rocm-{match.group(1)}"
    print("[Warning: Could not determine which AMD GPU type is installed]")
    return None

# ===== GIT LOGIC =====
def init_git_tracking():
    """Initializes git tracking for the install (adds an 'origin' remote
    pointing at REPO_URL) without fetching or touching any local files.
    Safe to call any time — does nothing if .git already exists."""
    from dulwich.repo import Repo
    from dulwich import porcelain

    if os.path.exists(".git"):
        return

    print("[Initializing Git tracking...]")
    repo = Repo.init(".")
    porcelain.remote_add(repo, "origin", REPO_URL)

def pull_latest_code(branch):
    """Ensures the local files exactly match the given branch of the repository"""
    from dulwich import porcelain
    from dulwich.repo import Repo

    init_git_tracking()
    repo = Repo(".")

    print(f"[Fetching latest code from GitHub ({branch} branch)...]")
    porcelain.fetch(repo, "origin")

    print("[Restoring all program files to original state...]")
    porcelain.reset(repo, "hard", f"origin/{branch}")

    # The hard reset can rewrite files with their git-stored mode, so
    # re-apply the exec bit and clear quarantine.
    fix_permissions()

# ===== BACKEND SELECTION =====
BACKEND_OPTIONS = [
    ("cu130", "NVIDIA CUDA 13.0 (RTX, newer GPUs)"),
    ("cu126", "NVIDIA CUDA 12.6 (GTX, older GPUs)"),
    ("xpu",   "Intel Arc/Xe (XPU)"),
    ("rocm",  "AMD ROCm (Radeon)"),
    ("cpu",   "CPU (Slow)"),
]
BACKEND_LABELS = dict(BACKEND_OPTIONS)

# ----- AMD ROCm -----
# ROCm has its own uv project in rocm/ (see generate_rocm_project.py) so that AMD's package index
# never affects the other backends. It installs into the same .venv, and its backend keys include
# the GPU, e.g. "rocm-gfx1100".
ROCM_DIR = "rocm"
ROCM_BOOTSTRAP = "rocm-bootstrap==0.3.0"   # AMD's GPU detector

# (gfx target, GPUs) from the table in https://github.com/ROCm/TheRock/blob/main/RELEASES.md.
# generate_rocm_project.py reads this list too, so after editing it, regenerate rocm/.
ROCM_GPUS = [
    ("gfx1201", "Radeon RX 9070 / 9070 XT, AI PRO R9700 / R9600D"),
    ("gfx1200", "Radeon RX 9060 / 9060 XT"),
    ("gfx1100", "Radeon RX 7900 XTX / 7900 XT, PRO W7900 / W7800"),
    ("gfx1101", "Radeon RX 7800 XT / 7700 XT, PRO V710 / W7700"),
    ("gfx1102", "Radeon RX 7600"),
    ("gfx1151", "Ryzen AI Max+ PRO 395"),
    ("gfx1150", "Ryzen AI 9 HX 375"),
    ("gfx1152", "Ryzen AI 7 350"),
    ("gfx1153", "Radeon 820M integrated graphics"),
    ("gfx1103", "Ryzen 7 7840U / Ryzen 9 270"),
    ("gfx1030", "Radeon RX 6900 XT / 6800 XT, PRO W6800 / V620"),
    ("gfx1031", "Radeon RX 6750 XT / 6700 XT"),
    ("gfx1032", "Radeon RX 6600 XT / 6600, PRO W6600"),
    ("gfx1034", "Radeon RX 6500 XT"),
    ("gfx1035", "Radeon 680M integrated graphics"),
    ("gfx1036", "Ryzen 7000 desktop integrated graphics (Raphael)"),
    ("gfx1033", "Van Gogh integrated graphics (Steam Deck)"),
    ("gfx1010", "Radeon RX 5700 / 5700 XT"),
    ("gfx1011", "Radeon Pro V520"),
    ("gfx1012", "Radeon Pro W5500"),
]
ROCM_GPU_NAMES = dict(ROCM_GPUS)
# Suggested last when several AMD GPUs are found (built-in graphics plus a Radeon card).
ROCM_IGPU_TARGETS = {"gfx1033", "gfx1035", "gfx1036", "gfx1103", "gfx1150", "gfx1152", "gfx1153"}


def backend_label(backend):
    """Name shown for a backend key; ROCm keys also show their GPU."""
    if backend.startswith("rocm-"):
        target = backend.removeprefix("rocm-")
        return f"AMD ROCm ({target}: {ROCM_GPU_NAMES.get(target, target)})"
    return BACKEND_LABELS.get(backend, backend)


def rocm_targets_available():
    """GPU targets that rocm/pyproject.toml can install."""
    try:
        with open(os.path.join(ROCM_DIR, "pyproject.toml"), "rb") as f:
            extras = tomllib.load(f)["project"]["optional-dependencies"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return set()
    return {e.removeprefix("rocm-") for e in extras if e.startswith("rocm-")}


def detect_amd_targets():
    """gfx targets found by AMD's detector. Empty if there is no AMD GPU or the detector can't run."""
    cmd = [get_uv_exe(), "tool", "run", "--python", PYTHON_VERSION, "--python-preference", "only-managed",
           "--from", ROCM_BOOTSTRAP, "rocm-bootstrap-detect", "--unique"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=180, env=get_uv_env())
    except (OSError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []
    lines = [line.strip() for line in r.stdout.splitlines()]
    return [line for line in lines if re.fullmatch(r"gfx[0-9a-f]+", line)]


def choose_rocm_target(targets, default=None):
    """Asks which of `targets` is the user's GPU. Returns a gfx target, or None for 'my GPU isn't listed'."""
    rows = [(t, name) for t, name in ROCM_GPUS if t in targets]
    print("\nSelect your AMD GPU:")
    for i, (target, name) in enumerate(rows, 1):
        print(f"{i}) {name} [{target}]" + ("  <- detected" if target == default else ""))
    not_listed = len(rows) + 1
    print(f"{not_listed}) My GPU isn't listed (use CPU instead)")

    default_number = next((str(i) for i, (target, _) in enumerate(rows, 1) if target == default), "")
    answer = ask(f"> [{default_number}] " if default_number else "> ", default_number,
                 [str(i) for i in range(1, not_listed + 1)])
    number = int(answer)
    return rows[number - 1][0] if number < not_listed else None


def choose_rocm_backend():
    """Turns the 'AMD ROCm' menu choice into a backend key such as 'rocm-gfx1100'."""
    available = rocm_targets_available()
    if not available:
        print(f"\n[The AMD ROCm install files (the '{ROCM_DIR}' folder) are missing from this download.]")
        print("Please download the application again, or choose a different backend.")
        sys.exit(1)

    print("\nIdentifying your AMD GPU...")
    found = detect_amd_targets()
    usable = [t for t in found if t in available]
    if len(usable) == 1:
        print(f"Detected: {ROCM_GPU_NAMES.get(usable[0], usable[0])} [{usable[0]}]")
        return f"rocm-{usable[0]}"

    if usable:
        print("Several AMD GPUs were detected.")
    elif found:
        print(f"[Your AMD GPU ({', '.join(found)}) is not supported by the ROCm builds.]")
    else:
        print("[Could not identify your AMD GPU automatically.]")
    discrete = [t for t in usable if t not in ROCM_IGPU_TARGETS]
    default = (discrete or usable or [None])[0]
    target = choose_rocm_target(usable or available, default)
    if target is None:
        print("[Using CPU instead.]")
        return "cpu"
    return f"rocm-{target}"


# ----- Hardware recommendation (printed above the backend menu) -----
# Prints nothing when it can't be confident.

# nvidia-smi is not always on PATH, so also look in its usual install locations.
NVIDIA_SMI_FALLBACKS = [
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",   # older Windows drivers
    "/usr/lib/wsl/lib/nvidia-smi",                                  # WSL2
]


def _run(cmd):
    """Runs a command and returns its stdout, or '' on any failure."""
    try:
        return subprocess.check_output(cmd, text=True, errors="replace",
                                       stderr=subprocess.DEVNULL, timeout=15)
    except Exception:
        return ""


def nvidia_smi_recommendation():
    """Step 1: ask the NVIDIA driver (nvidia-smi) for each GPU's compute capability.
    The newest GPU decides. Returns (backend_key, message), or None if nvidia-smi
    is missing or can't answer."""
    smi = shutil.which("nvidia-smi") or next((p for p in NVIDIA_SMI_FALLBACKS if os.path.exists(p)), None)
    if not smi:
        return None
    gpus = []
    for line in _run([smi, "--query-gpu=name,compute_cap", "--format=csv,noheader"]).splitlines():
        name, _, cc = line.rpartition(",")
        try:
            if name.strip():
                gpus.append((name.strip(), float(cc)))
        except ValueError:   # e.g. an error message, or a field an old driver doesn't support
            pass
    if not gpus:
        return None
    name, cc = max(gpus, key=lambda g: g[1])
    if cc < 5.0:
        return "cpu", f"{name} is too old for the CUDA builds of PyTorch. CPU is recommended."
    if cc < 7.5:   # Maxwell / Pascal / Volta were dropped in CUDA 13
        return "cu126", f"{name} detected. Recommended: NVIDIA CUDA 12.6 (older GPUs)."
    return "cu130", f"{name} detected. Recommended: NVIDIA CUDA 13.0."


def detect_gpus():
    """Step 2: list [(name, pci_vendor_id)] from Win32_VideoController (Windows) or lspci (Linux).
    The vendor ID (10de NVIDIA, 1002 AMD, 8086 Intel) is still known when a card has no driver."""
    gpus = []
    if platform.system() == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | "
                    "ForEach-Object { $_.Name + '|' + $_.PNPDeviceID }"])
        for line in out.splitlines():
            name, _, pnp = line.strip().rpartition("|")
            m = re.search(r"VEN_([0-9A-F]{4})", pnp, re.I)
            if name:
                gpus.append((name.strip(), m.group(1).lower() if m else None))
    elif platform.system() == "Linux":
        for line in _run(["lspci", "-nn"]).splitlines():
            if ": " not in line or not re.search(r"VGA|3D controller|Display controller", line):
                continue
            desc = line.split(": ", 1)[1]
            m = re.search(r"\[([0-9a-f]{4}):[0-9a-f]{4}\]", desc)
            name = re.sub(r"\s*(\[[0-9a-f]{4}:[0-9a-f]{4}\]|\(rev [0-9a-f]+\))", "", desc).strip()
            gpus.append((name, m.group(1) if m else None))
    return gpus


def nvidia_name_recommendation(names):
    """Fallback when nvidia-smi can't be used. Goes by the newest recognised GPU name.
    Returns (backend_key, message), or None unless the choice is certain."""
    new = [n for n in names if re.search(r"\brtx\b|\bgtx\s*16\d\d(?!\d)", n.lower())]   # Turing or newer
    if new:
        return "cu130", f"{new[0]} detected. Recommended: NVIDIA CUDA 13.0 (based on the GPU name)."
    # CUDA 12.6 only if every NVIDIA GPU is a recognised Maxwell/Pascal card; an unrecognised
    # name could be a newer GPU, so stay silent in that case.
    if all(re.search(r"\bgtx\s*(9\d\d|10\d\d|750)(?!\d)", n.lower()) for n in names):
        return "cu126", f"{names[0]} detected. Recommended: NVIDIA CUDA 12.6 (based on the GPU name)."
    return None


def hardware_recommendation():
    """Returns (backend_key, message), or (None, None) if it can't be made with confidence."""
    rec = nvidia_smi_recommendation()
    if rec:
        return rec
    gpus = detect_gpus()
    if not gpus:
        return None, None
    nvidia = [n for n, v in gpus if v == "10de" or "nvidia" in n.lower()]
    if nvidia:
        return nvidia_name_recommendation(nvidia) or (None, None)
    # A card with no driver shows up as "Basic Display Adapter": stay silent about those.
    real = [(n, v) for n, v in gpus if not re.search(r"basic display|standard vga", n, re.I)]
    amd = [n for n, v in real if v == "1002" or re.search(r"\bamd\b|radeon", n, re.I)]
    if amd:
        return "rocm", f"{amd[0]} detected. Recommended: AMD ROCm (your exact GPU is identified in the next step)."
    for n, _ in real:
        if re.search(r"\barc\b", n, re.I) and "intel" in n.lower():
            return "xpu", f"{n} detected. Recommended: Intel Arc (XPU)."
    if len(real) < len(gpus):
        return None, None
    return "cpu", (f"No supported GPU detected ({', '.join(n for n, _ in gpus)}). "
                   "CPU is recommended; processing will be slow.")


def choose_backend(installed=None):
    """Prompts for the hardware backend. Pressing Enter accepts the detected recommendation.
    `installed` is the currently installed backend (if known); it is shown so a reinstall
    can be used to correct an earlier wrong choice."""
    if platform.system() == "Darwin":
        return "macos"

    rec_key, rec_text = hardware_recommendation()
    if rec_text:
        print("\n--- Hardware Detection ---")
        print(rec_text)
        print("--------------------------")
    if installed:
        print(f"Currently installed: {backend_label(installed)}")

    keys = [key for key, _ in BACKEND_OPTIONS]
    print("\nSelect PyTorch backend:")
    for i, (key, label) in enumerate(BACKEND_OPTIONS, 1):
        print(f"{i}) {label}" + ("  <- recommended" if key == rec_key else ""))

    # With no confident recommendation there is no default, so Enter just re-prompts.
    default = str(keys.index(rec_key) + 1) if rec_key in keys else ""
    prompt = f"> [{default}] " if default else "> "
    choice = ask(prompt, default, [str(i) for i in range(1, len(keys) + 1)])
    key = keys[int(choice) - 1]
    return choose_rocm_backend() if key == "rocm" else key

def sync_env(backend, reinstall=False):
    """Uses uv sync to update or reinstall the environment."""
    cmd = [get_uv_exe(), "sync"]
    extra_env = {}
    if backend and backend.startswith("rocm-"):
        # The environment path must be absolute, or uv would put it inside rocm/.
        cmd += ["--project", ROCM_DIR]
        extra_env["UV_PROJECT_ENVIRONMENT"] = os.path.abspath(".venv")
    cmd += ["--frozen", "--python", PYTHON_VERSION, "--python-preference", "only-managed"]
    if backend:
        cmd.extend(["--extra", backend])

    label = backend_label(backend) if backend else "Default/MPS"
    if reinstall:
        print(f"\n[Reinstalling dependencies for {label}...]")
        # --reinstall refreshes all;
        cmd.extend(["--reinstall"])
    else:
        print(f"\n[Syncing dependencies for {label}...]")

    run_command(cmd, extra_env)

def create_shortcuts():
    """Creates/refreshes the appropriate desktop entry point for the current OS."""
    system = platform.system()
    if system == "Windows":
        create_windows_shortcut()
    elif system == "Darwin":
        create_mac_app()
    elif system == "Linux":
        create_linux_desktop_entry()

TORCH_PROBE = """
import torch, torchvision
print(f"torch {torch.__version__}, torchvision {torchvision.__version__}")
device = None
if torch.cuda.is_available():
    device = "cuda"         # ROCm reports through the CUDA API
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    device = "xpu"
elif torch.backends.mps.is_available():
    device = "mps"
if device is None:
    print("gpu=none")
else:
    try:
        x = torch.randn(512, 512, device=device)
        (x @ x).sum().item()    # a real GPU calculation, not just device detection
        print("gpu=ok " + (torch.cuda.get_device_name(0) if device == "cuda" else device))
    except Exception as e:
        reason = str(e).strip().splitlines()
        print("gpu=error " + (reason[0][:200] if reason else type(e).__name__))
tag = lambda v: v.partition("+")[2]    # the +cu130 / +cpu / +rocm... part
raise SystemExit(0 if tag(torch.__version__) == tag(torchvision.__version__) else 3)
"""

def report_gpu_result(backend, gpu):
    """Prints what the GPU test in TORCH_PROBE found. `gpu` is its 'gpu=...' output line."""
    if gpu.startswith("gpu=ok"):
        print(f"[GPU acceleration: working ({gpu.partition(' ')[2]})]")
        return
    if gpu.startswith("gpu=error"):
        print("[WARNING: A GPU was found, but a test calculation on it failed:]")
        print("  " + gpu.partition(" ")[2])
    else:
        print("[WARNING: GPU acceleration is NOT available, so the app will run on the CPU (slow).]")
    print("Make sure your graphics drivers are up to date. If the problem continues, run the")
    print("installer again, choose Reinstall/Repair, and try a different backend.")
    if backend.startswith("rocm-"):
        print(f"For AMD, also check that {backend.removeprefix('rocm-')} is the right type for your GPU.")
        if platform.system() == "Linux":
            print("On Linux, your user must also be in the 'render' and 'video' groups.")

def verify_torch(backend=None):
    """Checks that torch and torchvision import and are the same build, and prints the result.
    For a GPU backend it also runs a small calculation on the GPU and reports whether it worked."""
    win = platform.system() == "Windows"
    py = os.path.join(".venv", "Scripts" if win else "bin", "python.exe" if win else "python")
    print("\n[Verifying PyTorch...]")
    try:
        r = subprocess.run([py, "-c", TORCH_PROBE], capture_output=True, text=True, errors="replace",
                           timeout=180, env=get_uv_env())
    except subprocess.TimeoutExpired:
        print("[WARNING: The PyTorch check timed out, so PyTorch could not be verified.]")
        return
    except Exception as e:
        print(f"[WARNING: Could not run the PyTorch check ({e}).]")
        return
    out = r.stdout.splitlines()
    versions = next((line for line in out if line.startswith("torch ")), "")
    if r.returncode == 0:
        print(f"[PyTorch OK: {versions}]")
        if backend and backend != "cpu":
            report_gpu_result(backend, next((line for line in out if line.startswith("gpu=")), "gpu=none"))
        return
    if r.returncode == 3:
        print(f"[WARNING: torch and torchvision are different builds ({versions}).]")
    else:
        print("[WARNING: torch and torchvision failed to load together:]")
        print("\n".join(r.stderr.strip().splitlines()[-5:]))
    print("The app will probably not work correctly. Run the installer again and choose")
    print("Reinstall/Repair. If the problem continues, please report the message above.")

def resolve_backend():
    """Returns the currently installed backend, or prompts the user to pick
    one if it can't be detected (e.g. after a failed/partial install)."""
    backend = get_installed_backend()
    if not backend:
        if platform.system() != "Darwin":
            print("[Could not determine your previously installed backend — please reselect it to continue.]")
        backend = choose_backend()
    return backend

def continue_in_new_manage_py(flag):
    """Runs the newly pulled manage.py with `flag` and exits with its exit code.
    This process still has the old code in memory, so the new code has to do the rest."""
    sys.exit(subprocess.call([sys.executable, os.path.abspath(__file__), flag]))

def perform_update(branch):
    """Pulls the given branch, then lets the new manage.py finish the update.
    Shared by every update/recovery path in handle_update()."""
    pull_latest_code(branch)
    continue_in_new_manage_py("--finish-update")

def finish_update():
    """Second half of an update: resyncs the environment and refreshes shortcuts.
    Runs in the newly pulled manage.py, started by perform_update()."""
    backend = resolve_backend()
    sync_env(backend)
    create_shortcuts()
    cleanup_cache()
    verify_torch(backend)
    print("\nUpdate complete!")

# ===== CORE ACTIONS =====
def handle_update(branch, confirm=True):
    # Read the local version, with recovery if pyproject.toml is missing
    # or unreadable — a likely sign of a failed or partial install.
    try:
        local_v = get_local_version()
    except Exception as e:
        print(f"[Could not read local version: {e}]")
        print("[pyproject.toml may be missing or corrupt — this can happen after a failed install.]")
        if ask_yes_no("Download the latest program files to recover?", default=True):
            perform_update(branch)
        else:
            print("[No changes made. Consider using Reinstall/Repair from the main menu.]")
        return

    # Version numbers aren't a meaningful signal on a non-release branch --
    # always pull whatever's currently there instead of gating on them.
    if branch != DEFAULT_BRANCH:
        print(f"[Pulling latest '{branch}' branch...]")
        perform_update(branch)
        return

    remote_v = get_remote_version(branch)

    if remote_v is None:
        print("[Could not check for updates. Check your internet connection and try again.]")
        return

    if is_newer_version(remote_v, local_v):
        print(f"\nUpdate available: {remote_v} (current: {local_v})")
        if confirm and not ask_yes_no("Install update now?", default=True):
            print("Update skipped.")
            return
        perform_update(branch)
    else:
        print(f"[Already up to date (Version {local_v}).]")

def choose_branch():
    """Asks which branch to download program files from (--dev mode only)."""
    print("\nDownload from:")
    print("1) Main (stable)")
    print("2) Dev (latest, may be unstable)")
    return "dev" if ask("> [1] ", "1", ["1", "2"]) == "2" else DEFAULT_BRANCH

def repair(dev_mode=False):
    """Reinstall/Repair. If the user wants the latest program files, pulls them
    and lets the new manage.py do the repair. The dev branch is only offered
    in dev mode (--dev)."""
    if ask_yes_no("\nDownload the latest program files to restore any damaged or out of date files?", default=False):
        branch = choose_branch() if dev_mode else DEFAULT_BRANCH
        pull_latest_code(branch)
        continue_in_new_manage_py("--finish-repair")
    setup(reinstall=True)

def setup(reinstall=False):
    # Git tracking is initialized as soon as setup() runs (fresh install or
    # reinstall), so future "Check for Updates" runs can fetch/reset
    # cleanly. This only adds the 'origin' remote -- it never touches files.
    init_git_tracking()

    # -- Gather all choices upfront ----------------------------------------

    # 1. Backend selection. On reinstall, the current backend is shown so
    #    a wrong earlier choice can be corrected.
    backend = choose_backend(get_installed_backend() if reinstall else None)

    # 2. Model download -- fresh install only
    download_models_now = False
    if not reinstall:
        print("\nModel download:")
        print("1) Download models as needed (default -- models download the first time they are used)")
        print("2) Download all models now (~10GB)")
        download_models_now = ask("> [1] ", "1", ["1", "2"]) == "2"

    # -- Execute ------------------------------------------------------------
    # Always make scripts executable / clear quarantine, even if the
    # dependency install below fails partway.
    fix_permissions()

    run_command([get_uv_exe(), "python", "install", "--no-bin", PYTHON_VERSION])

    sync_env(backend, reinstall=reinstall)

    create_shortcuts()

    cleanup_cache()

    # Run the model downloader last so all dependencies are in place.
    if download_models_now:
        print("\nDownloading all models...")
        run_command([get_uv_exe(), "run", "--no-sync", os.path.join("sammie", "model_downloader.py")])

    verify_torch(backend)
    print("\nSetup Complete!")


# ===== CREATE SHORTCUTS =====
def create_mac_app():
    """Creates a double-clickable .app bundle on macOS."""

    app_dir = os.path.abspath(os.path.dirname(__file__))
    app_bundle = os.path.join(app_dir, "Sammie-Roto-2.app")
    macos_dir = os.path.join(app_bundle, "Contents", "MacOS")
    resources_dir = os.path.join(app_bundle, "Contents", "Resources")
    os.makedirs(macos_dir, exist_ok=True)
    os.makedirs(resources_dir, exist_ok=True)
    version = get_local_version()  # pulls from pyproject.toml

    src_icon = os.path.join(app_dir, "sammie", "resources", "icon.icns")
    dest_icon = os.path.join(resources_dir, "icon.icns")
    if os.path.exists(src_icon):
        shutil.copy(src_icon, dest_icon)

    # Launcher script
    launcher_path = os.path.join(macos_dir, "launcher")
    with open(launcher_path, "w") as f:
        f.write(
            '#!/usr/bin/env bash\n'
            f'cd {shlex.quote(app_dir)}\n'
            'exec /bin/bash ./run_sammie.sh\n'
        )
    os.chmod(launcher_path, os.stat(launcher_path).st_mode | 0o755)

    # Info.plist
    plist_path = os.path.join(app_bundle, "Contents", "Info.plist")
    with open(plist_path, "w") as f:
        f.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"'
            ' "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0">\n'
            '<dict>\n'
            '    <key>CFBundleName</key>\n'
            '    <string>Sammie-Roto-2</string>\n'
            '    <key>CFBundleIconFile</key>\n'
            '    <string>icon.icns</string>\n'
            '    <key>CFBundleExecutable</key>\n'
            '    <string>launcher</string>\n'
            '    <key>CFBundleIdentifier</key>\n'
            '    <string>com.zarxrax.sammie-roto-2</string>\n'
            '    <key>CFBundleVersion</key>\n'
            f'    <string>{version}</string>\n'
            '    <key>CFBundleShortVersionString</key>\n'
            f'    <string>{version}</string>\n'
            '    <key>CFBundlePackageType</key>\n'
            '    <string>APPL</string>\n'
            '</dict>\n'
            '</plist>\n'
        )

    # Clear quarantine flag so Gatekeeper doesn't block it
    try:
        subprocess.run(
            ["xattr", "-dr", "com.apple.quarantine", app_bundle],
            check=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        pass  # Not quarantined, nothing to clear

    print(f"Created {app_bundle}")

    # Create a symlink on the Desktop so the user has a convenient launch
    # point, mirroring the Windows/Linux shortcuts, without moving the
    # actual .app bundle (which must stay next to run_sammie.sh).
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if os.path.isdir(desktop):
        desktop_link = os.path.join(desktop, "Sammie-Roto-2.app")
        try:
            if os.path.islink(desktop_link) or os.path.exists(desktop_link):
                os.remove(desktop_link)
            os.symlink(app_bundle, desktop_link)
            print(f"Created Desktop shortcut at: {desktop_link}")
        except OSError as e:
            print(f"[Warning: Could not create Desktop shortcut: {e}]")

    # Also symlink into ~/Applications so Spotlight/Launchpad can find and
    # launch it like a normal installed app.
    user_apps = os.path.join(os.path.expanduser("~"), "Applications")
    try:
        os.makedirs(user_apps, exist_ok=True)
        apps_link = os.path.join(user_apps, "Sammie-Roto-2.app")
        if os.path.islink(apps_link) or os.path.exists(apps_link):
            os.remove(apps_link)
        os.symlink(app_bundle, apps_link)
        print(f"Added to your Applications folder: {apps_link}")
    except OSError as e:
        print(f"[Warning: Could not add to ~/Applications: {e}]")

    print("Double-click the Desktop icon to launch, or find it via Spotlight!")

def create_linux_desktop_entry():
    """Installs a menu entry and icon following the XDG conventions."""
    data_home = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    apps_dir = os.path.join(data_home, "applications")
    icon_dir = os.path.join(data_home, "icons", "hicolor", "256x256", "apps")
    os.makedirs(apps_dir, exist_ok=True)
    os.makedirs(icon_dir, exist_ok=True)

    app_dir = os.path.abspath(os.path.dirname(__file__))
    src_icon = os.path.join(app_dir, "sammie", "resources", "icon.png")
    run_sh = os.path.join(app_dir, "run_sammie.sh").replace("%", "%%")

    icon_name = "sammie-roto-2"
    if os.path.exists(src_icon):
        shutil.copy(src_icon, os.path.join(icon_dir, icon_name + ".png"))
    else:
        icon_name = src_icon  # fall back to the path (may not exist)

    content = "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        "Name=Sammie-Roto-2",
        "Comment=Video Rotoscoping and Masking Tool",
        f'Exec="{run_sh}" %f',
        f"Path={app_dir}",
        f"Icon={icon_name}",
        "Terminal=false",
        "StartupNotify=true",
        "Categories=AudioVideo;Video;AudioVideoEditing;Graphics;",
        "MimeType=video/mp4;video/x-matroska;video/quicktime;video/webm;image/png;image/jpeg;",
        "StartupWMClass=Sammie-Roto-2",
        "",
    ])
    desktop_path = os.path.join(apps_dir, "sammie-roto-2.desktop")
    with open(desktop_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(desktop_path, 0o644)
    print(f"Created application menu entry: {desktop_path}")

def create_windows_shortcut():
    """Creates a desktop shortcut on Windows."""
    app_dir = os.path.abspath(os.path.dirname(__file__))
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    shortcut_path = os.path.join(desktop, "Sammie-Roto-2.lnk")
    target = os.path.join(app_dir, "run_sammie.bat")
    icon = os.path.join(app_dir, "sammie", "resources", "icon.ico")

    ps_script = (
        f'$ws = New-Object -ComObject WScript.Shell;'
        f'$s = $ws.CreateShortcut("{shortcut_path}");'
        f'$s.TargetPath = "{target}";'
        f'$s.WorkingDirectory = "{app_dir}";'
        f'$s.IconLocation = "{icon}";'
        f'$s.Save()'
    )

    try:
        subprocess.check_call(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"Created Windows desktop shortcut at: {shortcut_path}")
    except subprocess.CalledProcessError as e:
        print(f"[Warning: Could not create Windows shortcut: {e}]")

# ===== ENTRY =====
def _lock_path():
    """Where launcher.py's QLockFile lives: QDir.tempPath() + '/sammie-roto.lock'.
    Qt uses TMP/TEMP on Windows and $TMPDIR (else /tmp) on Linux/macOS."""
    if platform.system() == "Windows":
        base = os.environ.get("TMP") or os.environ.get("TEMP") or os.environ.get("USERPROFILE") or ""
    else:
        base = os.environ.get("TMPDIR") or "/tmp"
    return os.path.join(base, "sammie-roto.lock")


def _pid_alive(pid):
    """True if a process with this PID exists. No subprocess, so no language/encoding problems."""
    if pid <= 0:
        return False   # os.kill(0, 0) would signal our whole process group
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5             # ERROR_ACCESS_DENIED: exists, not ours
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259                        # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)   # signal 0 only checks existence (never use this on Windows: it kills)
        return True
    except ProcessLookupError:
        return False      # Process doesn't exist -- stale lock
    except PermissionError:
        return True       # Process exists but we can't signal it -- assume running


def is_app_running():
    """Checks whether Sammie-Roto is currently running by reading the PID from Qt's
    lock file (written by launcher.py) and verifying that process is actually alive."""
    lock_path = _lock_path()
    if not os.path.exists(lock_path):
        return False
    try:
        # Qt also writes the process name, host name, etc. on later lines; only line 1 (the PID) matters
        with open(lock_path, "r", encoding="utf-8", errors="replace") as f:
            pid = int(f.readline().strip())
    except (ValueError, OSError):
        return False  # Unreadable or malformed -- treat as stale
    return _pid_alive(pid)


def main():
    if "--finish-update" in sys.argv[1:]:
        finish_update()
        return
    if "--finish-repair" in sys.argv[1:]:
        setup(reinstall=True)
        return
    if "--finish-install" in sys.argv[1:]:
        setup()
        return

    dev_mode = "--dev" in sys.argv[1:]
    # --update is passed by the GUI's "Exit and Update" button. The user already
    # confirmed there, so skip the menu and the prompt and go straight to the update.
    update_mode = "--update" in sys.argv[1:]

    # If app is not installed, bypass main menu and proceed to setup
    if not os.path.exists(".venv"):
        if os.path.exists("python-3.12.8-embed-amd64"):
            print("ERROR: It appears you are trying to install over an older Sammie-Roto installation.")
            print("Please delete the existing folder then extract the files to a new folder and try again.")
            sys.exit(1)

        if dev_mode and choose_branch() == "dev":
            # Get the dev code first, then let the newly pulled manage.py run the
            # install (this process still has the old code in memory).
            pull_latest_code("dev")
            continue_in_new_manage_py("--finish-install")

        setup()
        return

    if update_mode:
        # The GUI starts this script just before it exits; give it a few seconds
        # to finish saving and release its lock file.
        for _ in range(20):
            try:
                if not is_app_running():
                    break
            except Exception:
                break
            time.sleep(0.5)
        # The lock is released slightly before the process finishes exiting,
        # so give it another moment to let go of its files.
        time.sleep(1)


    try:
        running = is_app_running()
    except Exception:
        running = False   # advisory check only; never block install/repair because of it
    if running:
        print("\n[Warning: Sammie-Roto-2 appears to be running.]")
        print("[Please close it before continuing to avoid corrupting your installation.]")
        if not ask_yes_no("Continue anyway?", default=False):
            sys.exit(0)

    if update_mode:
        handle_update(DEFAULT_BRANCH, confirm=False)
        return

    print("\nSammie-Roto-2 Manager")

    update_label = "Check for Updates (main branch)" if dev_mode else "Check for Updates"
    actions = [(update_label, lambda: handle_update(DEFAULT_BRANCH))]
    if dev_mode:
        actions.append(("Pull Latest Dev Commits", lambda: handle_update("dev")))
    actions.append(("Reinstall/Repair (fixes problems, or lets you change your graphics backend)", lambda: repair(dev_mode)))
    actions.append(("Exit", lambda: sys.exit(0)))

    for i, (label, _) in enumerate(actions, 1):
        print(f"{i}) {label}")

    choice = ask("> ", valid=[str(i) for i in range(1, len(actions) + 1)])
    actions[int(choice) - 1][1]()

if __name__ == "__main__":
    main()