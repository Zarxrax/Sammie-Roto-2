import os, sys, subprocess, platform, tomllib, shutil
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

def cleanup_cache():
    """After a successful install/reinstall/update, prunes the project cache."""
    run_command([get_uv_exe(), "cache", "prune"])

# ===== UTILS =====
def run_command(cmd):
    """Wrapper to handle uv commands."""
    print(">", " ".join(cmd))
    try:
        subprocess.check_call(cmd, env=get_uv_env())
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
        return None

    if not os.path.exists(".venv"):
        return None

    try:
        result = subprocess.check_output(
            [get_uv_exe(), "pip", "show", "torch", "--python", ".venv"], 
            text=True, stderr=subprocess.DEVNULL, env=get_uv_env()
        )
        version_line = next((l for l in result.splitlines() if l.startswith("Version:")), "").lower()
        for backend in ["cu130", "cu126", "rocm", "xpu", "cpu"]:
            if backend in version_line:
                return backend
    except (subprocess.CalledProcessError, StopIteration):
        print("[Warning: Could not detect installed backend]")
        pass
    
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

# ===== BACKEND SELECTION =====
def choose_backend():
    """Manually prompt the user for their hardware backend."""
    if platform.system() == "Darwin":
        return None

    print("\nSelect PyTorch backend:")
    print("1) NVIDIA CUDA 13.0 (RTX, newer GPUs)")
    print("2) NVIDIA CUDA 12.6 (GTX, older GPUs)")
    print("3) Intel Arc/Xe (XPU)")
    if platform.system() == "Linux":
        print("4) AMD ROCm")
    print("5) CPU (Slow)")

    choice = input("> ").strip()
    mapping = {"1": "cu130", "2": "cu126", "3": "xpu", "4": "rocm", "5": "cpu"}
    if platform.system() == "Windows" and mapping.get(choice) == "rocm":
        return "cpu"
    return mapping.get(choice, "cpu")

def sync_env(backend, reinstall=False):
    """Uses uv sync to update or reinstall the environment."""
    cmd = [get_uv_exe(), "sync", "--frozen"]
    if backend:
        cmd.extend(["--extra", backend])
    
    if reinstall:
        print(f"\n[Reinstalling dependencies for {backend or 'Default/MPS'}...]")
        # --reinstall refreshes all;
        cmd.extend(["--reinstall"])
    else:
        print(f"\n[Syncing dependencies for {backend or 'Default/MPS'}...]")

    run_command(cmd)

def create_shortcuts():
    """Creates/refreshes the appropriate desktop entry point for the current OS."""
    system = platform.system()
    if system == "Windows":
        create_windows_shortcut()
    elif system == "Darwin":
        create_mac_app()
    elif system == "Linux":
        create_linux_desktop_entry()

def resolve_backend(context="continue"):
    """Returns the currently installed backend, or prompts the user to pick
    one if it can't be detected (e.g. after a failed/partial install)."""
    backend = get_installed_backend()
    if not backend:
        if platform.system() != "Darwin":
            print(f"[Could not determine your previously installed backend — please reselect it to {context}.]")
        backend = choose_backend()
    return backend

def perform_update(branch):
    """Pulls the given branch, resyncs the environment, and refreshes
    shortcuts. Shared by both update paths in handle_update()."""
    pull_latest_code(branch)
    sync_env(resolve_backend())
    create_shortcuts()
    print("\nUpdate complete!")
    cleanup_cache()

# ===== CORE ACTIONS =====
def handle_update(branch):
    # Read the local version, with recovery if pyproject.toml is missing
    # or unreadable — a likely sign of a failed or partial install.
    try:
        local_v = get_local_version()
    except Exception as e:
        print(f"[Could not read local version: {e}]")
        print("[pyproject.toml may be missing or corrupt — this can happen after a failed install.]")
        recover = input("Pull latest code from GitHub to recover? (Y/n): ").strip().lower()
        if recover != "n":
            pull_latest_code(branch)
            sync_env(resolve_backend("continue recovery"))
            print("[Recovery complete!]")
            cleanup_cache()
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
        if input("Install update now? (Y/n): ").strip().lower() == "n":
            print("Update skipped.")
            return
        perform_update(branch)
    else:
        print(f"[Already up to date (Version {local_v}).]")

def setup(branch, reinstall=False):
    # Git tracking is initialized as soon as setup() runs (fresh install or
    # reinstall), so future "Check for Updates" runs can fetch/reset
    # cleanly. This only adds the 'origin' remote -- it never touches files.
    init_git_tracking()

    # -- Gather all choices upfront ----------------------------------------

    # 1. Backend selection (with re-prompt on invalid input)
    backend = choose_backend()

    # 2. Pull latest code?
    if reinstall:
        prompt = (
            "\nAlso pull the latest code from GitHub? This will overwrite "
            "any local changes to program files. (y/N): "
        )
        pull_code = input(prompt).strip().lower() == "y"
    else:
        prompt = (
            "\nPull the latest code from GitHub now? Recommended if you're "
            "not sure the downloaded files are the newest release. (Y/n): "
        )
        pull_code = input(prompt).strip().lower() != "n"

    # 3. Model download -- fresh install only
    download_models_now = False
    if not reinstall:
        print("\nModel download:")
        print("1) Download models as needed (default -- models download the first time they are used)")
        print("2) Download all models now (~10GB)")
        model_choice = input("> ").strip()
        download_models_now = model_choice == "2"

    # -- Summarise and confirm ----------------------------------------------
    backend_labels = {
        "cu130": "NVIDIA CUDA 13.0",
        "cu126": "NVIDIA CUDA 12.6",
        "xpu":   "Intel Arc/Xe",
        "rocm":  "AMD ROCm",
        "cpu":   "CPU",
        None:    "CPU/Apple Silicon/MPS",
    }

    print("\n--- Setup summary ---")
    print(f"  Branch           : {branch}")
    print(f"  Pull latest code : {'Yes' if pull_code else 'No'}")
    if platform.system() != "Darwin":
        print(f"  PyTorch backend  : {backend_labels.get(backend, backend)}")
    if not reinstall:
        print(f"  Download models  : {'Download all now (~10GB)' if download_models_now else 'Download as needed'}")
    print("---------------------")

    confirm = input("\nProceed with setup? (Y/n): ").strip().lower()
    if confirm == "n":
        print("Setup cancelled.")
        sys.exit(0)

    # -- Execute ------------------------------------------------------------
    if pull_code:
        pull_latest_code(branch)

    run_command([get_uv_exe(), "python", "install", "--no-bin", PYTHON_VERSION])

    sync_env(backend, reinstall=reinstall)

    create_shortcuts()

    # Make run_sammie.sh executable on Unix-like systems
    if platform.system() != "Windows":
        run_sh = "run_sammie.sh"
        if os.path.exists(run_sh):
            os.chmod(run_sh, os.stat(run_sh).st_mode | 0o755)

    print("\nSetup Complete!")
    cleanup_cache()

    # Run the model downloader last so all dependencies are in place.
    if download_models_now:
        print("\nDownloading all models...")
        run_command([get_uv_exe(), "run", os.path.join("sammie", "model_downloader.py")])


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
            'cd "$(dirname "$0")/../../../"\n'
            './run_sammie.sh\n'
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
    """Creates a .desktop file for GNOME and KDE integration."""
    home = os.path.expanduser("~")
    apps_dir = os.path.join(home, ".local", "share", "applications")
    os.makedirs(apps_dir, exist_ok=True)
    
    desktop_path = os.path.join(apps_dir, "sammie-roto-2.desktop")
    app_dir = os.path.abspath(os.path.dirname(__file__))
    icon_path = os.path.join(app_dir, "sammie", "resources", "icon.png")
    run_sh_path = os.path.join(app_dir, "run_sammie.sh")

    content = [
        "[Desktop Entry]",
        "Type=Application",
        "Name=Sammie-Roto-2",
        "Comment=Video Rotoscoping and Masking Tool",
        f"Exec=\"{run_sh_path}\"",
        f"Icon={icon_path}",
        "Terminal=false",
        "Categories=Graphics;Video;VideoEditing;",
        "StartupWMClass=Sammie-Roto-2",
    ]

    with open(desktop_path, "w") as f:
        f.write("\n".join(content))
    
    os.chmod(desktop_path, 0o755)
    print(f"Created Linux desktop shortcut at: {desktop_path}")

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
def is_app_running():
    """Checks whether Sammie-Roto is currently running by reading the PID
    from Qt's lock file and verifying the process is actually alive."""
    lock_path = os.path.join(
        os.environ.get("TEMP", os.environ.get("TMP", "")) if platform.system() == "Windows" else "/tmp",
        "sammie-roto.lock"
    )
    if not os.path.exists(lock_path):
        return False
    try:
        with open(lock_path, "r") as f:
            pid = int(f.readline().strip())
    except (ValueError, OSError):
        return False  # Unreadable or malformed — treat as stale

    if platform.system() == "Windows":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True
        )
        return str(pid) in result.stdout
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False  # Process doesn't exist — stale lock
        except PermissionError:
            return True   # Process exists but we can't signal it — assume running
def main():
    dev_mode = "--dev" in sys.argv[1:]

    # If app is not installed, bypass main menu and proceed to setup
    if not os.path.exists(".venv"):
        if os.path.exists("python-3.12.8-embed-amd64"):
            print("ERROR: It appears you are trying to install over an older Sammie-Roto installation.")
            print("Please delete the existing folder then extract the files to a new folder and try again.")
            sys.exit(1)

        branch = DEFAULT_BRANCH
        if dev_mode:
            print("\nInstall from:")
            print("1) Main (stable)")
            print("2) Dev (latest, may be unstable)")
            branch = "dev" if input("> ").strip() == "2" else DEFAULT_BRANCH

        setup(branch)
        return

    if is_app_running():
        print("\n[Warning: Sammie-Roto-2 appears to be running.]")
        print("[Please close it before continuing to avoid corrupting your installation.]")
        confirm = input("Continue anyway? (y/N): ").strip().lower()
        if confirm != "y":
            sys.exit(0)

    print("\nSammie-Roto-2 Manager")

    actions = [("Check for Updates", lambda: handle_update(DEFAULT_BRANCH))]
    if dev_mode:
        actions.append(("Pull Latest Dev Commits", lambda: handle_update("dev")))
    actions.append(("Reinstall/Repair", lambda: setup(DEFAULT_BRANCH, reinstall=True)))
    actions.append(("Exit", lambda: sys.exit(0)))

    for i, (label, _) in enumerate(actions, 1):
        print(f"{i}) {label}")

    choice = input("> ").strip()
    try:
        index = int(choice) - 1
        if 0 <= index < len(actions):
            actions[index][1]()
        else:
            sys.exit(0)
    except ValueError:
        sys.exit(0)

if __name__ == "__main__":
    main()