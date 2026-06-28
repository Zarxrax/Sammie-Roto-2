import os, sys, subprocess, platform, tomllib, shutil
import urllib.request

# ===== CONFIG =====
PYTHON_VERSION = "3.12"
REPO_URL = "https://github.com/Zarxrax/Sammie-Roto-2.git"
RAW_PYPROJECT_URL = "https://raw.githubusercontent.com/Zarxrax/Sammie-Roto-2/main/pyproject.toml"

# ===== UTILS =====
def run_command(cmd):
    """Wrapper to handle uv commands.
    Clears VIRTUAL_ENV from the environment before each call to prevent uv's
    internal build isolation environment from leaking into child processes and
    causing 'does not match project environment path' warnings."""
    print(">", " ".join(cmd))
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\nError executing command: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: 'uv' command not found. Please install it from https://astral.sh/uv")
        sys.exit(1)

def get_local_version():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
        return data["project"]["version"]

def get_remote_version():
    try:
        with urllib.request.urlopen(RAW_PYPROJECT_URL) as response:
            data = tomllib.loads(response.read().decode())
            return data["project"]["version"]
    except Exception as e:
        print(f"[Warning: Could not check remote version: {e}]")
        return None

def parse_version(v):
    """Splits a dotted version string into (release_tuple, is_prerelease).
    The release tuple is the leading numeric part of each dot-separated segment,
    e.g. '2.0.0b1' -> ((2, 0, 0), True), '1.10.2' -> ((1, 10, 2), False).
    Any non-digit characters after the leading digits of a segment (a, alpha,
    b, beta, rc, c, dev, etc.) mark the version as a pre-release."""
    release = []
    prerelease = False
    for segment in v.strip().lower().split("."):
        digits = ""
        i = 0
        while i < len(segment) and segment[i].isdigit():
            digits += segment[i]
            i += 1
        release.append(int(digits) if digits else 0)
        if i < len(segment):
            prerelease = True
    return tuple(release), prerelease

def is_newer_version(remote_v, local_v):
    """Returns True if remote_v is a strictly newer *final* release than local_v.

    Pre-release versions are never offered as updates: if remote_v is itself a
    pre-release (e.g. an alpha/beta accidentally left on main), this returns
    False regardless of local_v. But a user currently running a pre-release
    will still be offered the matching final release once one is published,
    since a final release always outranks a pre-release of the same release
    number (e.g. '2.0.0' is newer than '2.0.0b2')."""
    r_release, r_pre = parse_version(remote_v)
    l_release, l_pre = parse_version(local_v)

    if r_pre:
        return False

    length = max(len(r_release), len(l_release))
    r_release += (0,) * (length - len(r_release))
    l_release += (0,) * (length - len(l_release))

    if r_release != l_release:
        return r_release > l_release

    # Same release number and remote is confirmed final (checked above) —
    # it's newer only if the local install is itself a pre-release.
    return l_pre

def get_installed_backend():
    """Detects which torch extra is currently installed (used for updates)."""
    if platform.system() == "Darwin":
        return None

    if not os.path.exists(".venv"):
        return None

    try:
        result = subprocess.check_output(
            ["uv", "pip", "show", "torch", "--python", ".venv"], 
            text=True, stderr=subprocess.DEVNULL
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

def pull_latest_code(hard_reset=False):
    """Ensures the local files match the repository."""
    from dulwich import porcelain
    from dulwich.repo import Repo

    init_git_tracking()
    repo = Repo(".")

    print("[Fetching latest code from GitHub...]")
    porcelain.fetch(repo, "origin")
    
    if hard_reset:
        print("[Restoring all program files to original state...]")
        porcelain.reset(repo, "hard", "origin/main")
    else:
        porcelain.reset(repo, "soft", "origin/main")

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
    cmd = ["uv", "sync"]
    if backend:
        cmd.extend(["--extra", backend])
    
    if reinstall:
        print(f"\n[Reinstalling dependencies for {backend or 'Default/MPS'}...]")
        # --reinstall refreshes all;
        cmd.extend(["--reinstall"])
    else:
        print(f"\n[Syncing dependencies for {backend or 'Default/MPS'}...]")

    run_command(cmd)

# ===== CORE ACTIONS =====
def handle_update():
    # Read the local version, with recovery if pyproject.toml is missing
    # or unreadable — a likely sign of a failed or partial install.
    try:
        local_v = get_local_version()
    except Exception as e:
        print(f"[Could not read local version: {e}]")
        print("[pyproject.toml may be missing or corrupt — this can happen after a failed install.]")
        recover = input("Pull latest code from GitHub to recover? (Y/n): ").strip().lower()
        if recover != "n":
            pull_latest_code(hard_reset=True)
            backend = get_installed_backend()
            if not backend:
                if platform.system() != "Darwin":
                    print("[Could not determine your previously installed backend — please reselect it to continue recovery.]")
                backend = choose_backend()
            sync_env(backend)
            print("[Recovery complete!]")
        else:
            print("[No changes made. Consider using Reinstall/Repair from the main menu.]")
        return

    remote_v = get_remote_version()

    if remote_v is None:
        print("[Could not check for updates. Check your internet connection and try again.]")
        return

    if is_newer_version(remote_v, local_v):
        print(f"\nUpdate available: {remote_v} (current: {local_v})")
        confirm = input("Install update now? (Y/n): ").strip().lower()
        if confirm == "n":
            print("Update skipped.")
            return
        pull_latest_code(hard_reset=True)
        backend = get_installed_backend()
        if not backend:
            if platform.system() != "Darwin":
                print("[Could not determine your previously installed backend — please reselect it to continue the update.]")
            backend = choose_backend()
        sync_env(backend)

        # Recreate shortcuts
        if platform.system() == "Windows":
            create_windows_shortcut()
        if platform.system() == "Darwin":
            create_mac_app()
        if platform.system() == "Linux":
            create_linux_desktop_entry()

        print("\nUpdate complete!")
    else:
        print(f"[Already up to date (Version {local_v}).]")

def setup(reinstall=False):
    # Git tracking is initialized as soon as setup() runs (fresh install or
    # reinstall), so future "Check for Updates" runs can fetch/reset
    # cleanly. This only adds the 'origin' remote -- it never touches files.
    init_git_tracking()

    # -- Gather all choices upfront ----------------------------------------
    # Ask every question before doing any work, so we can summarise and
    # confirm before anything irreversible happens. The user can walk away
    # after confirming and let the whole process complete unattended.

    # 1. Pull latest code?
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

    # 2. Backend selection (with re-prompt on invalid input)
    backend = choose_backend()

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
        pull_latest_code(hard_reset=True)

    run_command(["uv", "python", "install", "--no-bin", PYTHON_VERSION])

    sync_env(backend, reinstall=reinstall)

    # Create desktop shortcut on Windows
    if platform.system() == "Windows":
        create_windows_shortcut()

    # Create .app bundle on macOS
    if platform.system() == "Darwin":
        create_mac_app()

    # Create .desktop file on Linux
    if platform.system() == "Linux":
        create_linux_desktop_entry()

    # Make run_sammie.sh executable on Unix-like systems
    if platform.system() != "Windows":
        run_sh = "run_sammie.sh"
        if os.path.exists(run_sh):
            os.chmod(run_sh, os.stat(run_sh).st_mode | 0o755)

    print("\nSetup Complete!")

    # Run the model downloader last so all dependencies are in place.
    if download_models_now:
        print("\nDownloading all models...")
        run_command(["uv", "run", os.path.join("sammie", "model_downloader.py")])


# ===== CREATE SHORTCUTS =====
def create_mac_app():
    """Creates a double-clickable .app bundle on macOS."""

    app_name = "Sammie-Roto-2.app"
    app_dir = os.path.abspath(os.path.dirname(__file__))
    macos_dir = os.path.join(app_name, "Contents", "MacOS")
    resources_dir = os.path.join(app_name, "Contents", "Resources")
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
    plist_path = os.path.join(app_name, "Contents", "Info.plist")
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
            ["xattr", "-dr", "com.apple.quarantine", app_name],
            check=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        pass  # Not quarantined, nothing to clear

    print(f"Created {app_name} — double-click it to launch!")


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
    # Check if app is running before doing anything

    if not os.path.exists(".venv"):
        if os.path.exists("python-3.12.8-embed-amd64"):
            print("ERROR: It appears you are trying to install over an older Sammie-Roto installation.")
            print("Please delete the existing folder then extract the files to a new folder and try again.")
            sys.exit(1)
        setup()
    else:
        if is_app_running():
            print("\n[Warning: Sammie-Roto-2 appears to be running.]")
            print("[Please close it before continuing to avoid corrupting your installation.]")
            confirm = input("Continue anyway? (y/N): ").strip().lower()
            if confirm != "y":
                sys.exit(0)

        print("\nSammie-Roto-2 Manager")
        print("1) Check for Updates")
        print("2) Reinstall/Repair")
        print("3) Exit")
        
        choice = input("> ").strip()
        if choice == "1":
            handle_update()
        elif choice == "2":
            setup(reinstall=True)
        else:
            sys.exit(0)

if __name__ == "__main__":
    main()