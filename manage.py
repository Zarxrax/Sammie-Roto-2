import os, sys, subprocess, platform, tomllib, shutil
import urllib.request

# ===== CONFIG =====
PYTHON_VERSION = "3.12"
REPO_URL = "https://github.com/Zarxrax/Sammie-Roto-2.git"
RAW_PYPROJECT_URL = "https://raw.githubusercontent.com/Zarxrax/Sammie-Roto-2/main/pyproject.toml"

# ===== UTILS =====
def run_command(cmd):
    """Wrapper to handle uv commands."""
    print(">", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
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
def pull_latest_code(hard_reset=False):
    """Ensures the local files match the repository."""
    from dulwich import porcelain
    from dulwich.repo import Repo

    if not os.path.exists(".git"):
        print("[Initializing Git tracking...]")
        repo = Repo.init(".")
        porcelain.remote_add(repo, "origin", REPO_URL)
    else:
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
    local_v = get_local_version()
    remote_v = get_remote_version()

    if remote_v and remote_v > local_v:
        print(f"[Update found: {remote_v} (Local: {local_v})]")
        pull_latest_code(hard_reset=True)
        backend = get_installed_backend()
        if not backend:
            backend = choose_backend()
        sync_env(backend)
        #if platform.system() == "Windows":
        #    create_windows_shortcut()
        #if platform.system() == "Darwin":
        #    create_mac_app()
        #if platform.system() == "Linux":
        #    create_linux_desktop_entry()
        print("\nUpdate complete!")
    else:
        print(f"[Already up to date (Version {local_v}).]")

def setup(reinstall=False):
    # Fetch/Restore files if it's a reinstall
    if reinstall:
        pull_latest_code(hard_reset=True)

    # Python Check
    run_command(["uv", "python", "install", "--no-bin", PYTHON_VERSION])
    
    # Backend Selection
    # If fresh install or reinstall, always ask. 
    backend = choose_backend()
    
    # Environment Sync
    sync_env(backend, reinstall=reinstall)
    
    # Git tracking (initial setup only)
    if not os.path.exists(".git"):
        pull_latest_code(hard_reset=False)

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
def main():
    # Check if app is running before doing anything

    if not os.path.exists(".venv"):
        setup()
    else:
        print("\nPlease ensure that Sammie-Roto-2 is not running before continuing.")
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