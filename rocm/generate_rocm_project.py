# /// script
# requires-python = ">=3.12"
# ///
r"""
Generates the separate ROCm uv project:  rocm/pyproject.toml  and  rocm/uv.lock

Why a separate project: AMD's package index also hosts copies of numpy, pillow and other
ordinary packages. In the main project that would change what every NVIDIA / CPU / macOS user
installs. In its own folder, only people who choose AMD ROCm are affected.

It can live inside rocm/ (keeping the main folder tidy): it finds the repository root by itself,
as the nearest folder above the script that has both manage.py and pyproject.toml (use --root to
override). Run it with the uv in the repository's .uv folder, the one the installer uses; the script
uses that same uv, and the installer's Python and cache folders, for its own uv commands. Always
include --no-project: without it, uv may sync a project environment first (the app's .venv, which
would replace the installed backend). From the repository root:
    .uv\uv.exe run --no-project rocm\generate_rocm_project.py                regenerate rocm/pyproject.toml and rocm/uv.lock
    .uv\uv.exe run --no-project rocm\generate_rocm_project.py --check        offline: is rocm/ still in sync with the rest?
    .uv\uv.exe run --no-project rocm\generate_rocm_project.py --report       also print install sizes and the torch versions on the index
    .uv\uv.exe run --no-project rocm\generate_rocm_project.py --torch 2.14.0 --torchvision 0.29.0
(On Linux and macOS use .uv/uv and forward slashes.)

Inputs (the only things you edit):
    pyproject.toml    its [project] dependencies are copied into rocm/pyproject.toml
    manage.py         ROCM_GPUS lists the GPUs offered; each becomes an extra "rocm-<gfx target>"
    the constants below: AMD index channel and torch / torchvision versions

What it does:
    1. Locks a throwaway project against AMD's index, only to learn which packages each GPU needs.
    2. Writes rocm/pyproject.toml: your base dependencies, one extra per GPU, every AMD-only
       package pinned as a direct dependency (so AMD's index is used for those and nothing else),
       and an override that keeps Linux-only packages (triton) off Windows.
    3. Locks it (rocm/uv.lock) and checks the result.
GPUs that are not installable on BOTH Windows and Linux at these versions are left out and
listed in the file header. Commit rocm/pyproject.toml and rocm/uv.lock together.

Output is ASCII-only so it prints cleanly on any Windows console.
"""
import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote

# ===== CONFIG =====
ROCM_DIR = "rocm"
DEFAULT_TORCH = "2.13.0"
DEFAULT_TORCHVISION = "0.28.0"
CHANNELS = {
    "stable":  "https://stable.repo.amd.com/rocm/whl-next/",
    "nightly": "https://nightly.repo.amd.com/rocm/whl-next/",
}

PROJECT = "rocm-multiarch-test"   # name of the throwaway discovery project

# Packages that only make sense from the AMD index. Generic ones (numpy, pillow,
# jinja2...) are mirrored there too but must keep coming from PyPI.
AMD_ONLY = re.compile(r"^(torch|torchvision|triton|rocm(-.*)?|amd-.*)$")
# Packages that MUST have a wheel on every supported OS. (triton is Linux-only and the
# 'rocm' meta package is a source distribution, so neither is required.)
REQUIRED = re.compile(r"^(torch|torchvision|rocm-sdk-.*|amd-.*)$")
REQUIRED_ENVS = [
    "sys_platform == 'win32' and platform_machine == 'AMD64'",
    "sys_platform == 'linux' and platform_machine == 'x86_64'",
]


# ---------------------------------------------------------------- inputs
def find_root(start_dirs):
    """Nearest folder (searching each start folder and its parents) holding BOTH manage.py and
    pyproject.toml. manage.py is what tells the main project from rocm/, which has its own pyproject.toml."""
    for start in start_dirs:
        for d in [start, *start.parents]:
            if (d / "manage.py").is_file() and (d / "pyproject.toml").is_file():
                return d
    return None


def find_uv(root):
    """The uv in the repository's .uv folder (the one the installer uses), else the one on PATH.
    With the local uv, also uses the installer's Python and cache folders when they exist."""
    uv_dir = root / ".uv"
    local = uv_dir / ("uv.exe" if sys.platform == "win32" else "uv")
    os.environ.pop("VIRTUAL_ENV", None)     # as manage.py does: keep uv's own run environment out of uv commands
    if local.is_file():
        for variable, folder in (("UV_PYTHON_INSTALL_DIR", "python"), ("UV_CACHE_DIR", "uv_cache")):
            if (uv_dir / folder).is_dir():
                os.environ.setdefault(variable, str(uv_dir / folder))
        return str(local)
    return shutil.which("uv") or sys.exit("uv was not found: expected it in the repository's .uv folder or on PATH.")


def script_command(root):
    """The command that runs this script, relative to the repository root (for messages)."""
    try:
        rel = Path(__file__).resolve().relative_to(root).as_posix()
    except ValueError:
        rel = Path(__file__).name
    return f"uv run --no-project {rel}"


def read_targets(manage_py):
    """gfx targets listed in ROCM_GPUS in manage.py (read without importing the file)."""
    tree = ast.parse(Path(manage_py).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ROCM_GPUS" for t in node.targets):
            targets = [t for t, _ in ast.literal_eval(node.value)]
            if len(targets) != len(set(targets)):
                sys.exit("ROCM_GPUS in manage.py lists a GPU target twice")
            return targets
    sys.exit(f"ROCM_GPUS was not found in {manage_py}")


def read_root_project(root):
    """Name, python requirement and base dependencies from the main pyproject.toml."""
    data = tomllib.loads((Path(root) / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return {"name": data["name"], "requires_python": data["requires-python"],
            "dependencies": list(data.get("dependencies", []))}


# ---------------------------------------------------------------- pyproject generation
def device_extra(targets, torch_ver=None, tv_ver=None):
    devs = ",".join(f"device-{t}" for t in targets)
    return [f"torch[{devs}]" + (f"=={torch_ver}" if torch_ver else ""),
            f"torchvision[{devs}]" + (f"=={tv_ver}" if tv_ver else "")]


def reference_extras(targets, torch_ver=None, tv_ver=None):
    return {f"rocm-{t}": device_extra([t], torch_ver, tv_ver) for t in targets}


def render_pyproject(project, url, explicit, extras, source_names, required_envs=None, overrides=(), header=""):
    lines = []
    if header:
        lines += [f"# {h}".rstrip() for h in header.splitlines()] + [""]
    lines += ["[project]", f"name = {json.dumps(project['name'])}", 'version = "0.0.0"',
              f"requires-python = {json.dumps(project['requires_python'])}"]
    if project["dependencies"]:
        lines.append("dependencies = [")
        lines += [f"    {json.dumps(d)}," for d in project["dependencies"]]
        lines.append("]")
    else:
        lines.append("dependencies = []")
    lines += ["", "[project.optional-dependencies]"]
    for name, deps in extras.items():
        lines.append(f"{name} = [")
        lines += [f'    "{d}",' for d in deps]
        lines.append("]")
    lines += ["", "[tool.uv]", "package = false"]
    if overrides:       # restrict single-OS packages (triton) to that OS, whatever their dependents say
        lines.append("override-dependencies = [")
        lines += [f'    "{o}",' for o in sorted(overrides)]
        lines.append("]")
    if required_envs:   # make `uv lock` fail if any package lacks a wheel for these platforms
        lines.append("required-environments = [")
        lines += [f'    "{e}",' for e in required_envs]
        lines.append("]")
    lines += ["", "[tool.uv.sources]"]
    lines += [f'{n} = {{ index = "amd-rocm" }}' for n in sorted(source_names)]
    lines += ["", "[[tool.uv.index]]", 'name = "amd-rocm"', f'url = "{url}"',
              f"explicit = {str(explicit).lower()}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- wheel filename logic
def python_compatible(py_tags, abi_tag):
    """True if a wheel's python/abi tags work on CPython 3.12."""
    for t in py_tags.split("."):
        if t.startswith("py3"):
            return True
        m = re.fullmatch(r"cp3(\d+)", t)
        if m:
            minor = int(m.group(1))
            if minor == 12 and abi_tag in ("cp312", "none"):
                return True
            if abi_tag == "abi3" and minor <= 12:
                return True
    return False


def wheel_matches(url_or_name, plat):
    """True if this wheel installs on 64-bit Windows ("win") or x86-64 Linux ("linux") with Python 3.12."""
    name = url_or_name.rsplit("/", 1)[-1]
    if not name.lower().endswith(".whl"):
        return False
    parts = name[:-4].split("-")
    if len(parts) < 5:
        return False
    py_tags, abi_tag, plat_tags = parts[-3], parts[-2], parts[-1].split(".")
    if not python_compatible(py_tags, abi_tag):
        return False
    if "any" in plat_tags:
        return True
    if plat == "win":
        return "win_amd64" in plat_tags
    return any("linux" in t and "x86_64" in t for t in plat_tags)


# ---------------------------------------------------------------- lock file reading
def load_lock(path):
    packages = tomllib.loads(path.read_text(encoding="utf-8")).get("package", [])
    by_name = {}
    for p in packages:
        by_name.setdefault(p["name"], []).append(p)
    return by_name


def is_amd(pkg):
    return "repo.amd.com" in pkg.get("source", {}).get("registry", "")


def dep_key(d):
    return d["name"], tuple(sorted(d.get("extra", [])))


def closure(by_name, roots):
    """All package names reachable from roots [(name, extras)], following extras."""
    seen, stack, names = set(), list(roots), set()
    while stack:
        key = stack.pop()
        if key in seen:
            continue
        seen.add(key)
        name, extras = key
        names.add(name)
        for pkg in by_name.get(name, []):
            stack += [dep_key(d) for d in pkg.get("dependencies", [])]
            for e in extras:
                stack += [dep_key(d) for d in pkg.get("optional-dependencies", {}).get(e, [])]
    return names


def project_extras(by_name):
    proj = by_name[PROJECT][0]
    return {e: [dep_key(d) for d in deps] for e, deps in proj.get("optional-dependencies", {}).items()}


def amd_wheels(by_name, name, plat):
    return [w["url"] for p in by_name.get(name, []) if is_amd(p)
            for w in p.get("wheels", []) if wheel_matches(w["url"], plat)]


def missing_wheels(by_name, names, plat):
    """Required packages in `names` that have no wheel for this OS."""
    return sorted(n for n in names if REQUIRED.match(n) and not amd_wheels(by_name, n, plat))


# ---------------------------------------------------------------- sizes from the server
def remote_size(url):
    for method, headers in (("HEAD", {}), ("GET", {"Range": "bytes=0-0"})):
        try:
            req = urllib.request.Request(url, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 206:
                    total = r.headers.get("Content-Range", "").rsplit("/", 1)[-1]
                    if total.isdigit():
                        return int(total)
                else:
                    n = r.headers.get("Content-Length", "")
                    if n.isdigit():
                        return int(n)
        except Exception:
            continue
    return None


def fetch_sizes(urls):
    urls = sorted(set(urls))
    print(f"  Asking the server for the size of {len(urls)} files (this can take a minute)...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        return dict(zip(urls, pool.map(remote_size, urls)))


# ---------------------------------------------------------------- reports
def mb(n):
    return "   n/a" if n is None else f"{n / 1e6:6.0f}"


def size_report(by_name):
    extras = project_extras(by_name)
    roots_by_extra = {"(base, no GPU code)": [("torch", ()), ("torchvision", ())], **extras}
    names_by_extra = {e: {n for n in closure(by_name, r) if AMD_ONLY.match(n)}
                      for e, r in roots_by_extra.items()}
    all_urls = [u for names in names_by_extra.values() for n in names
                for plat in ("win", "linux") for u in amd_wheels(by_name, n, plat)]
    sizes = fetch_sizes(all_urls)

    def total(names, plat):
        if missing_wheels(by_name, names, plat):
            return None
        s = 0
        for n in names:
            got = [sizes.get(u) for u in amd_wheels(by_name, n, plat) if sizes.get(u)]
            if got:
                s += max(got)
        return s

    base = {p: total(names_by_extra["(base, no GPU code)"], p) for p in ("win", "linux")}
    print("\n  Install size per extra, MB (AMD/ROCm packages only; numpy, pillow etc. not counted)")
    print(f"  {'extra':22} {'win':>6} {'linux':>6} | {'win +':>6} {'linux +':>7}  device packages")
    for e, names in names_by_extra.items():
        w, l = total(names, "win"), total(names, "linux")
        dw = None if w is None or base["win"] is None else w - base["win"]
        dl = None if l is None or base["linux"] is None else l - base["linux"]
        dev = sorted(n for n in names if "device-" in n)
        short = ", ".join(n.replace("amd-torchvision-device-", "tv-").replace("amd-torch-device-", "t-")
                           .replace("rocm-sdk-device-", "sdk-") for n in dev)
        if len(short) > 70:
            short = short[:67] + "..."
        if dev:
            note = short
        elif e.startswith("("):
            note = ""
        else:
            note = "<-- NO DEVICE PACKAGES: target not offered for these versions"
        for plat, label in (("win", "Windows"), ("linux", "Linux")):
            miss = missing_wheels(by_name, names, plat)
            if miss:
                note += f"  [{label}: no wheel for {', '.join(miss[:2])}{' ...' if len(miss) > 2 else ''}]"
        print(f"  {e:22} {mb(w)} {mb(l)} | {mb(dw)} {mb(dl)}   {note}")
    print("  ('+' columns = what the GPU code adds on top of the base.")
    print("   n/a = a required package has no wheel for that OS, so it cannot be installed there.)")


def version_sort_key(v):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", v)]


def list_versions(index_url, pkg):
    try:
        req = urllib.request.Request(f"{index_url}{pkg}/", headers={"Accept": "text/html"})
        html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
    except Exception as e:
        print(f"    could not list {pkg}: {e}")
        return
    found = {}
    for href in re.findall(r'href="([^"#]+)', html):
        fname = unquote(href.rsplit("/", 1)[-1])
        if not fname.endswith(".whl"):
            continue
        ver = fname[:-4].split("-")[1]
        for plat in ("win", "linux"):
            if wheel_matches(fname, plat):
                found.setdefault(ver, set()).add(plat)
    print(f"\n  {pkg} versions on the index (Python 3.12 wheels):")
    for ver in sorted(found, key=version_sort_key, reverse=True)[:12]:
        print(f"    {ver:28} {' '.join(sorted(found[ver]))}")
    if not found:
        print("    (none found)")


def generate_direct_dep_pyproject(project, url, by_name, header=""):
    """Explicit index; every AMD-only package listed as a direct, pinned dependency.
    Returns (pyproject_text, skipped) where skipped maps extra -> reason. An extra is
    skipped if it has no device packages or any required package lacks a wheel on
    Windows or Linux (checked here, from the lock data).
    Packages that only have wheels for one OS (triton) also get an entry in
    override-dependencies restricting them to that OS. Without it, torch's own
    unconditional requirement on triton makes `uv sync` try to install triton on
    Windows, where it does not exist. (This is safe because rocm/ is its own project:
    the override would be wrong in a project that also installs CUDA builds of torch.)"""
    extras_out, source_names, skipped, overrides = {}, set(), {}, set()
    for e, roots in project_extras(by_name).items():
        names = sorted(n for n in closure(by_name, roots) if AMD_ONLY.match(n))
        if not any("device-" in n for n in names):
            skipped[e] = "no device packages for these versions"
            continue
        bad = [f"{label}: {', '.join(m[:2])}" for plat, label in (("win", "Windows"), ("linux", "Linux"))
               if (m := missing_wheels(by_name, names, plat))]
        if bad:
            skipped[e] = "no wheel for " + "; ".join(bad)
            continue
        deps = []
        for n in names:
            entries = [p for p in by_name.get(n, []) if is_amd(p)]
            if not entries:
                continue
            versions = {p["version"] for p in entries}
            spec = f"{n}=={next(iter(versions))}" if len(versions) == 1 else n
            win, lin = bool(amd_wheels(by_name, n, "win")), bool(amd_wheels(by_name, n, "linux"))
            if win and not lin:
                spec += " ; sys_platform == 'win32'"
                overrides.add(spec)
            elif lin and not win:
                spec += " ; sys_platform == 'linux'"
                overrides.add(spec)
            deps.append(spec)
            source_names.add(n)
        extras_out[e] = deps
    text = render_pyproject(project, url, True, extras_out, source_names, REQUIRED_ENVS, overrides, header)
    return text, skipped


EXPORT_LINE = re.compile(r"^([A-Za-z0-9_.-]+)==\S+?(?:\s*;\s*(.+))?$")


def unrestricted_single_os_packages(export_text, by_name):
    """Names of packages that have wheels for only one OS but appear in a `uv export`
    without a platform marker, i.e. packages `uv sync` would try to install everywhere."""
    single = {n.lower().replace("_", "-") for n in by_name if is_amd(by_name[n][0]) and AMD_ONLY.match(n)
              and bool(amd_wheels(by_name, n, "win")) != bool(amd_wheels(by_name, n, "linux"))
              and (amd_wheels(by_name, n, "win") or amd_wheels(by_name, n, "linux"))}
    bad = []
    for line in export_text.splitlines():
        m = EXPORT_LINE.match(line.strip())
        if m and m.group(1).lower().replace("_", "-") in single and not m.group(2):
            bad.append(m.group(1))
    return sorted(set(bad))


def check_export(uv, folder, by_name, extra):
    """True if no single-OS package would be installed on every OS (the triton problem)."""
    r = subprocess.run([uv, "export", "--frozen", "--no-hashes", "--no-header", "--extra", extra],
                       cwd=folder, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(f"  CHECK: could not run `uv export` ({(r.stderr.strip().splitlines() or [''])[-1]})")
        return False
    bad = unrestricted_single_os_packages(r.stdout, by_name)
    if bad:
        print(f"  CHECK FAILED: {', '.join(bad)} would be installed on every OS but only has wheels for one.")
        return False
    print(f"  CHECK OK: single-OS packages are restricted to their OS in the lock ({extra}).")
    return True


# ---------------------------------------------------------------- driver
def uv_lock(uv, folder, channel):
    cmd = [uv, "lock"] + (["--prerelease", "allow"] if channel == "nightly" else [])
    r = subprocess.run(cmd, cwd=folder, capture_output=True, text=True, errors="replace")
    if r.returncode == 0:
        print("  RESULT: uv lock succeeded")
        return True
    print("  RESULT: uv lock FAILED. Last lines of the error:")
    for line in (r.stderr.strip().splitlines() or ["(no output)"])[-15:]:
        print("    " + line)
    return False


def verify_sources(by_name):
    """Problems with where packages came from in the final lock (empty list = fine)."""
    problems = []
    for name in ("torch", "torchvision"):
        if not by_name.get(name):
            problems.append(f"{name} is not in the lock")
        elif not all(is_amd(p) for p in by_name[name]):
            problems.append(f"{name} was resolved from somewhere other than AMD's index")
    generic = sorted(n for n, ps in by_name.items() if not AMD_ONLY.match(n) and any(is_amd(p) for p in ps))
    if generic:
        problems.append("ordinary packages came from AMD's index instead of PyPI: " + ", ".join(generic[:8]))
    return problems


def skipped_header_line(skipped):
    if not skipped:
        return "Skipped: none"
    return "Skipped: " + ", ".join(e.removeprefix("rocm-") for e in skipped)


def cmd_generate(args):
    root = Path(args.root).resolve()
    uv = find_uv(root)
    print(f"uv: {uv} ({subprocess.run([uv, '--version'], capture_output=True, text=True).stdout.strip()})")
    targets = read_targets(root / "manage.py")
    project = read_root_project(root)
    url = CHANNELS[args.channel]
    print(f"{len(targets)} GPU targets from manage.py; {len(project['dependencies'])} base dependencies from pyproject.toml")

    work = Path(tempfile.mkdtemp(prefix="rocm-gen-"))
    try:
        scratch = {"name": PROJECT, "requires_python": project["requires_python"], "dependencies": []}
        (work / "pyproject.toml").write_text(
            render_pyproject(scratch, url, False, reference_extras(targets, args.torch, args.torchvision),
                             ["torch", "torchvision"]), encoding="utf-8")
        print("=" * 72)
        print(f"STEP 1: discovering the packages each GPU needs ({args.channel} channel)")
        if not uv_lock(uv, work, args.channel):
            sys.exit("Discovery failed, so nothing was written.")
        by_name = load_lock(work / "uv.lock")
        if args.report:
            size_report(by_name)
            for pkg in ("torch", "torchvision"):
                list_versions(url, pkg)

        header = "\n".join([
            "GENERATED FILE - DO NOT EDIT BY HAND.",
            f"Regenerate with:  {script_command(root)}   (from the repository root, using the uv in .uv)",
            "Base dependencies are copied from ../pyproject.toml; the GPU list is ROCM_GPUS in ../manage.py.",
            f"AMD index: {url}",
            f"torch {args.torch}, torchvision {args.torchvision}",
            "",
            "Skipped targets are not installable on both Windows and Linux at these versions.",
        ])
        text, skipped = generate_direct_dep_pyproject(project, url, by_name, header)
        text = text.replace("# Skipped targets are not installable on both Windows and Linux at these versions.",
                            "# Skipped targets are not installable on both Windows and Linux at these versions.\n"
                            f"# {skipped_header_line(skipped)}", 1)
    finally:
        if args.keep_work:
            print("Scratch folder kept:", work)
        else:
            shutil.rmtree(work, ignore_errors=True)

    rocm = root / ROCM_DIR
    rocm.mkdir(exist_ok=True)
    (rocm / "pyproject.toml").write_text(text, encoding="utf-8")
    included = [t for t in targets if f"rocm-{t}" not in skipped]
    print("=" * 72)
    print(f"STEP 2: wrote {rocm / 'pyproject.toml'}")
    print(f"  GPUs included ({len(included)}): {', '.join(included) or 'none'}")
    for e, why in skipped.items():
        print(f"  LEFT OUT {e.removeprefix('rocm-'):10} {why}")
    if not included:
        sys.exit("No GPU target is installable, so there is nothing to lock.")
    if args.no_lock:
        print("(--no-lock: stopping before uv lock)")
        return 0

    print("=" * 72)
    print("STEP 3: locking the rocm project")
    if not uv_lock(uv, rocm, args.channel):
        return 1
    final = load_lock(rocm / "uv.lock")
    ok = True
    for p in verify_sources(final):
        print("  CHECK FAILED:", p)
        ok = False
    if ok:
        print("  CHECK OK: torch/torchvision come from AMD's index; ordinary packages come from PyPI.")
    ok = check_export(uv, rocm, final, f"rocm-{included[0]}") and ok
    print("=" * 72)
    if ok:
        print("Done. Commit rocm/pyproject.toml and rocm/uv.lock together with your other changes.")
        return 0
    print("Generated, but a check failed (see above). Do not commit these files until it is resolved.")
    return 1


def cmd_check(args):
    """Offline check that rocm/ is in sync with pyproject.toml and manage.py."""
    root = Path(args.root).resolve()
    rocm = root / ROCM_DIR
    problems = []
    if not (rocm / "uv.lock").exists():
        problems.append("rocm/uv.lock is missing")
    if not (rocm / "pyproject.toml").exists():
        problems.append("rocm/pyproject.toml is missing")
    else:
        text = (rocm / "pyproject.toml").read_text(encoding="utf-8")
        gen = tomllib.loads(text)
        proj = read_root_project(root)
        if gen["project"].get("dependencies", []) != proj["dependencies"]:
            problems.append("base dependencies differ from pyproject.toml - regenerate")
        if gen["project"]["requires-python"] != proj["requires_python"]:
            problems.append("requires-python differs from pyproject.toml - regenerate")
        extras = {e.removeprefix("rocm-") for e in gen["project"].get("optional-dependencies", {})}
        targets = set(read_targets(root / "manage.py"))
        m = re.search(r"^# Skipped: (.*)$", text, re.M)
        skipped = {t.strip() for t in m.group(1).split(",")} if m and m.group(1) != "none" else set()
        if extras - targets:
            problems.append("rocm/ has GPUs not listed in manage.py ROCM_GPUS: " + ", ".join(sorted(extras - targets)))
        if targets - extras - skipped:
            problems.append("GPUs in manage.py ROCM_GPUS missing from rocm/ - regenerate: "
                            + ", ".join(sorted(targets - extras - skipped)))
    if problems:
        print("rocm/ is OUT OF DATE:")
        for p in problems:
            print("  -", p)
        print(f"Regenerate with:  {script_command(root)}   (from the repository root, using the uv in .uv)")
        return 1
    print("rocm/ is in sync with pyproject.toml and manage.py.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Generate the separate ROCm uv project (rocm/).")
    ap.add_argument("--root", help="repository root (default: found automatically, see above)")
    ap.add_argument("--channel", default="stable", choices=sorted(CHANNELS))
    ap.add_argument("--torch", default=DEFAULT_TORCH, help=f"torch version (default {DEFAULT_TORCH})")
    ap.add_argument("--torchvision", default=DEFAULT_TORCHVISION, help=f"torchvision version (default {DEFAULT_TORCHVISION})")
    ap.add_argument("--check", action="store_true", help="offline: check rocm/ is in sync, then exit")
    ap.add_argument("--report", action="store_true", help="also print install sizes and available torch versions")
    ap.add_argument("--no-lock", action="store_true", help="write rocm/pyproject.toml but do not lock it")
    ap.add_argument("--keep-work", action="store_true", help="keep the scratch folder used for discovery")
    args = ap.parse_args()
    root = Path(args.root).resolve() if args.root else find_root([Path(__file__).resolve().parent, Path.cwd()])
    if root is None or not (root / "manage.py").is_file() or not (root / "pyproject.toml").is_file():
        sys.exit("Could not find the repository root (a folder containing both manage.py and pyproject.toml).\n"
                 "Run this from inside the repository, or pass --root <folder>.")
    args.root = root
    sys.exit(cmd_check(args) if args.check else cmd_generate(args))


if __name__ == "__main__":
    main()
