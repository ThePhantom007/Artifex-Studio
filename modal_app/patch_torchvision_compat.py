"""
Patches the torchvision.transforms.functional_tensor -> functional_tensor
import that basicsr (and, in some versions, facexlib) still use, which
newer torchvision releases removed.

Also purges any __pycache__ directories under the patched packages. `pip
install` compiles .pyc bytecode at install time; without this step, Python
can keep executing the *pre-patch* bytecode even after the .py source has
been correctly rewritten, since some pip/compileall configurations cache
with the "unchecked hash" invalidation mode that skips the normal
source-vs-cache staleness check.
"""
import importlib.util
import pathlib
import shutil
import sys

OLD = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
NEW = "from torchvision.transforms.functional import rgb_to_grayscale"


def _purge_pycache(root: pathlib.Path) -> None:
    for cache_dir in root.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)


def patch(module_name: str, relative_path: str | None = None, required: bool = True) -> int:
    spec = importlib.util.find_spec(module_name)
    if not spec or not spec.submodule_search_locations:
        if required:
            sys.exit(f"ERROR: could not locate installed package '{module_name}'")
        print(f"skip: '{module_name}' not installed")
        return 0

    root = pathlib.Path(list(spec.submodule_search_locations)[0])
    targets = [root / relative_path] if relative_path else list(root.rglob("*.py"))

    patched = 0
    for target in targets:
        if not target.is_file():
            continue
        text = target.read_text()
        if OLD in text:
            target.write_text(text.replace(OLD, NEW))
            patched += 1
            print(f"patched {target}")

    if patched:
        _purge_pycache(root)
        print(f"purged __pycache__ under {root}")

    return patched


n = patch("basicsr", "data/degradations.py", required=True)
if n == 0:
    sys.exit(
        "ERROR: expected torchvision.transforms.functional_tensor import not "
        "found in basicsr/data/degradations.py -- basicsr's source changed, "
        "update modal_app/patch_torchvision_compat.py"
    )

patch("facexlib", required=False)  # best-effort: only some versions hit this