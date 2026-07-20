"""
Patches the torchvision.transforms.functional_tensor -> functional_tensor
import that basicsr (and, in some versions, facexlib) still use, which
newer torchvision releases removed.

Run at image-build time only (see modal_app/app.py).
"""
import importlib.util
import pathlib
import sys

OLD = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
NEW = "from torchvision.transforms.functional import rgb_to_grayscale"


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
    return patched


n = patch("basicsr", "data/degradations.py", required=True)
if n == 0:
    sys.exit(
        "ERROR: expected torchvision.transforms.functional_tensor import not "
        "found in basicsr/data/degradations.py -- basicsr's source changed, "
        "update modal_app/patch_torchvision_compat.py"
    )

patch("facexlib", required=False)  # best-effort: only some versions hit this