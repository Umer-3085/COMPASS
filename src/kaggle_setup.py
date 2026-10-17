"""Prepare the Kaggle data mount for training (no manual path surgery).

Kaggle dataset mounts come with quirks this module absorbs:

1. **Arbitrary nesting** — a dataset's files can sit at any depth inside
   the mount (e.g. ``.../compass-preprocessed/compass_preprocessed/
   preprocessed/...``). We discover the real data tree by locating
   ``template/ct_template.nii.gz`` instead of hard-coding a path.
2. **Hotfix folders** — when a fixed folder (e.g. ``case_01fix/mr/train/
   case_01/...``) was dropped into the dataset via the browser upload and
   could not be merged in place, it sits as a sibling tree inside the
   mount. We detect any ``<X>/{mr|ct}/<split>/case_XX/`` tree outside the
   discovered data root and overlay it onto the working copy.

Workflow: discover -> copy the preprocessed tree to a writable dir
(``/kaggle/working/compass_data``) -> overlay hotfixes -> sanity-check
every case's first slice is on the 235x235 template grid -> write the
final root to ``_data_root.txt`` inside the work dir and print it.

Usage (Kaggle notebook):
    python -m src.kaggle_setup
    # -> DATA_ROOT=/kaggle/working/compass_data   (last line, also in file)

Local/test usage: ``prepare(inputs, work)`` is importable and fully
covered by tests/test_kaggle_setup.py.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

TEMPLATE_REL = Path("template") / "ct_template.nii.gz"
EXPECTED_SLICE = (235, 235)          # template grid, in-plane
MARKER = "_data_root.txt"


def discover(inputs: Path, explicit: Path | None = None) -> Path:
    """Find the preprocessed tree under the attached datasets.

    Locates ``template/ct_template.nii.gz`` and returns the directory
    that contains ``template/`` (the data root). With multiple hits the
    shortest path wins (most shallowly nested = the actual mount).
    """
    if explicit is not None:
        if not (explicit / TEMPLATE_REL).exists():
            raise FileNotFoundError(f"no {TEMPLATE_REL} under {explicit}")
        return explicit
    hits = sorted(inputs.rglob(TEMPLATE_REL.name),
                  key=lambda p: (len(p.parts), str(p)))
    hits = [h for h in hits if h.parent.name == "template"]
    if not hits:
        raise FileNotFoundError(
            f"{TEMPLATE_REL} not found under {inputs} — attach the "
            "'compass-preprocessed' dataset as input data")
    root = hits[0].parent.parent
    if len(hits) > 1:
        print(f"[kaggle_setup] multiple candidates, picked: {root}")
    return root


def find_hotfix_roots(inputs: Path, data_root: Path) -> list[Path]:
    """Case trees in the mount that are NOT part of the data root.

    Any ``case_XX`` dir (with meta.json) found outside ``data_root``
    contributes its grandparent ``X`` (where X/{mr,ct}/... lives) as a
    hotfix root to overlay.
    """
    roots: set[Path] = set()
    data_resolved = data_root.resolve()
    for meta in inputs.rglob("case_*/meta.json"):
        case_dir = meta.parent                      # X/<mod>/<split>/case_XX
        if case_dir.resolve().is_relative_to(data_resolved):
            continue                                # part of the real tree
        try:
            parts = case_dir.relative_to(inputs).parts
        except ValueError:
            continue
        # require <...>/<mr|ct>/<train|val|test>/case_XX so stray folders
        # at the mount root are ignored
        if len(parts) >= 4 and parts[-3] in ("mr", "ct") \
                and parts[-2] in ("train", "val", "test"):
            roots.add(case_dir.parent.parent.parent)
    return sorted(roots)


def overlay(work: Path, fix_roots: list[Path]) -> int:
    """Copy every case dir from the fix roots onto the work tree."""
    n = 0
    for fix in fix_roots:
        for meta in sorted(fix.rglob("case_*/meta.json")):
            case_dir = meta.parent
            rel = case_dir.relative_to(fix)         # e.g. mr/train/case_01
            shutil.copytree(case_dir, work / rel, dirs_exist_ok=True)
            n += 1
            print(f"[kaggle_setup] overlaid {rel} from {fix.name}/")
    return n


def sanity_check(work: Path) -> None:
    """Every case's first exported slice must be on the template grid."""
    bad: list[tuple[Path, tuple]] = []
    checked = 0
    for pat in ("mr/*/case_*", "ct/*/case_*"):
        for case_dir in sorted(work.glob(pat)):
            slices = sorted((case_dir / "slices").glob("*.npy"))
            if not slices:
                continue
            arr = np.load(slices[0], mmap_mode="r")
            checked += 1
            if tuple(arr.shape) != EXPECTED_SLICE:
                bad.append((slices[0].relative_to(work), tuple(arr.shape)))
    if checked == 0:
        raise RuntimeError(f"no cases with slices found under {work}")
    if bad:
        lines = "\n".join(f"  {p}: {s}" for p, s in bad[:10])
        raise RuntimeError(
            f"{len(bad)} case(s) not on the {EXPECTED_SLICE} template grid "
            f"(stale data?):\n{lines}\n"
            "Fix: upload a corrected case folder into the dataset (the "
            "setup overlay applies it automatically)")
    print(f"[kaggle_setup] sanity check OK: {checked} cases, "
          f"slices {EXPECTED_SLICE}")


def prepare(inputs: Path, work: Path, force_copy: bool = False,
            explicit: Path | None = None) -> Path:
    data_root = discover(inputs, explicit)
    print(f"[kaggle_setup] data tree discovered at: {data_root}")

    template_ok = (work / TEMPLATE_REL).exists()
    if force_copy or not template_ok:
        if work.exists():
            shutil.rmtree(work)
        print(f"[kaggle_setup] copying {data_root} -> {work} ...")
        shutil.copytree(data_root, work)
    else:
        print(f"[kaggle_setup] work copy already present: {work}")

    fixes = find_hotfix_roots(inputs, data_root)
    if fixes:
        print(f"[kaggle_setup] hotfix folders found: "
              f"{[f.name for f in fixes]}")
        overlay(work, fixes)
    else:
        print("[kaggle_setup] no hotfix folders")

    sanity_check(work)
    (work / MARKER).write_text(str(work))
    print(f"DATA_ROOT={work}")
    return work


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Kaggle data mount")
    p.add_argument("--inputs", type=Path, default=Path("/kaggle/input"))
    p.add_argument("--work", type=Path,
                   default=Path("/kaggle/working/compass_data"))
    p.add_argument("--data-root", type=Path, default=None,
                   help="skip discovery; use this preprocessed tree")
    p.add_argument("--force-copy", action="store_true",
                   help="re-copy even if the work dir already exists")
    args = p.parse_args()
    try:
        prepare(args.inputs, args.work, args.force_copy, args.data_root)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[kaggle_setup] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
