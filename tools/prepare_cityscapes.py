"""Reshape the original cityscapes layout into the layout this repo expects.

Before
SelfSupervisedFiseyeRectification
└── data
    └── cityscapes
        └── leftImg8bit_trainvaltest
            └── leftImg8bit
                ├── train
                │   ├── aachen
                │   ├── bochum
                │   ...
                ├── val
                └── test

After
SelfSupervisedFiseyeRectification
└── data
    └── cityscapes
        ├── train
        │   ├── aachen
        │   ├── bochum
        │   ...
        ├── val
        ├── test
        ├── train.lst
        ├── val.lst
        └── test.lst

Splits are *file lists*, not folders, because that is what
``src/datasets/base_dataset.py`` reads. The previous version of this script was a
docstring with no code, so the ``.lst`` files the datasets require were never created.

By default images are moved; ``--copy`` keeps the download intact, ``--link`` hardlinks
(falls back to copy across filesystems), and ``--dry_run`` prints what would happen.
"""

import argparse
import os
import shutil
import sys

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def findSource(root):
    """Locate leftImg8bit inside the downloaded archive layout."""
    candidates = [
        os.path.join(root, "leftImg8bit_trainvaltest", "leftImg8bit"),
        os.path.join(root, "leftImg8bit"),
        root,
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) and any(
            os.path.isdir(os.path.join(candidate, split)) for split in SPLITS
        ):
            return candidate
    raise FileNotFoundError(
        "could not find leftImg8bit under {} (expected a directory containing "
        "{} after unpacking leftImg8bit_trainvaltest.zip)".format(
            root, "/".join(SPLITS)
        )
    )


def collect(source):
    """{split: [absolute image paths]} in a stable order."""
    listing = {}
    for split in SPLITS:
        split_dir = os.path.join(source, split)
        files = []
        if not os.path.isdir(split_dir):
            print("WARNING: {} has no '{}' split, skipping it".format(source, split))
        else:
            for city in sorted(os.listdir(split_dir)):
                city_dir = os.path.join(split_dir, city)
                if not os.path.isdir(city_dir):
                    continue
                for name in sorted(os.listdir(city_dir)):
                    if name.lower().endswith(IMAGE_EXTENSIONS):
                        files.append(os.path.abspath(os.path.join(city_dir, name)))
        listing[split] = files
    return listing


def place(path, destination, mode):
    if os.path.exists(destination):
        return "skip"
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if mode == "dry_run":
        return "dry"
    if mode == "copy":
        shutil.copy2(path, destination)
    elif mode == "link":
        try:
            os.link(path, destination)
        except OSError:  # different filesystem
            shutil.copy2(path, destination)
    else:
        shutil.move(path, destination)
    return "ok"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default="data/cityscapes",
                        help="dataset folder (default: data/cityscapes)")
    parser.add_argument("--mode", choices=("move", "copy", "link", "dry_run"), default="move")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.data_root)
    if not os.path.isdir(root):
        parser.error("no such dataset directory: {} (download cityscapes and put it "
                     "under data/)".format(root))

    source = findSource(root)
    print("source: {}".format(source))
    listing = collect(source)

    total = 0
    for split in SPLITS:
        files, counts = [], {"ok": 0, "skip": 0, "dry": 0}
        for path in listing[split]:
            destination = os.path.abspath(os.path.join(root, os.path.relpath(path, source)))
            if destination != path:  # already in place -> nothing to do
                counts[place(path, destination, args.mode)] += 1
            files.append(destination)

        listing[split] = files
        total += len(files)
        print("{:<6} {:>6} images  ({} moved/copied, {} already there)".format(
            split, len(files), counts["ok"], counts["skip"]))

        list_path = os.path.join(root, "{}.lst".format(split))
        with open(list_path, "w", encoding="utf-8") as f:
            f.write("\n".join(files) + ("\n" if files else ""))
        print("=> wrote {} (absolute paths)".format(list_path))

    if total == 0:
        print("nothing found -- is {} unpacked?".format(source), file=sys.stderr)
        return 1
    print("\nNext: python3 src/train.py cfg/cityscapes.yml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
