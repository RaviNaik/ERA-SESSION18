from pathlib import Path


def create_dirs(dirs, parent):
    parent = Path(parent)
    for dir in dirs:
        child_dir = Path(parent, dir)
        child_dir.mkdir(parents=True, exist_ok=True)
