"""File scanning needed by evaluation; original diffusion utilities are archived."""
import os


def scan_files_in_dir(directory, postfix=None, progress_bar=None):
    files = []
    with os.scandir(directory) as entries:
        for entry in sorted(entries, key=lambda e: e.name):
            if entry.is_file():
                if postfix is None or os.path.splitext(entry.name)[1].lower() in postfix:
                    files.append(entry)
            elif entry.is_dir(follow_symlinks=False):
                files.extend(scan_files_in_dir(entry.path, postfix))
    return files
