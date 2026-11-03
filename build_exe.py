"""
Build automation script for Essentials Photo Gallery.
Executes PyInstaller build using photogallery.spec and verifies output integrity.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    print("==================================================")
    print(" Essentials Photo Gallery - PyInstaller Build")
    print("==================================================")
    print(f"Project root: {project_root}")
    print(f"Python interpreter: {sys.executable} ({sys.version.split()[0]})")

    # 1. Clean previous build and dist artifacts if needed
    build_dir = project_root / "build"
    dist_dir = project_root / "dist" / "EssentialsPhotoGallery"

    if build_dir.exists():
        print(f"Cleaning existing build directory: {build_dir}")
        try:
            shutil.rmtree(build_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: could not fully clean build directory: {e}")

    spec_file = project_root / "photogallery.spec"
    if not spec_file.exists():
        print(f"ERROR: Spec file not found at {spec_file}")
        return 1

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        str(spec_file),
    ]

    print("\nRunning command:")
    print(" ".join(cmd))
    print("--------------------------------------------------")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nERROR: PyInstaller build failed!")
        return result.returncode

    exe_path = dist_dir / "EssentialsPhotoGallery.exe"
    if not exe_path.exists():
        print(f"\nERROR: Expected executable not found at: {exe_path}")
        return 1

    print("\n==================================================")
    print(" Build completed successfully!")
    print(f" Output executable: {exe_path}")
    print(" Directory ready for Inno Setup installer packaging.")
    print("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
