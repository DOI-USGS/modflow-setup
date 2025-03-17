import os
import subprocess as sp
from pathlib import Path


def test_import():
    """Test that Modflow-setup is installed, and can be imported
    (from another location besides the repo top-level, which contains the
    'mfsetup' folder)."""
    os.system("python -c 'import mfsetup'")
    results = sp.check_call(["python", "-c", "import mfsetup"], cwd=Path('..'))
