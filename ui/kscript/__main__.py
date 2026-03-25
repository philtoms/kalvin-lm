"""Entry point for running the KScript TUI app."""

import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

from ui.kscript.app import main

if __name__ == "__main__":
    main()
