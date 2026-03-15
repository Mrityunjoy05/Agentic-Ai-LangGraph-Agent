
import sys
from pathlib import Path

# Add parent directory to sys.path for modular imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

print(parent_dir)
print(Path(__file__))

