"""Root conftest: ensure artifact root is on sys.path so tests can import src.*"""
import sys
from pathlib import Path

# Add the artifact root to sys.path so that `import src.xxx` resolves correctly
# regardless of the working directory pytest is launched from.
artifact_root = Path(__file__).parent
if str(artifact_root) not in sys.path:
    sys.path.insert(0, str(artifact_root))
