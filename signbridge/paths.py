"""Where bundled files (models) live: the app bundle's Resources folder, or the repo root in development."""

import os

ROOT = os.environ.get("SIGNBRIDGE_RESOURCES") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource(path):
    """Absolute path for a repo-relative resource path like models/popsign.npz."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)
