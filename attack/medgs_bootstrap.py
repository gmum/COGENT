"""
Adds the MedGS repository to sys.path so its modules can be imported.

The MedGS root is resolved in this order:
  1. ``MEDGS_ROOT`` environment variable
  2. ``--medgs_root`` command-line flag (or ``--medgs_root=...``)
  3. Common locations next to this file or the current working directory
"""
import os
import sys


def bootstrap_medgs_path():
    """Find the MedGS repository and prepend it to ``sys.path``.

    Returns:
        The absolute path that was added, or ``None`` if no MedGS root
        was found in any candidate location.
    """
    candidates = []

    env = os.environ.get("MEDGS_ROOT")
    if env:
        candidates.append(env)

    for i, a in enumerate(sys.argv):
        if a == "--medgs_root" and i + 1 < len(sys.argv):
            candidates.append(sys.argv[i + 1])
        elif a.startswith("--medgs_root="):
            candidates.append(a.split("=", 1)[1])

    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        os.path.join(here, "..", "MedGS"),
        os.path.join(os.getcwd(), "MedGS"),
        here,
        os.getcwd(),
    ]

    for c in candidates:
        if not c:
            continue
        c_abs = os.path.abspath(c)
        if (os.path.isdir(os.path.join(c_abs, "scene"))
                and os.path.isdir(os.path.join(c_abs, "gaussian_renderer"))):
            if c_abs not in sys.path:
                sys.path.insert(0, c_abs)
            return c_abs
    return None
