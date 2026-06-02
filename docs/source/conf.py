from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

project = "zppy-interfaces"
author = "E3SM Project"
copyright = f"{datetime.now():%Y}, {author}"

_repo_root = Path(__file__).resolve().parents[2]
_version_file = _repo_root / "zppy_interfaces" / "version.py"
_version_match = re.search(
    r'__version__\s*=\s*"([^"]+)"', _version_file.read_text(encoding="utf-8")
)
release = _version_match.group(1) if _version_match else "unknown"
version = release

extensions = []
templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["style.css"]
html_title = "zppy-interfaces documentation"
html_sidebars = {
    "**": [
        "globaltoc.html",
        "localtoc.html",
        "relations.html",
        "searchbox.html",
        "versions.html",
    ]
}

smv_tag_whitelist = r"^v.*$"
smv_branch_whitelist = r"^(main|master)$"
smv_remote_whitelist = r"^origin$"
