"""Download the complete pinned Muse checkpoint to the current user's scratch."""

import os
from pathlib import Path


if __name__ == "__main__":
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["MUSE_MODEL_DIR"] = f"/scratch/{os.environ['USER']}/pinnacles-agents/models/Muse-Glimmer-30B"
    script = Path(__file__).resolve().with_name("download-muse.sh")
    os.execvp("bash", ["bash", str(script)])
