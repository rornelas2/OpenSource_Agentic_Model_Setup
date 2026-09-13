"""Resume the complete Muse download over HTTP, including both model shards."""

import os
from pathlib import Path


if __name__ == "__main__":
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    script = Path(__file__).resolve().with_name("download-muse.sh")
    os.execvp("bash", ["bash", str(script)])
