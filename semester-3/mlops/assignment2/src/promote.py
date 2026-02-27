import shutil
import os
from src.versioning import get_latest_version, get_next_version

def promote():
    dev_version = get_latest_version("models/dev")
    prod_version = get_next_version("models/prod")

    os.makedirs("models/prod", exist_ok=True)

    shutil.copytree(
        f"models/dev/{dev_version}",
        f"models/prod/{prod_version}"
    )

    print(f"🚀 Promoted dev/{dev_version} ➜ prod/{prod_version}")

if __name__ == "__main__":
    promote()
