import os
import re

VERSION_REGEX = re.compile(r"v(\d+)")

def get_next_version(base_path: str) -> str:
    """
    Returns next version as v1, v2, v3... based on existing folders.
    Creates base_path if it does not exist.
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        return "v1"

    versions = []
    for name in os.listdir(base_path):
        match = VERSION_REGEX.fullmatch(name)
        if match:
            versions.append(int(match.group(1)))

    if not versions:
        return "v1"

    return f"v{max(versions) + 1}"


def get_latest_version(base_path: str) -> str:
    """
    Returns latest version folder (e.g., v3).
    Creates base_path if it does not exist.
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        return "v1"

    versions = []
    for name in os.listdir(base_path):
        match = VERSION_REGEX.fullmatch(name)
        if match:
            versions.append(int(match.group(1)))

    if not versions:
        return "v1"

    return f"v{max(versions)}"
