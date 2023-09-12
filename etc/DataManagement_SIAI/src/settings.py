import os
from pathlib import Path


def get_proj_root():
    """Returns the path of project root directory

    Returns:
        str: text sequence type (strings) of the path of project root directory
    """
    return (
        os.path.dirname(Path(__file__).resolve().parent)
        if "__file__" in locals() or "__file__" in globals()
        else os.path.dirname(os.path.realpath("__file__"))
    )


PROJ_ROOT = get_proj_root()
DATA_ROOT = os.path.join(PROJ_ROOT, "data")
SRC_ROOT = os.path.join(PROJ_ROOT, "src")
QUERY_ROOT = os.path.join(PROJ_ROOT, "data\\query_results")

if __name__ == "__main__":
    print(f"PROJ_ROOT: {PROJ_ROOT}")
