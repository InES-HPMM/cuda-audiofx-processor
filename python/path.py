import os
import sys
from pathlib import Path

# project_root_env = "PROJECT_DIR"
# project_root = os.getenv(project_root_env)
# if project_root == None:
#     raise EnvironmentError(f"{project_root_env} env variable is not set")
project_root = "/home/nvidia/git/mt"


def root(*argv) -> str: return os.path.join(project_root, *argv)
def res(*argv) -> str: return root(os.path.join("res", *argv))
def out(*argv) -> str: return root(os.path.join("out", *argv))
def ir(*argv) -> str: return res(os.path.join("ir", *argv))


def mkdir(path: str, parents=True, exist_ok=True): Path(path).mkdir(parents=parents, exist_ok=exist_ok)
