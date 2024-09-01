#!/usr/bin/env python3
# this script does a little housekeeping for your platform
import os.path
import platform
import subprocess
import sys
from textwrap import dedent
from typing import List, Optional

from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.models.search_scope import SearchScope
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.network.session import PipSession
from pip._internal.req import InstallRequirement
from pip._vendor import tomli as tomllib # replace with tomllib once pyoxidizer supports 3.11
from pip._vendor.packaging.requirements import Requirement
from setuptools import setup, find_packages

"""
The project config.
"""
with open("pyproject.toml", "rb") as f:
    project = tomllib.load(f)["project"]

"""
The package index to the torch built with AMD ROCm.
"""
amd_torch_index = ("https://download.pytorch.org/whl/rocm6.0", "https://download.pytorch.org/whl/nightly/rocm6.1")

"""
The package index to torch built with CUDA.
Observe the CUDA version is in this URL.
"""
nvidia_torch_index = ("https://download.pytorch.org/whl/cu121", "https://download.pytorch.org/whl/nightly/cu124")

"""
The package index to torch built against CPU features.
"""
cpu_torch_index = ("https://download.pytorch.org/whl/cpu", "https://download.pytorch.org/whl/nightly/cpu")

"""
Indicates if this is installing an editable (develop) mode package.
"""
is_editable = "--editable" in sys.argv or "-e" in sys.argv
force_cpu = "--force-cpu" in sys.argv


def _is_nvidia() -> bool:
    system = platform.system().lower()
    nvidia_smi_paths = []

    if system == "windows":
        nvidia_smi_paths.append(os.path.join(os.environ.get("SystemRoot", ""), "System32", "nvidia-smi.exe"))
    elif system == "linux":
        nvidia_smi_paths.extend(["/usr/bin/nvidia-smi", "/opt/nvidia/bin/nvidia-smi"])

    for nvidia_smi_path in nvidia_smi_paths:
        try:
            output = subprocess.check_output([nvidia_smi_path, "-L"]).decode("utf-8")

            if "GPU" in output:
                return True
        except:
            pass

    return False


def _is_amd() -> bool:
    system = platform.system().lower()
    rocminfo_paths = []

    # todo: torch windows doesn't support amd
    if system == "windows":
        rocminfo_paths.append(os.path.join(os.environ.get("ProgramFiles", ""), "AMD", "ROCm", "bin", "rocminfo.exe"))
    elif system == "linux":
        rocminfo_paths.extend(["/opt/rocm/bin/rocminfo", "/usr/bin/rocminfo"])

    for rocminfo_path in rocminfo_paths:
        output = None
        try:
            output = subprocess.check_output([rocminfo_path]).decode("utf-8")
        except:
            pass

        if output is None:
            return False
        elif "Device" in output:
            return True
        elif "Permission Denied" in output:
            msg = dedent(f"""
                    {output}

                    To resolve this issue on AMD:

                    sudo -i
                    usermod -a -G video $LOGNAME
                    usermod -a -G render $LOGNAME

                    You will need to reboot. Save your work, then:

                    reboot

                """)
            print(msg, file=sys.stderr)
            raise RuntimeError(msg)
    return False


def _is_linux_arm64():
    os_name = platform.system()
    architecture = platform.machine()

    return os_name == 'Linux' and architecture == 'aarch64'


def dependencies(force_nightly: bool = False) -> List[str]:
    deps = ["torch", "torchvision", "torchaudio"]

    # Check for an existing torch installation.
    try:
        import torch
        print(f"sdbx setup.py: torch version was {torch.__version__} and built without build isolation, using this torch instead of upgrading", file=sys.stderr)
        deps = [f"torch=={torch.__version__}"] + deps[1:]
        return deps
    except ImportError:
        pass

    # some torch packages redirect to https://download.pytorch.org/whl/
    _alternative_indices = [amd_torch_index, nvidia_torch_index, ("https://download.pytorch.org/whl/", "https://download.pytorch.org/whl/")]
    session = PipSession()

    # (stable, nightly) tuple
    index_urls = [('https://pypi.org/simple', 'https://pypi.org/simple')]

    # prefer nvidia over AMD because AM5/iGPU systems will have a valid ROCm device
    if _is_nvidia() and not force_cpu:
        index_urls = [nvidia_torch_index] + index_urls
    elif _is_amd() and not force_cpu:
        index_urls = [amd_torch_index] + index_urls
        deps += ["pytorch-triton-rocm"]
    else:
        index_urls += [cpu_torch_index]

    # if len(index_urls) == 1:
    #     return deps

    if sys.version_info >= (3, 13) or force_nightly:
        # use the nightlies for python 3.13
        print("Using nightlies for Python 3.13 or higher. PyTorch may not yet build for it", file=sys.stderr)
        index_urls_selected = [nightly for (_, nightly) in index_urls]
        _alternative_indices_selected = [nightly for (_, nightly) in _alternative_indices]
    else:
        index_urls_selected = [stable for (stable, _) in index_urls]
        _alternative_indices_selected = [stable for (stable, _) in _alternative_indices]
    try:
        # pip 23, 24
        finder = PackageFinder.create(LinkCollector(session, SearchScope([], index_urls_selected, no_index=False)),
                                      SelectionPreferences(allow_yanked=False, prefer_binary=False,
                                                           allow_all_prereleases=True))
    except:
        try:
            # pip 22
            finder = PackageFinder.create(LinkCollector(session, SearchScope([], index_urls_selected)),  # type: ignore
                                          SelectionPreferences(allow_yanked=False, prefer_binary=False,
                                                               allow_all_prereleases=True), 
                                          use_deprecated_html5lib=False)
        except:
            raise Exception("upgrade pip with\npython -m pip install -U pip")
    for i, package in enumerate(deps[:]):
        print(f"Checking {package} for a better version", file=sys.stderr)
        requirement = InstallRequirement(Requirement(package), comes_from=project["name"])
        candidate = finder.find_best_candidate(requirement.name, requirement.specifier)
        if candidate.best_candidate is not None:
            if any([url in candidate.best_candidate.link.url for url in _alternative_indices_selected]):
                deps[i] = f"{requirement.name} @ {candidate.best_candidate.link.url}"
    return deps

setup(
    install_requires=dependencies() + project["optional-dependencies"]["core"]
)
