import glob
import os
from pathlib import Path
from setuptools import find_packages, setup
from os import path

import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

PROJECT_NAME = "mmkg-video-hoi"
PACKAGE_NAME = PROJECT_NAME.replace("-", "_")
DESCRIPTION = "MMKG Models"

TORCH_VERSION = [int(x) for x in torch.__version__.split(".")[:2]]
assert TORCH_VERSION >= [1, 13], "Requires PyTorch >= 1.13"


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(
        this_dir, "mmkg_video_hoi", "detectron2", "layers", "csrc"
    )

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and (CUDA_HOME is not None)) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda

        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-O3",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "mmkg_video_hoi.detectron2._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


if __name__ == "__main__":
    version = "0.1.0"

    print(f"Building {PROJECT_NAME}-{version}")

    setup(
        name=PROJECT_NAME,
        version=version,
        author="Ming Yang",
        author_email="ymviv@qq.com",
        url=f"https://github.com/vivym/{PROJECT_NAME}",
        download_url=f"https://github.com/vivym/{PROJECT_NAME}/tags",
        description=DESCRIPTION,
        long_description=Path("README.md").read_text(),
        long_description_content_type="text/markdown",
        packages=find_packages(exclude=("tests",)),
        package_data={PACKAGE_NAME: ["*.dll", "*.so", "*.dylib", "*.txt", "*.txt.gz", "*.yml", "*.yaml"]},
        zip_safe=False,
        python_requires=">=3.9",
        install_requires=[
            "pillow",
            "aiofiles",
            "fastapi",
            "uvicorn[standard]",
            "python-multipart",
            "tqdm",
            "python-opencv",
            "simplejson",
            "pandas",
            "av",
            "psutil",
        ],
        ext_modules=get_extensions(),
        cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    )
