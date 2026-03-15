from setuptools import setup, find_packages

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("./requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="video-2d-to-3d-converter",
    version="1.0.0",
    author="Abid Shafi",
    author_email="admin@pointer.pk",
    description="Convert 2D videos to 3D Side-by-Side format for VR viewing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PointerSoftware/2D-to-3D-SBS-Converter",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Video :: Conversion",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "video-2d-to-3d=converter.app:main",
        ],
    },
)