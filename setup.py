import setuptools
from setuptools.command.install import install

with open("README.md", "r") as fh:
    long_description = fh.read()


def requirements():
    # The dependencies are the same as the contents of requirements.txt
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip()]


setuptools.setup(
    name="pandda_gemmi",
    version="0.0.1",
    author="Conor Francis Wild; Nick Pearce",
    author_email="conor.wild@sky.com",
    description="A package for handling many crystalographic datasets simultainiously",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/ConorFWild/pandda_gemmi.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=requirements(),
    scripts=["pandda_gemmi/pandda_rhofit.sh",]
)
