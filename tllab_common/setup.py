import os
import setuptools
import git

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tllab_common",
    packages=['tllab_common'],
    package_dir={'tllab_common': os.path.dirname(__file__)},
    version=[i for i in git.Git('.').log('-1', '--date=format:%Y%m%d%H%M').splitlines() if i.startswith('Date:')][0][-12:],
    author="Lenstra lab NKI",
    author_email="t.lenstra@nki.nl",
    description="Common code for the Lenstra lab.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.rhpc.nki.nl/LenstraLab/tllab_common",
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
    install_requires=['untangle', 'python-javabridge', 'python-bioformats', 'pandas', 'psutil', 'numpy', 'tqdm',
                      'tifffile', 'czifile', 'pyyaml', 'dill', 'colorcet', 'multipledispatch', 'pytest-xdist', 'numba',
                      'scipy'],
    scripts=['bin/wimread'],
)