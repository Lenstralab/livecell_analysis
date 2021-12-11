import setuptools
import git
import os
import site
import shutil
from setuptools.command.develop import develop
from setuptools.command.install import install

with open("README.md", "r") as fh:
    long_description = fh.read()


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        install_scripts()


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        install_scripts()


def install_scripts():
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin')
    dest = os.path.join(site.getuserbase(), 'bin')
    os.makedirs(dest, exist_ok=True)
    for s in os.listdir(src):
        shutil.copy(os.path.join(src, s), os.path.join(dest, s))
    add_path()


def add_path(path='$HOME/.local/bin', file=None):
    line = f'PATH="{path}:$PATH"'
    file = file or os.path.join(os.environ['HOME'], '.bashrc')
    if os.path.exists(file) and os.path.join(site.getuserbase(), 'bin') not in os.environ['PATH'].split(':'):
        with open(file, 'r') as f:
            lines = f.readlines()
        if line not in lines:
            with open(file, 'a') as f:
                f.write('\n' + line)


setuptools.setup(
    name="livecellanalysis",
    version=[i for i in git.Git('.').log('-1', '--date=format:%Y%m%d%H%M').splitlines() if i.startswith('Date:')][0][-12:],
    author="Lenstra lab NKI",
    author_email="t.lenstra@nki.nl",
    description="Live cell analysis code for the Lenstra lab.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.rhpc.nki.nl/LenstraLab/LiveCellAnalysis",
    packages=['LiveCellAnalysis'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    setup_requires=['git-python'],
    tests_require=['pytest-xdist'],
    install_requires=['numpy', 'scipy', 'tqdm', 'matplotlib', 'lfdfiles', 'parfor', 'pyyaml', 'scikit-image', 'psutil',
                      'Pillow', 'hidden_markov'],
    cmdclass={'develop': PostDevelopCommand, 'install': PostInstallCommand},
    scripts=[os.path.join('bin', script) for script in
             os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin'))],
)
