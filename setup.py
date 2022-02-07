import setuptools
import os


with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='livecellanalysis',
    version='2021.12.0',
    author='Lenstra lab NKI',
    author_email='t.lenstra@nki.nl',
    description='Live cell analysis code for the Lenstra lab.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Lenstralab/livecell_analysis.git',
    packages=['LiveCellAnalysis'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    tests_require=['pytest-xdist'],
    install_requires=['numpy', 'scipy', 'tqdm', 'matplotlib', 'lfdfiles', 'parfor', 'pyyaml', 'scikit-image', 'psutil',
                      'Pillow', 'hidden_markov',
                      'tllab_common@git+https://github.com/Lenstralab/tllab_common.git@45d5e61faffd39c4b4504745a8647253bd48f80d'],
    scripts=[os.path.join('bin', script) for script in
             os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin'))],
)
