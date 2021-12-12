# Live Cell Analysis
as used by Brouwer et al.

## Installation
    git clone --recurse-submodules https://gitlab.rhpc.nki.nl/LenstraLab/LiveCellAnalysis.git
    cd LiveCellAnalysis
    git submodule foreach --recursive git checkout master
    pip install -e . --user

If you forget about the --recurse-submodules part, the submodule tllab_common will not be pulled from git, but the
version installed on curie will be used instead.

## Updating
### PyCharm
Git > Update Project... (Ctrl+T)

## Usage
### Track Movies
Prepare your parameter file, see pipeline_livecell_track_movies_parameters_template.yml for an example.

From the terminal:

    livecell_track_movies /path/to/parameter_file.yml

or:
    
    cd LiveCellAnalysis
    ./pipeline_livecell_track_movies.py /path/to/parameter_file.yml
or:
   
    cd LiveCellAnalysis
    ipython
    %run pipeline_livecell_track_movies.py /path/to/parameter_file.yml

### Correlation Functions
Prepare you parameter file, see pipeline_livecell_correlationfunctions_parameters_template.yml for an example.

From the terminal:

    livecell_correlationfunctions /path/to/parameter_file.yml
or:

    cd LiveCellAnalysis
    ./pipeline_livecell_correlationfunctions.py /path/to/parameter_file.yml
or:
   
    cd LiveCellAnalysis
    ipython
    %run pipeline_livecell_correlationfunctions.py /path/to/parameter_file.yml

### Bootstrap Testing
Find out how to use from the terminal:

    bootstrap_test --help

### Edit list.py files through a GUI
Find out how to use from the terminal:

    listpyedit --help

### Convert orbital tracking data to tracks and a list.py
Find out how to use from the terminal:

    orbital2listpy --help
