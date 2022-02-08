# Live Cell Analysis
as used by Brouwer et al.

## Installation
If not done already:
- Install python (at least 3.7): https://www.python.org
- Install pip and git
- First install SimpleElastix: https://simpleelastix.readthedocs.io/GettingStarted.html

Then install the livecell scripts (up to 5 minutes):

    git clone https://github.com/Lenstralab/livecell_analysis.git
    pip install -e livecell_analysis --user

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
Prepare your parameter file, see pipeline_livecell_correlationfunctions_parameters_template.yml for an example.

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

## Running the demo
### Tracking

    cd livecell_analysis
    livecell_track_movies test_files/pipeline_livecell_track_movies_test.yml

The script will go through the various stages of the analysis and normally finish in about 10 minutes.
It will make a folder 'demo_output' (defined in the parameter .yml file) which contains results in .tif, .txt and .pdf
formats.

### Correlations
    cd livecell_analysis (if not done already in the previous step)
    livecell_correlationfunctions test_files/pipeline_livecell_correlationfunctions_test.yml

The script will go through the various stages of the analysis and normally finish in about 5 minutes.
It will make a folder 'demo_output_correlations' (defined in the parameter .yml file) which contains results in .tif,
.txt and .pdf formats.

## Testing
This script was tested with python 3.8 on Ubuntu 20.04
