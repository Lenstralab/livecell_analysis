#!/usr/bin/python

import sys, os, shutil, yaml, copy
import numpy as np
from tllab_common.misc import getConfig
from LiveCellAnalysis.pipeline_livecell_track_movies import pipeline, getPaths, calculate_general_parameters
import tllab_common
from tllab_common.wimread import imread

fname = os.path.realpath(__file__)
test_files = os.path.join(os.path.dirname(fname), 'test_files')

# This file defines tests to be run to assert the correct working of our scripts
# after updates. Add a test below as a function, name starting with 'test', and 
# optionally using 'assert'.
#
# Place extra files used for these tests in the folder test_files, add imports 
# above this text.
#
# Then navigate to the directory containing this file and run ./test.py directly
# from the terminal. If you see red text then something is wrong and you need to
# fix the code before committing to gitlab.
#
#wp@tl20200124

## ----- add your tests here -----
    
def make_test_pipeline(parameter_file):
    def pipeline_fun(tmp_path):
        tmp_path = str(tmp_path)
        parameters = getConfig(os.path.join(test_files, parameter_file))
        parameters['outputfolder'] = tmp_path
        tmp_parameter_file = os.path.join(tmp_path, 'params.yml')

        with open(tmp_parameter_file, 'w') as f:
            yaml.dump(parameters, f)
        if parameters['TSregfile']:
            p = copy.deepcopy(parameters)
            calculate_general_parameters(p, parameter_file)
            for exp in range(0, p['lenExpList']):
                p['pathIn'] = p['expList'][exp]
                getPaths(p)

                if not os.path.exists(p['pathOut']):
                    os.makedirs(p['pathOut'])

                for file in parameters['TSregfile'][exp]:
                    shutil.copyfile(os.path.join(test_files, file+'.txt'), os.path.join(p['pathOut'], file+'.txt'))

        parameters = pipeline(tmp_parameter_file)

        files = ('max.tif', 'pipeline_livecell_track_movies.py', 'sum.tif', 'params.yml')

        assert os.path.exists(parameters['pathOut']), 'Output folder has not been generated'
        assert len(os.listdir(parameters['pathOut']))>4, 'There aren''t enough files in the output folder'
        for file in files:
            assert os.path.exists(parameters['expPathOut']+'_'+file),\
                'File {} has not been generated'.format(parameters['expPathOut']+'_'+file)
    return pipeline_fun

test_track  = make_test_pipeline('pipeline_livecell_track_movies_test.yml')

## ----- This part runs the tests -----
    
if __name__ == '__main__':
    if len(sys.argv)<2:
        py = ['3.8']
    else:
        py = sys.argv[1:]

    for p in py:
        print('Testing using python {}'.format(p))
        print('Loaded tllab_common code from {}'.format(tllab_common.__file__))
        os.system('python{} -m pytest -n=12 -p no:warnings --verbose {}'.format(p, fname))
        print('')

    imread.kill_vm()
