#!/usr/bin/python

import sys, os
import numpy as np
from wimread import imread
from findcells import findcells
from tiffwrite import IJTiffWriter, tiffwrite
from itertools import product

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
# wp@tl20200124


def test_IJTiffWriter(tmp_path):
    tmp_path = str(tmp_path)
    shape = (2, 3, 4)
    dshape = (64, 128)
    data = np.random.randint(0, 255, dshape + shape[1:])
    file = os.path.join(tmp_path, 'test.tif')
    with IJTiffWriter(file, shape) as tif:
        for n in product(*[range(i) for i in shape[1:]]):
            tif.save(data[..., n[0], n[1]], 1, *n)

    with imread(file) as im:
        assert im.shape == dshape + shape, 'shape is wrong'
        for n in product(*[range(i) for i in shape]):
            if n[0] == 0:
                assert np.all(im(*n) == 0), 'zero frames not filled'
            else:
                assert np.all(im(*n) == data[..., n[1], n[2]]), 'data not stored correctly'
    # shutil.rmtree(tmp_path, True)


def test_tiffwrite(tmp_path):
    tmp_path = str(tmp_path)
    shape = (2, 3, 4)
    dshape = (64, 128)
    data = np.random.randint(0, 255, dshape + shape, dtype='uint8')
    file = os.path.join(tmp_path, 'test2.tif')

    tiffwrite(file, data, 'XYCZT')
    with imread(file) as im:
        assert im.shape == dshape + shape, 'shape is wrong'
        for n in product(*[range(i) for i in shape]):
            assert np.all(im(*n) == data[..., n[0], n[1], n[2]]), 'data not stored correctly'


def test_findcell_a(tmp_path):
    tmp_path = str(tmp_path)
    with imread(os.path.join(test_files, 'findcell.a.tif')) as a:
        c, n = findcells(a(0), a(1), ccdist=150, thres=1, removeborders=True)
        assert np.all(c==a(2)), 'Cellmask wrong'
        assert np.all(n==a(3)), 'Nucleusmask wrong'
    files = [os.path.join(tmp_path, f) for f in ('cell.tif', 'nucleus.tif')]
    with IJTiffWriter(files, (1, 1, 1)) as tif:
        for i, f in enumerate((c, n)):
            tif.save(i, f, 0, 0, 0)

    for file, f in zip(files, (c, n)):
        with imread(file) as im:
            assert np.all(im(0) == f), 'data not stored correctly'

## ----- This part runs the tests -----

if __name__ == '__main__':
    if len(sys.argv) < 2:
        py = ['2.7', '3.8']
    else:
        py = sys.argv[1:]

    for p in py:
        print('Testing using python {}'.format(p))
        os.system('python{} -m pytest -n=12 -p no:warnings --verbose {}'.format(p, fname))
        print('')

    imread.kill_vm()
