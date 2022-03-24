import sys
sys.path.insert(0, '../../src')

from vnnlib import get_num_inputs_outputs, read_vnnlib_simple

vnnlib_filename = 'nets/prop_6.vnnlib'
vnnlib_spec = read_vnnlib_simple(vnnlib_filename, 5, 5)
xx = 1