import sys
sys.path.insert(0, '../src')
from boxdomain import BoxDomain
import numpy as np




# def test_fmatrix(self):
#     assert

if __name__ == '__main__':
    lbs = np.array([1, 2, 3, 4, 5, 6])
    ubs = np.array([10, 20, 3, 40, 50, 6])
    box = BoxDomain(lbs, ubs)
    fv = box.toFacetVertex()