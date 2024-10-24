from methods import BFGS
import numpy as np

# Sample Test Problem for BFGS
def testFunc(arr):
    return arr[0] - arr[1] + 2*arr[0]**2 + 2*arr[0]*arr[1] + arr[1]**2

x0 = [0, 0]

sol = BFGS(testFunc,x0)
assert(np.array_equal(sol, np.array([-1, 1.5])))