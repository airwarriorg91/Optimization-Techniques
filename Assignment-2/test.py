from methods import BFGS
import numpy as np
import sympy as sp

# Sample Test Problem for BFGS
def testFunc(arr):
    return arr[0] - arr[1] + 2*arr[0]**2 + 2*arr[0]*arr[1] + arr[1]**2


# Unimodal Benchmark problem

def unimodalBenchmark(arr):
    # Bohachevsky Unimodal function
    # f(x,y) = x^2 + 2*y^2 - 0.3*cos(3*pi*x) - 0.4*cos(4*pi*y) + 0.7
    # Minimum at (0,0)
    return arr[0]**2 + 2*(arr[1]**2) - 0.3*sp.cos(3*sp.pi*arr[0]) - 0.4*sp.cos(4*sp.pi*arr[1]) + 0.7

# Multimodal Benchmark problem

def multimodalBenchmark(arr):
    # Ackley Function
    # f(x,y) = -20exp(-0.2*sqrt(0.5*(x^2 + y^2))) - exp(0.5*(cos(2pix) + cos(2piy))) + 20 + exp(1)
    # Global Minimum at (0,0)
    return -20*sp.exp(-0.2*sp.sqrt(0.5*(arr[0]**2 + arr[1]**2))) - sp.exp(0.5*(sp.cos(2*sp.pi*arr[0]) + sp.cos(2*sp.pi*arr[1]))) + 20 + sp.exp(1)

#BFGS Test

x0 = [0, 0]
sol = BFGS(testFunc,x0)
assert(np.array_equal(sol, np.array([-1, 1.5])))

#Test for the unimodal function with global minima at [0,0]
x0 = [-5,5]
sol = BFGS(unimodalBenchmark,x0)
assert(np.array_equal(sol, np.array([0,0])))

#Test for the multimodal function with global minima at [0,0]

x0 = [-5,5]
sol = BFGS(multimodalBenchmark,x0)
assert(not np.array_equal(sol, np.array([0,0]))) # Converges to a local maxima

x0 = [-0.1,0.1]
sol = BFGS(multimodalBenchmark,x0)
assert(np.array_equal(sol, np.array([0,0]))) # Converges to the global maxima