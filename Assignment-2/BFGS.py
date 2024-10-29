# The file contains the method to minimize an objective function
# using BFGS method and Genetic Algorthim (Real-coded GA).


import numpy as np
import sympy as sp

def derv(func, variables):

    nVariables = len(variables)

    # Empty array for gradient functions
    df = np.empty(nVariables, dtype=object)  

    # Compute the gradient with respect to each input variable
    for i in range(nVariables):
        df[i] = sp.diff(func, variables[i])  # Compute gradient w.r.t. i-th variable

    return df

def dervAtPoint(der, var, values):
    val = dict(zip(var,values))  
    res = []
    for i in range(len(val)):
        res.append(der[i].subs(val).evalf())

    return res

def backtracking_line_search(fun, x, grad, direction, alpha=1.0, rho=0.8, c=1e-4):
    """
    Perform backtracking line search to find the optimal step size.
    
    Parameters:
    - fun: objective function
    - x: current point as a numpy array
    - grad: gradient at the current point
    - direction: search direction
    - alpha: initial step size (default is 1)
    - rho: factor to decrease alpha (0 < rho < 1), typically 0.8
    - c: Armijo condition constant, typically 1e-4

    Returns:
    - optimal alpha satisfying the Armijo condition
    """
    fx = fun(x)
    while fun(x + alpha * direction) > fx + c * alpha * np.dot(grad, direction):
        alpha *= rho  # Reduce step size by a factor of rho
    return alpha

def cauchyMethod(fun, x0):
    '''
    This function implements the cauchy's method.

    1. Input: Function, X0
    2. Compute the number of variables and create symbolic variables
    3. Create a symbolic function
    4. Find the gradient of the function
    5. Determine the step length
    6. Return the value of X1
    '''

    # Find the number of variables
    nVar = len(x0)

    # create an array of symbolic variables
    var = sp.symbols(f'x0:{nVar}')

    # create a symbolic function
    func = fun(var)

    # finding the gradient of the func
    der = derv(func, var)

    derX0 = dervAtPoint(der,var,x0)

    if derX0 == [0]*nVar:
        return x0
    
    dir = -np.transpose(derX0)
    step =  backtracking_line_search(fun,np.array(x0, dtype=float),derX0, dir)
    x1 = np.transpose(x0) + step*dir
    return x1

def matrixUpdate(B0, g, d):

    # Ensure g and d are column vectors
    g = g.reshape(-1, 1)  
    d = d.reshape(-1, 1) 

    # Calculate denominator
    den = np.dot(np.transpose(d), g)[0, 0] 

    # Update the matrix using the BFGS formula
    t1 = (1 + np.dot(np.transpose(g), B0) @ g / den) * (d @ np.transpose(d) / den)
    t2 = B0 @ g @ np.transpose(d) / den
    t3 = d @ np.transpose(g) @ B0 / den
    
    # Update B0
    B1 = B0 + t1 - t2 - t3

    return B1


def check(dx1, eps, x0, x1):
    a1 = np.all(np.abs(dx1) < eps)
    e = 1e-8
    a2 = np.all(np.abs(x1-x0) < e)
    return not (a1 or a2)

def BFGS(fun, x0, eps=1e-15):
    '''
    This function implements the BFGS.

    Input: Function, X0
    Interation loop until gradient = 0

    Intiate the loop using x1, df/dx at x1 from Cauchy's solution

    Within loop:
        1. compute g and d
        2. update B matrix
        3. update the value of x1 and x0 = x1
        4. update the value of df/dx at x1 and x0
    '''
    # Find the number of variables
    nVar = len(x0)

    # create an array of symbolic variables
    var = sp.symbols(f'x0:{nVar}')

    # create a symbolic function
    func = fun(var)

    # finding the gradient of the func
    der = derv(func, var)

    #find the first solution using cauchy's method
    x1 = cauchyMethod(fun, x0)

    #find the gradients at x0 and x1
    dx0 = dervAtPoint(der,var,x0)
    dx1 = dervAtPoint(der,var,x1)

    B0 = np.array([[1, 0], [0, 1]])

    while check(dx1,eps,x0,x1):

        # find g and d vectors
        g = np.transpose(dx1) - np.transpose(dx0)
        d = np.transpose(x1) - np.transpose(x0)

        # update B matrix
        B1 = matrixUpdate(B0,g,d)
    
        # compute optimal step length
        dir = -B1@np.transpose(dx1)
        step =  backtracking_line_search(fun, np.array(x1, dtype=float), dx1, dir)

        x0 = x1
        x1 = np.transpose(x1) + step*dir
        dx0 = dx1
        dx1 = dervAtPoint(der,var,x1)
        B0 = B1
    
    return np.round(np.array(x1,dtype='float'), 5)