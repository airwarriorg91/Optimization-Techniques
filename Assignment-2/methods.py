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
    val = createValueArray(var, values)
    res = []
    for i in range(len(val)):
        res.append(der[i].subs(val))

    return res

def createValueArray(var, values):
    val = {}
    for i in range(len(var)):
        val[str(var[i])] = values[i]

    return val

def cauchyMethod(fun, x0):
    '''
    This function implements the cauchy's method.

    1. Input: Function, X0
    2. Compute the number of variables and create symbolic variables
    3. Create a symbolic function
    4. Find the gradient of the function
    5. Find X1 in terms of A 
    6. Determine A
    7. Return the value of X1
    '''

    # Find the number of variables
    nVar = len(x0)

    # create an array of symbolic variables
    var = sp.symbols(f'x0:{nVar}')

    # create a symbolic function
    func = fun(var)

    # finding the gradient of the func
    der = derv(func, var)

    # create a symbol 'A' and compute x1
    A = sp.Symbol('A')
    derX0 = dervAtPoint(der,var,x0)
    x1 = np.transpose(x0) - A*np.transpose(derX0)
    valX1 = createValueArray(var, x1)

    #derivative of objective function w.r.t A
    dx1dA = sp.diff(func.subs(valX1),A)
    valA = np.array(sp.solve(dx1dA))
    
    idx = np.where(np.logical_and(np.isreal(valA), valA>0))[0]

    if idx.size > 0:
        return [x1[i].subs({'A': valA[idx][0]}) for i in range(nVar)]
    else:
        raise ValueError('No real positive value for A exists !')

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

def BFGS(fun, x0):
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

    # create a symbol 'A'
    A = sp.Symbol('A')

    #find the gradients at x0 and x1
    dx0 = dervAtPoint(der,var,x0)
    dx1 = dervAtPoint(der,var,x1)

    B0 = np.array([[1, 0], [0, 1]])
    zero = np.zeros(nVar, dtype=int)

    while not np.array_equal(dx1,zero):

        # find g and d vectors
        g = np.transpose(dx1) - np.transpose(dx0)
        d = np.transpose(x1) - np.transpose(x0)

        # update B matrix
        B1 = matrixUpdate(B0,g,d)

        # compute X1
        xnew = np.transpose(x1) - A*B1@np.transpose(dx1)
        valX1 = createValueArray(var, xnew)

        #derivative of objective function w.r.t A
        dx1dA = sp.diff(func.subs(valX1),A)
        valA = np.array(sp.solve(dx1dA))
        
        idx = np.where(np.logical_and(np.isreal(valA), valA>0))[0]

        if idx.size > 0:
            x0 = x1
            x1 = [xnew[i].subs({'A': valA[idx][0]}) for i in range(nVar)]
            dx0 = dx1
            dx1 = dervAtPoint(der,var,x1)
            B0 = B1
        else:
            raise ValueError('No real positive value for A exists !')
    
    return np.array(x1, dtype=float)