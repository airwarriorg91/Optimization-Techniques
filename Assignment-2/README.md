# Optimization Algorithms: BFGS and Genetic Algorithm

## Overview

This project implements two optimization algorithms, **BFGS (Broyden–Fletcher–Goldfarb–Shanno)** and **Genetic Algorithm (GA)**. Each algorithm is contained within its own module and can be used by importing the respective class.

### Features:
- **BFGS**: A gradient-based optimization algorithm used to find local minima of a function. The algorithm approximates the Hessian matrix and is efficient for smooth functions.
- **Genetic Algorithm (GA)**: An evolutionary optimization technique inspired by natural selection, useful for global optimization of non-linear, non-differentiable functions.

## Requirements

- Python 3.6+
- NumPy (for matrix operations)
- SymPy (for symbolic operations)

## Usage

To use the BFGS or GA algorithm in your project, import the respective class:

```python
from BFGS import BFGS
from GA import GA
```

## Method Descriptions

### BFGS
- **Purpose**: Used to minimize smooth, differentiable functions.
- **Usage**:
  ```python
  #Returns an array of solution
  result = BFGS(objective_function, initial_guess)
  ```
- **Parameters**:
  - `objective_function`: Function to be minimized.
  - `x0`: Initial guess for the variable values.

### GA (Genetic Algorithm)
- **Purpose**: Designed to optimize non-linear and non-differentiable functions using evolutionary principles.
- **Usage**:
  ```python
  sol = GA(generations, population_size, objective_function)
  sol.show()
  ```
- **Parameters**:
  - `objective_function`: Fitness function to be optimized.
  - `population_size`: Size of the population.
  - `generations`: Number of generations to evolve.

## Examples

### BFGS Example
```python
def objective(x):
    return (x - 3)**2

initial_guess = [0]
result = BFGS(objective_function, initial_guess)
print("Minimum found:", result)
```

### Genetic Algorithm Example
```python
def fitness(x):
    return x**2

generations = 10
population_size = 20
sol = GA(generations, population_size, fitness)
sol.show()
```

## License

This project is for educational purposes.

## Results from the Code

```shell
Solution from BFGS: [-1.   1.5] and Solution from GA: [-0.99964371  1.65817322]
Solution from BFGS: [-0.  0.] and Solution from GA: [-0.06193411  0.02544361]
Solution from BFGS: [ 0. -0.] and Solution from GA: [0.01887624 0.95234389]
```
(May vary with parameters)
