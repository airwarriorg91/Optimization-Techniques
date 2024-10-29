import numpy as np
import sympy as sp
import copy

class solution:

    # Parameters
    DNA = [0]  # DNA of the solution
    fitness = 0 # How good the solution is ?
    var = [] # Variables

    # Methods
    def eval_fitness(self, func):
        fun = func(self.var) # create a symbolic function
        vals = dict(zip(self.var, self.DNA)) # dictonary of var and values
        self.fitness = fun.subs(vals).evalf()

    def __init__(self, nDim, bounds, func):
        self.DNA = np.random.uniform(bounds[0], bounds[1], nDim)
        self.var = sp.symbols(f'x0:{nDim}')
        self.eval_fitness(func)

    def copy(self):
        new_sol = solution.__new__(solution)

        # Manually copy attributes
        new_sol.DNA = copy.deepcopy(self.DNA)
        new_sol.fitness = self.fitness
        new_sol.var = self.var
        return new_sol
    
    def show(self):
        print(f"DNA: {self.DNA}")
        print(f"Fitness: {self.fitness}")


class generation:

    #Parameters
    id = 0 # id^th generation
    nsol = 0 # Number of solutions in the generation
    population = [] # Array of solution objects
    parents = [] # Array of parents in current generation
    children = [] # Array of children of current generation
    fun = 0 # Objective Function

    #Methods

    def __init__(self, gen, members, func, ndim, bounds):
        
        if(members<0):
            raise ValueError("Number of members should be positive.")
        elif(members%2!=0):
            raise ValueError("Number of members in a generation should be even")

        self.nsol = members
        self.id = gen
        self.population = [solution(ndim,bounds,func) for i in range(self.nsol)]
        self.fun = func

    def evaluate(self):
        for child in self.children:
            child.eval_fitness(self.fun)

    def elitism(self):

        '''
        Function for Elitism based selection of parents

        Input: Self

        Output: Array of solution selected from the population

        Algorithm: Constant population size with Elitism

            1. Create a sorted list of elements based on fitness (minimum is the best)
            2. The top 10% are members of next generation directly.
            3. Rest 90% are parents for next generation.
        '''

        sortedPopulation = sorted(self.population, key=lambda obj: obj.fitness)

        elite_count = int(0.1 * self.nsol)
        self.children = sortedPopulation[:elite_count]
        self.parents = sortedPopulation[elite_count:]

        pass

    def crossover(self):

        '''
        Function to perform Simulated Binary Crossover for a given array of parents.

        Input: Array of parents in current generation

        Output: Array of children of current generation

        Algorithm: 
        1. Select two parents x1 and x2
        2. Find U and Calculate Beta,
            Beta = (2U)^1/(etac + 1) if U<0.5
            Beta = (1/(2U-1))^1/(etac + 1) if U>0.5
        3. x1_new = 0.5*((1+Beta)*x1 + (1-Beta)*x2)
           x2_new = 0.5*((1-Beta)*x1 + (1+Beta)*x2)
           
           eta = 5
        '''
        np.random.shuffle(self.parents)
        for i in range(0,len(self.parents),2):
            p1 = self.parents[i]
            p2 = self.parents[i+1]
            U = np.random.rand(1)
            eta = 5
            if (U<0.5):
                Beta = (2*U)**(1/(eta+1))
            else:
                Beta = (1/(2*U-1))**(1/(eta+1))
            
            # Make a copy of parents
            c1 = p1.copy()
            c2 = p2.copy()

            #Update the DNA from crossover
            c1.DNA = 0.5*(1+Beta)*p1.DNA + 0.5*(1-Beta)*p2.DNA
            c2.DNA = 0.5*(1-Beta)*p1.DNA + 0.5*(1+Beta)*p2.DNA

            #Update the fitness value
            c1.eval_fitness(self.fun)
            c2.eval_fitness(self.fun)

            #Append to the children array
            self.children.append(c1)
            self.children.append(c2)
        
    def mutation(self, mutationRate = 0.05, sigma = 0.5):

        '''
        Function to perform mutation in children of current generation

        Input: Array of children for current generation

        Output: Array of children with mutation

        Algorithm: Normally Disturbed mutation

        if prob < MutationRate:
            x_new = x + N(0, sigma)
        '''
        for sol in self.children:
            prob = np.random.rand()
            if prob <= mutationRate:
                sol.DNA += np.random.normal(0, sigma, size=len(sol.DNA))
        
    def show(self):
        print(f"Generation Number: {self.id}")
        print(f"Population: {[i.DNA for i in self.population]}")
        print(f"Selected Parents: {[i.DNA for i in self.parents]}")
        print(f"Children: {[i.DNA for i in self.children]}")

    def reset(self):
        self.population = []
        self.parents = []
        self.children = []

def GA(nGen, nSol, func, ndim=2, bounds=[-10,10]):
    '''
    Function to implement Genetic Algorithm

    Input: 
    1. Number of Generations
    2. Number of Solutions in each generation
    3. Objective Function
    4. Dimension of solution
    5. Bound of the solution

    Output: Optimal Solution

    Algorithm:
    1. Initiate first generation
    Inside loop:
        1. Elitism() of nth gen
        2. Crossover() of nth gen
        3. Mutation() of nth gen
        4. population of n+1th gen = children of nth gen
        5. Check most optimal solution of nth generation similar to most optimal solution of n+1th gen, if yes stop the algorithm.
        6. If not continue till all generations and return the most optimal solution.
    
    '''
    eps = 1e-2
    gen0 = generation(0, nSol, func, ndim, bounds)
    
    bestSol = 0
    for i in range(1, nGen):
        gen0.elitism()
        gen0.crossover()
        gen0.mutation(mutationRate=0.075, sigma=0.2)
        gen0.evaluate()

        gen1 = generation(i, nSol, func, ndim, bounds)
        gen1.reset()
        gen1.population = copy.deepcopy(gen0.children)

        gen1.elitism()
        gen1.crossover()
        gen1.mutation(mutationRate=0.1, sigma=0.1)
        gen1.evaluate()
        
        bestgen1 = sorted(gen1.children, key=lambda obj: obj.fitness)[0]
        
        if i==1:
            bestSol = bestgen1
        elif bestgen1.fitness < bestSol.fitness:
            bestSol = bestgen1

        # print(f"Generation {i}, Best Fitness: {bestgen1.fitness}")

        gen0.reset()
        gen0.population = copy.deepcopy(gen1.children)

    return bestSol