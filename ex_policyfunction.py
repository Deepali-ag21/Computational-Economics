"""
The policy function iteration
or Howard's improvement algorithm is one such method.
"""

"""
Policy function iteration

"""

import numpy as np
from scipy.optimize import fminbound
import scipy.sparse as sp


# Define interpolated functions using a new class of Python object
class LinInterp:
    ''' Provides linear interpolation in one dimension. '''
    def __init__(self, X, Y):
        ''' Parameters: X and Y are sequences or arrays 
        containing the (x,y) interpolation points. '''
        
        self.X, self.Y = X, Y
    def __call__(self, z):
        ''' If z is a float (or integer) returns a float; 
        if z is a sequence or array returns an array. '''
        if isinstance(z, int) or isinstance(z, float):
            return np.interp([z], self.X, self.Y)[0]
        else:
            return np.interp(z, self.X, self.Y)

# Maximiser function
def maximiser(F, a, b):
    return float(fminbound(lambda x: -F(x), a, b))
def maximum(F, a, b):
    return float(F(fminbound(lambda x: -F(x), a, b)))

# The utility function
def u(c, theta=1):
    if theta != 1:
        u = (c**(1-theta) - 1)/(1-theta)
    else:
        u = np.log(c)
    return u
  
# The production function
def f(k, alpha=0.3, A=1):
    return A * k**alpha

# Initialise the parameter values
alpha = 0.3 # Cobb-Douglas production function parameter
beta = 0.9  # Discount parameter
delta = 1   # Capital depreciation
theta = 1   # Utility function parameter; u = ln(c)


# Now we define the functions of analytical solutions

def Va(k, alpha=0.3, beta=0.9, delta=1):
    return (np.log(1-alpha*beta))/(1-beta) + \
        (alpha*beta*np.log(alpha*beta))/((1-alpha*beta)*(1-beta)) \
        + (np.log(k**alpha ))/(1-alpha*beta)

def opk(k, alpha=0.3, beta=0.9, delta=1):
    return beta*alpha * (k**alpha )

def opc(k, alpha=0.3, beta=0.9, delta=1):
    return (1-alpha*beta) * (k**alpha )


# Policy evaluation step
def policy_evaluation(h, kgrid):
    """
    Compute the value of a policy
    The argument takes a feasible policy (LinInterp)
    and return the value function (LinInterp) associated 
    with the policy function
    """
    b = []
    # A sparse matrix is one in which most of the elements are 0. 
    # That is, the matrix only contains data in a few positions.
    Q = sp.lil_matrix((kgrid.size, kgrid.size), dtype=int)

    # Find the index of the closest grid point
    for i in range(len(kgrid)):
        kp = h(kgrid[i])
        b.append(u(f(kgrid[i]) + (1-delta)*kgrid[i] - kp, theta))
        closest_index = np.argmin(np.abs(kgrid - kp))
        Q[i, closest_index] = 1

    Q = Q.tocsr()
    I = sp.identity(kgrid.size, format='csr')
    A = I - Q.multiply(beta)
    V_h = sp.linalg.spsolve(A, b)
    return LinInterp(grid, V_h)


# Policy improvement step   
def policy_improvement(w, kgrid):
    """
    Compute the v-greedy policy.
    The argument of the function is a value function (LinInterp object)
    and return another policy function (LinInterp object)
    """
    pols = []
    for k in kgrid:
        v = lambda kp: u(f(k) + (1-delta)*k - kp, theta) + beta*w(kp)
        pols.append(maximiser(v, 0.001, f(k)+(1-delta)*k))

    policy = LinInterp(kgrid, pols)
    return policy


# Initial policy function
def initial_policy(kgrid):
    '''
    Takes a grid argument 
    and returns a LinInterp object
    '''
    h = lambda k: k*0
    return LinInterp(kgrid, h(kgrid))


# Grid on the state space (k)
gridmin, gridmax, gridsize = 0.1, 5, 300
grid = np.linspace(gridmin, gridmax**1e-1, gridsize)**10

# Policy function iteration
count=0
maxiteration=200
tol=1e-6
policy_function = initial_policy(grid)

while count<maxiteration:
    V0 = policy_evaluation(policy_function, grid)
    new_policy_function = policy_improvement(V0, grid)
    err = np.max(np.abs(np.array(new_policy_function(grid)) \
        - np.array(policy_function(grid))))
    policy_function = new_policy_function
    count+=1
    if err<tol:
        print(count)
        break

# Plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_xlim(grid.min(), grid.max())
ax.plot(grid,opk(grid),label='Actual')
ax.plot(grid, policy_function(grid), label='Estimated', color='r', \
    lw=2, alpha=0.6)
ax.legend(loc='lower right')
plt.show()

