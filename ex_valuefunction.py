
"""
Implement the Bellman operator for Value function iteration

"""

import numpy as np
import matplotlib.pyplot as plt

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

# A python code that maximizes a function in a given interval. Since, fminbound routine of scipy solves minimization 
# problem, we specify the objective function as negative of the original objective function

from scipy.optimize import fminbound

def maximiser(F, a, b):
    return float(fminbound(lambda x: -F(x), a, b))
def maximum(F, a, b):
    return float(F(fminbound(lambda x: -F(x), a, b)))

# Grid on the state space (k)
gridmin, gridmax, gridsize = 0.1, 5, 300
grid = np.linspace(gridmin, gridmax**1e-1, gridsize)**10
fig, ax = plt.subplots()
ax.set_xticks(grid, minor=True)
ax.set_yticks(grid, minor=True)
ax.grid(which='both', alpha=0.5)
#plt.show()


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

# The following two functions find the optimal policy and 
# value functions using the Bellman operator

# Bellman Operator
def bellman(w):
    """
    The argument of the function is a LinInterp object
    and return another LinInterp object
    """
    vals = []
    for k in grid:
        v = lambda kp: u(f(k) + (1-delta) * k - kp, theta) + beta * w(kp)
        vals.append(maximum(v, 0, f(k))) #Should upper bound be f(k)+(1-delta)*k ?
    return LinInterp(grid, vals)

# Optimal policy    
def policy(w):
    """
    The argument of the function is a LinInterp object
    and return another LinInterp object
    """
    vals = []
    for k in grid:
        v = lambda kp: u(f(k) + (1-delta)*k - kp, theta) + beta*w(kp)
        vals.append(maximiser(v, 0, f(k)))
    return LinInterp(grid, vals)

# Now we define the functions of analytical solutions

def Va(k, alpha=0.3, beta=0.9, delta=1):
    return (np.log(1-alpha*beta))/(1-beta) + \
        (alpha*beta*np.log(alpha*beta))/((1-alpha*beta)*(1-beta)) \
        + (np.log(k**alpha ))/(1-alpha*beta)

def opk(k, alpha=0.3, beta=0.9, delta=1):
    return beta*alpha * (k**alpha )

def opc(k, alpha=0.3, beta=0.9, delta=1):
    return (1-alpha*beta) * (k**alpha )

# Initial guess of V0 = u(c)

V0=LinInterp(grid,u(grid))
fig, ax = plt.subplots()
ax.plot(grid,V0(grid), label='V0')
ax.plot(grid,Va(grid), label='Actual')
ax.legend(loc=8)
#plt.show()

# Actual
fig, ax = plt.subplots()
ax.set_xlim(grid.min(), grid.max())
ax.plot(grid,Va(grid), label='Actual', color='k', lw=2, alpha=0.6)

# After iteration
count=0
maxiteration=200
tol=1e-6

while count<maxiteration:
    V1 = bellman(V0)
    err = np.max(np.abs(np.array(V1(grid)) - np.array(V0(grid))))
    if np.mod(count,10) == 0:
        ax.plot(grid,V1(grid), color=plt.cm.jet(count / maxiteration), \
            lw=2, alpha=0.6)
    V0=V1
    count+=1
    if err<tol:
        print(count)
        break
ax.plot(grid,V1(grid), label='Estimated', color='r', lw=2, alpha=0.6)
ax.legend(loc='lower right')
plt.show()

# Find whether truly converged
err=Va(grid)-V1(grid)
plt.plot(grid,err)
print("The range of error is", (err.max()-err.min()))

# The final plot of the value function
fig, ax = plt.subplots()
ax.set_ylim(-10, -7)
ax.set_xlim(grid.min(), grid.max())
ax.plot(grid,Va(grid),label='Actual')
ax.plot(grid,V1(grid)+err.mean(),label='Estimated')
ax.legend(loc='lower right')
plt.show()


# Once we have converged value function, we find the policy function
# Optimal policy
h = policy(V1)
fig, ax = plt.subplots()
ax.plot(grid,opk(grid),label='Actual')
ax.plot(grid, h(grid), label='Estimated', color='r', \
    lw=2, alpha=0.6)
ax.legend(loc='lower right')
plt.show()
