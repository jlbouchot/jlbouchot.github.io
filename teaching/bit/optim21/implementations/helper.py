import numpy as np
import matplotlib.pyplot as plt

def mart_algo(Amat, x0, bvec, kmax = 1440): 
    # Check if the entries are non negative. If not, we should "lift" them somehow
    unhappy = True 
    k = 0
    nbRow = Amat.shape[0]
    nbUnknown = len(x0)
    
    # Compute the max along rows
    m = Amat.max(axis = 1)
    x = np.array(x0, dtype=np.float)
    i = -1
    while unhappy: 
        i = (i+1) % nbRow 
        Axi = np.matmul(Amat[i,:],x)
        for acoord in range(nbUnknown): 
            x[acoord] = x[acoord]*(bvec[i]/Axi)**(Amat[i,acoord]/m[i])
            
        k = k+1
        if k > kmax: 
            unhappy = False
            
    return x

def gradient_descent_linLS(matA, rhsB, x0 = None, gamma = None, nbIter = 2000, trueVal = None, verbose = 0): 
    '''Gradient descent algorithm with fixed step size for solving linear least squares'''
    if not gamma: 
        gamma = 1/np.max(np.linalg.eig(matA.transpose().dot(matA))[0])
        print("Step size not specified. Using {}".format(gamma))
    if x0 is None: 
        x0 = matA.transpose().dot(rhsB)
        print("x0 not specified. Starting at {}".format(x0))
    Atb = A.transpose().dot(rhsB)
    AtA = matA.transpose().dot(A)
    
    grad = lambda x: AtA.dot(x) - Atb
    
    for oneiter in range(nbIter): 
        x0 = x0 - gamma*grad(x0)
        curGradNorm = np.linalg.norm(grad(x0))
        if verbose and (oneiter % verbose) == 0: 
            print("Iter: {}. Iterate: {}. Norm gradient: {}".format(oneiter,x0, curGradNorm))
        if curGradNorm < 1e-5: 
            print("\t\t***Problem solved after {} iterations: {}. Norm of the gradient is {}\n\n\n".format(oneiter, x0, curGradNorm))
            break
    return x0
    
    
def gradient_descent_linLS_exactLineSearch(matA, rhsB, x0 = None, nbIter = 2000, trueVal = None, verbose = 0): 
    '''Gradient descent algorithm with exact line search for solving linear least squares'''
    if x0 is None: 
        x0 = matA.transpose().dot(rhsB)
        print("x0 not specified. Starting at {}".format(x0))
    Atb = A.transpose().dot(rhsB)
    AtA = matA.transpose().dot(A)
    
    grad = lambda x: AtA.dot(x) - Atb
    grad0 = grad(x0)
    curGradNorm = np.linalg.norm(grad0)
    
    for oneiter in range(nbIter): 
        gamma = curGradNorm**2/(np.linalg.norm(matA.dot(grad0))**2)
        x0 = x0 - gamma*grad(x0)
        grad0 = grad(x0)
        curGradNorm = np.linalg.norm(grad0)
        if verbose and (oneiter % verbose) == 0: 
            print("Iter: {}. Iterate: {}. Norm gradient: {}".format(oneiter,x0, curGradNorm))
        if curGradNorm < 1e-5: 
            print("\t\t***Problem solved after {} iterations: {}. Norm of the gradient is {}\n\n\n".format(oneiter, x0, curGradNorm))
            break
    return x0


def gradient_descent_linLS(matA, rhsB, x0 = None, gamma = None, nbIter = 2000, trueVal = None, verbose = 0): 
    '''Gradient descent algorithm with fixed step size for solving linear least squares'''
    if not gamma: 
        gamma = 1/np.max(np.linalg.eig(matA.transpose().dot(matA))[0])
        print("Step size not specified. Using {}".format(gamma))
    if x0 is None: 
        x0 = matA.transpose().dot(rhsB)
        print("x0 not specified. Starting at {}".format(x0))
    Atb = A.transpose().dot(rhsB)
    AtA = matA.transpose().dot(A)
    
    grad = lambda x: AtA.dot(x) - Atb
    
    for oneiter in range(nbIter): 
        x0 = x0 - gamma*grad(x0)
        curGradNorm = np.linalg.norm(grad(x0))
        if verbose and (oneiter % verbose) == 0: 
            print("Iter: {}. Iterate: {}. Norm gradient: {}".format(oneiter,x0, curGradNorm))
        if curGradNorm < 1e-5: 
            print("\t\t***Problem solved after {} iterations: {}. Norm of the gradient is {}\n\n\n".format(oneiter, x0, curGradNorm))
            break
    return x0

def gradient_descent(f, grad_f, x0, gamma, maxNbIter = 2000, eps_grad = 1e-5, verbose = 0):
    
    nbIter = 0
    gradfxkp1 = grad_f(x0)
    normgk = np.linalg.norm(gradfxkp1)
    all_xs = [x0]
    
    while(normgk >= eps_grad and nbIter <= maxNbIter): 
        xkp1 = x0 - gamma*gradfxkp1
        gradfxkp1 = grad_f(xkp1)
        normgk = np.linalg.norm(gradfxkp1)
        
        x0 = xkp1
        all_xs = np.vstack((all_xs, x0))
        
        nbIter = nbIter + 1
        
        if verbose and (nbIter % verbose) == 0: 
            print("Iter: {}. Iterate: {}. Norm gradient: {}".format(nbIter,x0, normgk))
        
    
    print("Exiting algo after {} iterations. Final result is {} and norm of gradient is {}".format(nbIter, x0, normgk))
    return x0, all_xs


def gradient_descent_quad_exact_lineSearch(f, grad_f, x0, Q, maxNbIter = 2000, eps_grad = 1e-5, verbose = 0):
    '''Solves a quadratic problem with exact line search. 
    Note: 
    1. This is really poorly implemented
    2. This is a very academic / engineered test. Hardly useable in practice (which explains why 1. hasn't been addressed)'''
    
    def exact_gamma(gradf, normfgf): 
        return normfgf**2/(gradf.transpose().dot(Q.dot(gradf)))/2
    
    nbIter = 0
    gradfxkp1 = grad_f(x0)
    normgk = np.linalg.norm(gradfxkp1)
    all_xs = x0
    
    
    while(normgk >= eps_grad and nbIter <= maxNbIter): 
        
        gamma = exact_gamma(gradfxkp1, normgk)
        
        xkp1 = x0 - gamma*gradfxkp1
        gradfxkp1 = grad_f(xkp1)
        normgk = np.linalg.norm(gradfxkp1)
        
        nbIter = nbIter + 1
        x0 = xkp1
        all_xs = np.vstack((all_xs,x0))
        if verbose and (nbIter % verbose) == 0: 
            print("Iter: {}. Iterate: {}. Norm gradient: {}".format(nbIter,x0, normgk))
        
    
    print("Exiting algo after {} iterations. Final result is {} and norm of gradient is {}".format(nbIter, x0, normgk))
    return x0, all_xs


# Some plotting functions
def plot_descent(f,x1range, x2range, x_iterates): 
    '''This function takes a function f defined on a two dimensional domain and ranges for the first and second coordinate and the sequence of iterates. 
    It plots the contours of f and the various iterates.'''
    ax = plt.figure()
    xaxis = np.arange(x1range[0], x1range[1], 0.05)
    yaxis = np.arange(x2range[0], x2range[1], 0.05)
    x, y = np.meshgrid(xaxis, yaxis)

    results = [[f(np.array([x[i][j], y[i][j]])) for j in range(len(xaxis))] for i in range(len(yaxis))]
    plt.contourf(x, y, results, levels=50, cmap='jet')
    
    plt.plot(x_iterates[:, 0], x_iterates[:, 1], '.-', color='w')
    
    return ax


def accelerated_gradient_descent(f, grad_f, x0, gamma, maxNbIter = 2000, eps_grad = 1e-5, verbose = 0):
    
    # Initialize all the x/y/z to the same thing
    y0 = x0
    z0 = x0
    
    nbIter = 0
    gradfykp1 = grad_f(y0)
    normgk = np.linalg.norm(gradfykp1)
    all_xs = [x0]
    
    while(normgk >= eps_grad and nbIter <= maxNbIter): 
        xkp1 = y0 - gamma*gradfykp1
        zkp1 = z0 - gamma/2*(nbIter+1)*gradfykp1
        ykp1 = ((nbIter+1)*xkp1 + 2*zkp1)/(nbIter+3)
        gradfykp1 = grad_f(ykp1)
        normgk = np.linalg.norm(gradfykp1)
        
        # Update the variables 
        x0 = xkp1
        y0 = ykp1
        z0 = zkp1
        # Save the x
        all_xs = np.vstack((all_xs, x0))
        
        nbIter = nbIter + 1
        
        if verbose and (nbIter % verbose) == 0: 
            print("Iter: {}. Iterate: {}. Norm gradient: {}".format(nbIter,x0, normgk))
        
    
    print("Exiting AGD after {} iterations. Final result is {} and norm of gradient is {}".format(nbIter, x0, normgk))
    return x0, all_xs
    
def projected_gradient_descent_dummy(f, grad_f, projection, x0, gamma, maxNbIter = 2000, eps_grad = 1e-5, verbose = 0):
    
    nbIter = 0
    gradfxkp1 = grad_f(x0)
    normgk = np.linalg.norm(gradfxkp1)
    all_xs = [x0]
    
    while(normgk >= eps_grad and nbIter <= maxNbIter): 
        xkp1 = projection(x0 - gamma*gradfxkp1)
        gradfxkp1 = grad_f(xkp1)
        normgk = np.linalg.norm(gradfxkp1)
        
        x0 = xkp1
        all_xs = np.vstack((all_xs, x0))
        
        nbIter = nbIter + 1
        
        if verbose and (nbIter % verbose) == 0: 
            print("Iter: {}. Iterate: {}. Norm gradient: {}".format(nbIter,x0, normgk))
        
    
    print("Exiting algo after {} iterations. Final result is {} and norm of delta x is {}".format(nbIter, x0, normgk))
    return x0, all_xs


def accelerated_projected_gradient_descent_dummy(f, grad_f, projection, x0, gamma, maxNbIter = 2000, eps_grad = 1e-5, verbose = 0):
    
    # Initialize all the x/y/z to the same thing
    y0 = x0
    z0 = x0
    
    nbIter = 0
    gradfykp1 = grad_f(y0)
    normgk = np.linalg.norm(gradfykp1)
    all_xs = [x0]
    
    while(normgk >= eps_grad and nbIter <= maxNbIter): 
        xkp1 = projection(y0 - gamma*gradfykp1)
        zkp1 = projection(z0 - gamma/2*(nbIter+1)*gradfykp1)
        ykp1 = ((nbIter+1)*xkp1 + 2*zkp1)/(nbIter+3)
        gradfykp1 = grad_f(ykp1)
        normgk = np.linalg.norm(gradfykp1)
        
        # Update the variables 
        x0 = xkp1
        y0 = ykp1
        z0 = zkp1
        # Save the x
        all_xs = np.vstack((all_xs, x0))
        
        nbIter = nbIter + 1
        
        if verbose and (nbIter % verbose) == 0: 
            print("Iter: {}. Iterate: {}. Norm gradient: {}".format(nbIter,x0, normgk))
        
    
    print("Exiting AGD after {} iterations. Final result is {} and norm of delta x is {}".format(nbIter, x0, normgk))
    return x0, all_xs
    
def projected_gradient_descent(f, grad_f, projection, x0, gamma, maxNbIter = 2000, eps_grad = 1e-5, verbose = 0):
    
    nbIter = 0
    gradfxkp1 = grad_f(x0)
    normdeltax = 10
    all_xs = [x0]
    
    while(normdeltax >= eps_grad and nbIter <= maxNbIter): 
        xkp1 = projection(x0 - gamma*gradfxkp1)
        gradfxkp1 = grad_f(xkp1)
        
        normdeltax = np.linalg.norm(xkp1-x0)
        x0 = xkp1
        all_xs = np.vstack((all_xs, x0))
        
        nbIter = nbIter + 1
        
        if verbose and (nbIter % verbose) == 0: 
            print("Iter: {}. Iterate: {}. Norm gradient: {}".format(nbIter,x0, normdeltax))
        
    
    print("Exiting algo after {} iterations. Final result is {} and norm of delta x is {}".format(nbIter, x0, normdeltax))
    return x0, all_xs


def accelerated_projected_gradient_descent(f, grad_f, projection, x0, gamma, maxNbIter = 2000, eps_grad = 1e-5, verbose = 0):
    
    # Initialize all the x/y/z to the same thing
    y0 = x0
    z0 = x0
    
    nbIter = 0
    gradfykp1 = grad_f(y0)
    normdeltax = 10
    all_xs = [x0]
    
    while(normdeltax >= eps_grad and nbIter <= maxNbIter): 
        xkp1 = projection(y0 - gamma*gradfykp1)
        zkp1 = projection(z0 - gamma/2*(nbIter+1)*gradfykp1)
        ykp1 = ((nbIter+1)*xkp1 + 2*zkp1)/(nbIter+3)
        gradfykp1 = grad_f(ykp1)
        normdeltax = np.linalg.norm(xkp1-x0)
        
        # Update the variables 
        x0 = xkp1
        y0 = ykp1
        z0 = zkp1
        # Save the x
        all_xs = np.vstack((all_xs, x0))
        
        nbIter = nbIter + 1
        
        if verbose and (nbIter % verbose) == 0: 
            print("Iter: {}. Iterate: {}. Norm gradient: {}".format(nbIter,x0, normdeltax))
        
    
    print("Exiting AGD after {} iterations. Final result is {} and norm of delta x is {}".format(nbIter, x0, normdeltax))
    return x0, all_xs

def proj_l2(x, r): 
    return 0 if np.linalg.norm(x) <= 1e-16 else x/np.linalg.norm(x)*r

def proj_nonneg(x):
    return np.array([np.max([aval,0]) for aval in x])
    
def proj_l1_nonneg(x, r): 
    '''We assume a projection onto the positive simplex with with radius r. No clue what will happen if this isn't the case'''
    if x.sum() - r <= 0: 
        return x 
        
    sorted_x = np.sort(x)[::-1] # Sorting in descending order. 
    # Number of nonzeros:
    cum_sorted = np.cumsum(sorted_x)
    nnz = np.nonzero(x*range(1,len(x)+1) > cum_sorted - r)[-1] # Returns the last element which is True
    
    # Compute the threshold in our soft thresholding
    theta_nnz = (cum_sorted[nnz]-r)/(nnz +1)
    return np.clip(x-theta_nnz, 0, None)


def proj_l1_general(x,r): 
    sign_of_x = np.sign(x)
    return sign_of_x*proj_l1_nonneg(sign_of_x*x, r)

# Below are old test things which, in fact, do not belong here. 

'''
# Testing the mart algorithm
B = np.array([[1,3,3,2],[1,2,3,3],[1,3,2,3],[1,1,1,1]])
b = np.array([2,2,2,1])

c = np.array([40,20,10,40])

print(mart_algo(B,c,b))


A = np.array([[10,1],[1,10]], dtype = np.double)
print(A)
xstar = np.array([1,1], dtype = np.double)
bvec = A.dot(xstar)
print(bvec)

gradient_descent_linLS(A,bvec, gamma = 0.01, x0 = np.array([0,0]), verbose = 5)
gradient_descent_linLS(A,bvec, verbose = 5)


A = np.array([[1000,1],[1,10]], dtype = np.double)
xstar = np.array([1,1], dtype = np.double)
bvec = A.dot(xstar)
gradient_descent_linLS(A,bvec, nbIter = 100000, verbose = 1000)
# gradient_descent_linLS(A,bvec, gamma = 0.000001, x0 = np.array([0,0]), nbIter = 100000, verbose = 1000)
# gradient_descent_linLS(A,bvec, gamma = 0.000005, x0 = np.array([0,0]), nbIter = 100000, verbose = 1000)


# With exact line search 
A = np.array([[10,1],[1,10]], dtype = np.double)
xstar = np.array([1,1], dtype = np.double)
bvec = A.dot(xstar)
gradient_descent_linLS_exactLineSearch(A, bvec, verbose = 5)

A = np.array([[1000,1],[1,10]], dtype = np.double)
xstar = np.array([1,1], dtype = np.double)
bvec = A.dot(xstar)
gradient_descent_linLS_exactLineSearch(A, bvec, verbose = 1000, nbIter = 10000)
'''