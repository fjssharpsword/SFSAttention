#https://github.com/j2kun/svd/blob/main/svd.py
#https://jeremykun.com/2016/05/16/singular-value-decomposition-part-2-theorem-proof-algorithm/
#https://en.wikipedia.org/wiki/Power_iteration
#https://www.ucg.ac.me/skladiste/blog_10701/objava_23569/fajlovi/power.pdf
import numpy as np
from numpy.linalg import norm

from random import normalvariate
from math import sqrt
import torch

def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]


def svd_1d(A, epsilon=1e-10):
    ''' The one-dimensional SVD '''

    n, m = A.shape
    x = randomUnitVector(min(n,m))
    lastV = None
    currentV = x

    if n > m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / norm(currentV)

        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            print("converged in {} iterations!".format(iterations))
            return currentV


def svd(A, k=None, epsilon=1e-10):
    '''
        Compute the singular value decomposition of a matrix A
        using the power method. A is the input matrix, and k
        is the number of singular values you wish to compute.
        If k is None, this computes the full-rank decomposition.
    '''
    A = np.array(A, dtype=float)
    n, m = A.shape
    svdSoFar = []
    if k is None:
        k = min(n, m)

    for i in range(k):
        matrixFor1D = A.copy()

        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D -= singularValue * np.outer(u, v)

        if n > m:
            v = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
            u_unnormalized = np.dot(A, v)
            sigma = norm(u_unnormalized)  # next singular value
            u = u_unnormalized / sigma
        else:
            u = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
            v_unnormalized = np.dot(A.T, u)
            sigma = norm(v_unnormalized)  # next singular value
            v = v_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
    return singularValues, us.T, vs

def power_iteration(W, eps=1e-10):
        """
        power iteration for max_singular_value
        """
        v = torch.FloatTensor(W.size(1), 1).normal_(0, 1)
        W_s = torch.matmul(W.T, W)
        while True:
            v_t = v
            v = torch.matmul(W_s, v_t)
            v = v/torch.norm(v)
            if abs(torch.dot(v.squeeze(), v_t.squeeze())) > 1 - eps: #converged
                break

        u = torch.matmul(W, v)
        s = torch.norm(u)
        u = u/s
        #return left vector, sigma, right vector
        return u, s, v

def power_iteration_square(A, num_simulations):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[0])
    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)
        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)
        # re normalize the vector
        b_k = b_k1 / b_k1_norm
    return b_k

if __name__ == "__main__":
    """
    movieRatings = np.array([
        [2, 5, 3],
        [1, 2, 1],
        [4, 1, 1],
        [3, 5, 2],
        [5, 3, 1],
        [4, 5, 5],
        [2, 4, 2],
        [2, 2, 5],
    ], dtype='float64')

    v = svd_1d(movieRatings)
    u = np.dot(movieRatings, v)
    s = norm(u)  # next singular value
    u = u / s
    print(s)
    u, s, v = power_iteration(torch.FloatTensor(movieRatings))
    print(s)
    u, s, v = torch.svd(torch.FloatTensor(movieRatings))
    print(s.max())

    x = torch.rand(192,32)
    u, s, v = power_iteration(x)
    print(s)
    u, s, v = torch.svd(x)
    print(s.max())
    """
    A = torch.rand(100,100)
    #b_k = power_iteration_square(A.numpy(), num_simulations=1000)
    u, s, v = power_iteration(A)
    print(s)
    loss_func = torch.nn.MSELoss()
    loss = loss_func(torch.flatten(A), torch.flatten(s*u*v.T))
    print(loss)
    e, v = torch.eig(A, eigenvectors=True)
    print(e.max())
