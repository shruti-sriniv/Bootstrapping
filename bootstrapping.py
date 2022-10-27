import numpy as np
import probclearn
import probcpredict
# Input: number of bootstraps B
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of B rows, 1 column
def run(B,X,y):
    z = np.zeros((B, 1))
    n, d = np.shape(X)
    for i in range(0, B):
        u = np.zeros(n)
        S = set()
        for j in range(0, n):
            k = np.random.randint(0, n)
            u[j] = k
            S = S.union({k})
        T = set(range(0, n)) - S
        Xtrain = np.zeros((n, d))
        Ytrain = np.zeros((n, 1))
        counter = 0
        for val in range(0, n):
            value = int(u[val])
            Ytrain[counter] = y[value]
            Xtrain[counter] = X[value]
            counter += 1
        q, mu_positive, mu_negative, sigma2_positive, sigma2_negative = probclearn.run(Xtrain, Ytrain)
        for t in T:
            t = int(t)
            if y[t] != probcpredict.run(q, mu_positive, mu_negative, sigma2_positive, sigma2_negative, np.reshape(X[t], (d, 1))):
                z[i] += 1
        z[i] /= len(T)
    return z