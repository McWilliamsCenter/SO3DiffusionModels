from __future__ import print_function, division
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle
import time
import sys
import jax 
from jaxlie import SO3

def c2st(X,Y,seed,n_folds):
    """Binary classifier with 2 hidden layers of 10x dim each, 
    following the architecture of Benchmarking Simulation-Based Inference 
    https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/metrics/c2st.py
    Parameters
        ----------
        X: First sample.
        Y: Second sample.
        seed: Seed for sklearn.
        n_folds: Number of folds. 
    Returns
    ----------
        Score
    """

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std
    Y = (Y - X_mean) / X_std


    ndim = X.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=10000,
        solver="adam",
        random_state=seed,
    )

    data = np.concatenate((X, Y))
    target = np.concatenate(
        (
            np.zeros((X.shape[0],)),
            np.ones((Y.shape[0],)),
        )
    )

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring="accuracy")

    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return scores

def main():
    
    X_loc = str(sys.argv[1])
    Y_loc = str(sys.argv[2])
    n_folds = int( sys.argv[3])

    with open(X_loc , 'rb') as file:
        X = np.load(file)

    with open(Y_loc , 'rb') as file:
        Y = np.load(file)
    seed = 1
    if X.shape[1] == 3:
        X = jax.vmap(lambda m: SO3.from_matrix(m).wxyz )(X) # print(X.shape)
    
    if Y.shape[1] == 3:
        Y = jax.vmap(lambda m: SO3.from_matrix(m).wxyz )(Y)

    print(X.shape)

    c2_score = c2st(X,Y,seed,n_folds)
    print(c2_score)


if __name__ == '__main__':
    main()
