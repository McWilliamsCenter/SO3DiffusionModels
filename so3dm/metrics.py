from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier

import numpy as np
import jax 
from jaxlie import SO3

def c2st(X,Y,seed,n_folds, down_sample = True, down_sample_len = 5_000 ):
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

    if X.shape[0] > down_sample_len:
        rand_idx = np.random.randint(0, high=X.shape[0], size =down_sample_len)
        X = X[rand_idx]
        Y = Y[rand_idx]
    
    X = jax.vmap(lambda m: SO3(m).log()  )(X) # print(X.shape)
    Y = jax.vmap(lambda m: SO3(m).log()  )(Y)
 
    ndim = X.shape[1]
 
    
    clf = MLPClassifier(
    activation="relu",
    hidden_layer_sizes=(10 * ndim, 10 * ndim),
    max_iter=500,
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
    print(scores)
    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return scores
