import numpy as np
import sys
import jax 
from jaxlie import SO3
from so3dm.metrics import c2st


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
        X = jax.vmap(lambda m: SO3.from_matrix(m).wxyz  )(X) # print(X.shape)
    
    if Y.shape[1] == 3:
        Y = jax.vmap(lambda m: SO3.from_matrix(m).wxyz  )(Y)


    c2_score = c2st(X,Y,seed,n_folds)
    print(c2_score)


if __name__ == '__main__':
    main()
