import numpy as np

def matrix_factorization_sgd_step(U: list, V: list, r: float, lr: float, reg: float) -> list:
    """
    Returns the updated user and item vectors in a two-item list.
    """
    U = np.asarray(U, dtype = float)
    V = np.asarray(V, dtype = float)
    e = r - np.dot(U, V)
    U_new = U + lr * (e * V - reg * U)
    V_new = V + lr * (e * U - reg * V)
    return [[round(float(x), 4) for x in U_new], [round(float(x), 4) for x in V_new]]