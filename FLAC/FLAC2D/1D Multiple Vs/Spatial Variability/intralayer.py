import numpy as np
import scipy

def Gaussian_intralayer(x_data, y_data, rH, a_HV):

    # Calculation of Correlation coefficient by Exponential function
    tau_x = np.abs(x_data[:, np.newaxis] - x_data[np.newaxis, :])
    tau_y = np.abs(y_data[:, np.newaxis] - y_data[np.newaxis, :])
    rho = np.exp(-2*(tau_x/rH + tau_y/(rH/a_HV)))

    rho.reshape(len(x_data), len(y_data))

    # Compute the Cholesky decomposition of the correlation matrix
    L = np.linalg.cholesky(rho)

    # Generate the random field
    G = L @ np.random.normal(0, 1, len(x_data))

    return G


def exponential_spatial(x_data, y_data, rH, a_HV):
    # Calculation of Correlation coefficient by Exponential function
    tau_x = np.abs(x_data[:, np.newaxis] - x_data[np.newaxis, :])
    tau_y = np.abs(y_data[:, np.newaxis] - y_data[np.newaxis, :])
    rho = np.exp(-2*(tau_x/rH + tau_y/(rH/a_HV)))

    # Reshape the correlation matrix
    rho = rho.reshape(len(x_data), len(y_data))

    return rho


def yang_intraleyer(x_data, y_data, cholesky_x_data, cholesky_y_data, rH, a_HV):

    # Meshgrid
    X, Y = np.meshgrid(x_data, y_data)
    X_chol, Y_chol = np.meshgrid(cholesky_x_data, cholesky_y_data)
    X_chol = X_chol.flatten()
    Y_chol = Y_chol.flatten()

    # Filter out the Cholesky nodes
    mask = np.ones(X.shape, dtype=bool)
    for xc, yc in zip(X_chol, Y_chol):
        mask &= ~((X == xc) & (Y == yc))

    X_krig = X[mask]
    Y_krig = Y[mask]

    # Transform the coordinates based on a_HV inverse
    transform_eq_6 = lambda x: x/a_HV
    transform_eq_6 = np.vectorize(transform_eq_6)

    # Calculate the spatial correlation for cholesky points
    G_chol = Gaussian_intralayer(transform_eq_6(X_chol), Y_chol, rH, a_HV)
    C_chol = exponential_spatial(transform_eq_6(X_chol), Y_chol, rH, a_HV)

    # Calculate the spatial correlation for cholesky and kriging points
    C_chol_krig = np.zeros((len(X_chol), len(X_filtered)))
    for i in range(len(X_chol)):
        for j in range(len(X_filtered)):
            tau_x = np.abs(X_chol[i] - X_filtered[j])
            tau_y = np.abs(Y_chol[i] - Y_filtered[j])
            C_chol_krig[i, j] = np.exp(-2*(tau_x/rH + tau_y/rV))

    # Compute the matrix A
    A_mat = np.hstack((rho_chol, np.ones((len(C_chol), 1))))
    A_mat = np.vstack((A_mat, np.append(np.ones((1, len(C_chol))), 0)))

    # Compute the right-hand side
    b = np.vstack((C_chol_krig, np.ones((1, C_chol_krig.shape[1]))))

    # Solve the linear system
    W = np.linalg.solve(A_mat, b)[:-1,:]

    # Compute the kriging weights
    G_krig = W.T @ G_chol

    return G_krig, G_chol

    