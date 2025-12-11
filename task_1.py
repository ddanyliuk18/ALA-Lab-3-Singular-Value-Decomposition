import numpy as np

def svd_implementation(matrix):
        matrix_transpose = matrix.T
        first_symmetric_matrix = matrix_transpose @ matrix
        eigenvalues, eigenvectors = np.linalg.eig(first_symmetric_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        singular_values = np.sqrt(eigenvalues)
        sorted_indx = np.argsort(singular_values)[::-1]
        singular_values = singular_values[sorted_indx]
        V = eigenvectors[:,sorted_indx]
        sigma = np.diag(singular_values)
        U = matrix @ V
        for i in range (len(singular_values)):
                if not singular_values[i] == 0:
                        U[:, i] = U[:, i] / singular_values[i]
                else:
                        print("There is division by zero")


        matrix_reconstructed = U@sigma@V.T
        is_true = np.isclose(matrix_reconstructed, matrix)
        if np.all(is_true):
                print("True")
        else:
                print("False")

test_matrix = np.array ([[2, 3, 4], [1, 9, 4], [4, 3, 2]])
svd_implementation(test_matrix)