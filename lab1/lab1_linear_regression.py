import sys
from copy import deepcopy

def matrix_transpose(A):
    A_t = [list(element) for element in zip(*A)]
    return A_t

def matrix_multiply(A, B):
    # check dimension of A and B should match. e,q, mxn and nx1
    assert len(A[0]) == len(B)
    all_col_of_B = list(zip(*B))
    #print ("All_col_of_B:", all_col_of_B)
    result = [[sum(element_A * element_B for element_A, element_B in zip(row_in_A, col_in_B)) for col_in_B in all_col_of_B] for row_in_A in A]
    return result

def pivot_matrix(A):
    dim = len(A)
    copy_A = deepcopy(A)
    identity_matrix = [[ float(i==j) for i in range(dim)] for j in range(dim)]
    #print (identity_matrix)
    for i in range(dim):
        max_row = max(range(i, dim), key=lambda j: abs(copy_A[j][i]))
        '''
        print ("current index:", i, "max index:", max_row)
        print ("current element:", copy_A[i][i], "max element:", copy_A[max_row][i])
        print ("current row:", copy_A[i], "max row:", copy_A[max_row])
        print ("copy_A:", copy_A)
        '''
        if i != max_row:
            copy_A[i], copy_A[max_row] = copy_A[max_row], copy_A[i]
            identity_matrix[i], identity_matrix[max_row] = identity_matrix[max_row], identity_matrix[i]

    return identity_matrix

def plu_decomposition(A):
    dim = len(A)

    L = [[0.0 for col in range(dim)] for row in range(dim)]
    U = [[0.0 for col in range(dim)] for row in range(dim)]

    P = pivot_matrix(A)
    #print ("Pivot matrix:", P)
    PA = matrix_multiply(P, A)
    #print ("PA:", PA)
    for j in range(dim):
        L[j][j] = 1.0
        
        for i in range(j+1):
            s = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = PA[i][j] - s

        for i in range(j, dim):
            s = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (PA[i][j] - s)/(U[j][j])

    return P, L, U

def main():
    # Load arguments
    arguments = sys.argv
    print (arguments)
    fname = arguments[1]
    poly_bases = int(arguments[2])
    LAMBDA = float(arguments[3])
    # Load data points and labels
    A = []
    b = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            coord = line.strip().split(',')
            coord = [float(p) for p in coord]
            b.append([float(coord.pop(-1))])
            A.append([pow(float(coord[0]), poly_bases-base-1) for base in range(poly_bases)])
    #print ("Matrix A:", A)
    #print ("Matrix b:", b)
    # Compute PLU decomposition of A
    A_t = matrix_transpose(A)
    #print ("A_t:", A_t)
    A_square = matrix_multiply(A_t, A)
    #print ("A_square:", A_square)
    A_square_plus_lambda = A_square
    for index, row in enumerate(A_square_plus_lambda):
        A_square_plus_lambda[index][index] += LAMBDA
    #print ("A_square_plus_lambda:", A_square_plus_lambda)
    P, L, U = plu_decomposition(A_square_plus_lambda)
    #print ("P", P)
    #print ("L", L)
    #print ("U", U)
    
    A_t_b = matrix_multiply(A_t, b)
    # calculate Ly=b
    y = [ [0.0] for _ in range(poly_bases)]
    for i in range(poly_bases):
        other_terms = 0
        for j in range(i):
            other_terms += y[j][0] * L[i][j]
        y[i][0] = (A_t_b[i][0] - other_terms)/L[i][i]
    #print ("y:", y)

    # calcuate Ux=y
    x = [ [0.0] for _ in range(poly_bases)]
    for i in range(poly_bases-1, -1, -1):
        other_terms = 0
        for j in range(poly_bases-1, i, -1):
            other_terms += x[j][0] * U[i][j]
        x[i][0] = (y[i][0] - other_terms)/U[i][i]
    #print ("x:", x)
    print ("==============================================================================================================")
    output_string = ""
    for base, cof in zip(range(poly_bases-1, -1, -1), x):
        output_string = output_string + (" %.4f * X^%d" % (cof[0], base))
        if base != 0:
            output_string += " +"
        else:
            output_string += " = b"
    print ("Final equation: ", output_string)
    MSE = 0
    for data_point, target in zip(A, b):
        prediction = 0
        for index in range(poly_bases):
            prediction += data_point[index] * x[index][0]
        MSE += (prediction-target[0]) ** 2
    MSE /= len(A)
    print ("Mean Square Errors:", MSE)
    print ("==============================================================================================================")

if __name__ == '__main__':
    main()
