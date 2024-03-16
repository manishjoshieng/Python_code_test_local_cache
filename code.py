import numpy as np
import time

# Matrix multiplication with temporary variable
def matrix_multiplication_with_temp(A, B):
    n = len(A)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            temp = 0
            for k in range(n):
                temp += A[i, k] * B[k, j]
            C[i, j] = temp
    return C

# Matrix multiplication without temporary variable
def matrix_multiplication_without_temp(A, B):
    n = len(A)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

# Example matrices
n = 64
A = np.random.rand(n, n)
B = np.random.rand(n, n)

# Measure execution time for matrix multiplication with temporary variable
start_time_with_temp = time.time()
result_with_temp = matrix_multiplication_with_temp(A, B)
end_time_with_temp = time.time()
execution_time_with_temp = end_time_with_temp - start_time_with_temp

# Measure execution time for matrix multiplication without temporary variable
start_time_without_temp = time.time()
result_without_temp = matrix_multiplication_without_temp(A, B)
end_time_without_temp = time.time()
execution_time_without_temp = end_time_without_temp - start_time_without_temp

# Check if results match
assert np.allclose(result_with_temp, result_without_temp)

print("Execution time with temporary variable:", execution_time_with_temp, "seconds")
print("Execution time without temporary variable:", execution_time_without_temp, "seconds")
