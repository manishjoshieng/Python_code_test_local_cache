#include <iostream>
#include <vector>
#include <chrono>
#include <random>

// Matrix multiplication with temporary variable
std::vector<std::vector<double>> matrixMultiplicationWithTemp(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    int n = A.size();
    std::vector<std::vector<double>> C(n, std::vector<double>(n, 0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double temp = 0;
            for (int k = 0; k < n; ++k) {
                temp += A[i][k] * B[k][j];
            }
            C[i][j] = temp;
        }
    }

    return C;
}

// Matrix multiplication without temporary variable
std::vector<std::vector<double>> matrixMultiplicationWithoutTemp(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    int n = A.size();
    std::vector<std::vector<double>> C(n, std::vector<double>(n, 0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

int main() {
    // Example matrices
    int n = 512;
    std::vector<std::vector<double>> A(n, std::vector<double>(n));
    std::vector<std::vector<double>> B(n, std::vector<double>(n));

    // Initialize matrices with random values
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = distribution(generator);
            B[i][j] = distribution(generator);
        }
    }

    // Measure execution time for matrix multiplication with temporary variable
    auto start_time_with_temp = std::chrono::high_resolution_clock::now();
    auto result_with_temp = matrixMultiplicationWithTemp(A, B);
    auto end_time_with_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> execution_time_with_temp = end_time_with_temp - start_time_with_temp;

    // Measure execution time for matrix multiplication without temporary variable
    auto start_time_without_temp = std::chrono::high_resolution_clock::now();
    auto result_without_temp = matrixMultiplicationWithoutTemp(A, B);
    auto end_time_without_temp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> execution_time_without_temp = end_time_without_temp - start_time_without_temp;

    // Check if results match
    bool results_match = true;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (result_with_temp[i][j] != result_without_temp[i][j]) {
                results_match = false;
                break;
            }
        }
    }

    if (results_match) {
        std::cout << "Results match\n";
    } else {
        std::cout << "Results do not match\n";
    }

    std::cout << "Execution time with temporary variable: " << execution_time_with_temp.count() << " seconds\n";
    std::cout << "Execution time without temporary variable: " << execution_time_without_temp.count() << " seconds\n";

    return 0;
}
