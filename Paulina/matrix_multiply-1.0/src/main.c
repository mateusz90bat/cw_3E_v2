#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100  // Rozmiar macierzy 100x100

void multiply_matrices_basic(int A[N][N], int B[N][N], int C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int A[N][N], B[N][N], C[N][N];
    
    // Inicjalizacja macierzy A i B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;  // Losowe wartości
            B[i][j] = rand() % 10;
        }
    }

    // Pomiary czasu
    printf("Rozpoczynam pomiar czasu...\n");
    clock_t start = clock();
    multiply_matrices_basic(A, B, C);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Czas wykonania wersji podstawowej: %f sekundy\n", time_taken);

    return 0;
}
