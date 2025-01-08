#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>

#define N 100  // Rozmiar macierzy

// Funkcja mnożąca dwie macierze przy użyciu NEON
void multiply_matrices_neon(int *A, int *B, int *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int32x4_t sum = vdupq_n_s32(0);  // Inicjalizacja sumy na 0
            for (int k = 0; k < N; k += 4) {  // Przetwarzanie 4 elementów jednocześnie
                // Ładowanie 4 elementów z macierzy A
                int32x4_t a_vec = vld1q_s32(&A[i * N + k]);
                // Ładowanie 4 elementów z macierzy B
                int32x4_t b_vec = vld1q_s32(&B[k * N + j]);
                // Mnożenie i dodawanie do sumy
                sum = vmlaq_s32(sum, a_vec, b_vec);
            }
            // Sumowanie wyników z wektora i zapisanie do macierzy C
            C[i * N + j] = vgetq_lane_s32(sum, 0) + vgetq_lane_s32(sum, 1) +
                            vgetq_lane_s32(sum, 2) + vgetq_lane_s32(sum, 3);
        }
    }
}

int main() {
    // Inicjalizacja macierzy A, B i C
    int *A = (int *)malloc(N * N * sizeof(int));
    int *B = (int *)malloc(N * N * sizeof(int));
    int *C = (int *)malloc(N * N * sizeof(int));

    // Inicjalizacja macierzy A i B losowymi wartościami
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 100 + 1;
        B[i] = rand() % 100 + 1;
    }

    // Pomiar czasu wykonania mnożenia
    clock_t start = clock();
    multiply_matrices_neon(A, B, C);
    clock_t end = clock();

    printf("Czas wykonania mnożenia (z NEON): %.6f sekundy\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Zwolnienie pamięci
    free(A);
    free(B);
    free(C);

    return 0;
}
