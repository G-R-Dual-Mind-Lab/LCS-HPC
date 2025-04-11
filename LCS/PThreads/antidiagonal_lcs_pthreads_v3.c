#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <limits.h>
#include <papi.h>

#define NUM_THREADS 16

// Global variables di input
int len_A, len_B;
char *str_A, *str_B;

// Per la versione ottimizzata con diagonali, la lunghezza massima di una diagonale:
int max_diag;  // = min(len_A, len_B) + 1

// I tre buffer per le diagonali (diagonale d-2, d-1 e quella corrente)
int *diag2;   // diagonal due indietro
int *diag1;   // diagonal precedente
int *diag_cur; // diagonal corrente

// Barriera per sincronizzare i thread
pthread_barrier_t barrier;

// Variabile globale per la diagonale corrente (aggiornata in ogni iterazione)
volatile int current_d;

//
// Funzione helper: restituisce il massimo di due interi
//
int max_int(int a, int b) {
    return (a > b ? a : b);
}

//
// La funzione eseguita da ciascun thread.
//
void *lcs_thread_func(void *arg) {
    int id = *(int *)arg;

    for (int d = 2; d <= len_A + len_B; d++) {
        current_d = d;

        int i_min = (d - len_B > 1) ? d - len_B : 1;
        int i_max = (d - 1 < len_A) ? d - 1 : len_A;
        int count = i_max - i_min + 1;

        int A_prev = 0, count_prev = 0;
        if (d >= 3) {
            A_prev = ((d - 1) - len_B > 1) ? (d - 1) - len_B : 1;
            int i_max_prev = ((d - 1) - 1 < len_A) ? d - 2 : len_A;
            count_prev = (i_max_prev - A_prev + 1 > 0) ? (i_max_prev - A_prev + 1) : 0;
        }

        int A_prev2 = 0, count_prev2 = 0;
        if (d >= 4) {
            A_prev2 = ((d - 2) - len_B > 1) ? (d - 2) - len_B : 1;
            int i_max_prev2 = ((d - 2) - 1 < len_A) ? d - 3 : len_A;
            count_prev2 = (i_max_prev2 - A_prev2 + 1 > 0) ? (i_max_prev2 - A_prev2 + 1) : 0;
        }

        int chunk = count / NUM_THREADS;
        int rem = count % NUM_THREADS;
        int start, end;
        if (id < rem) {
            start = id * (chunk + 1);
            end = start + (chunk + 1);
        } else {
            start = id * chunk + rem;
            end = start + chunk;
        }

        for (int k = start; k < end; k++) {
            int i = i_min + k;
            int j = d - i;

            int val = 0;
            if (str_A[i - 1] == str_B[j - 1]) {
                int diag_val = 0;
                if (d >= 4) {
                    int pos = (i - 1) - A_prev2;
                    if (pos >= 0 && pos < count_prev2)
                        diag_val = diag2[pos];
                }
                val = diag_val + 1;
            } else {
                int top = 0, left = 0;
                if (d >= 3) {
                    int pos_top = (i - 1) - A_prev;
                    if (pos_top >= 0 && pos_top < count_prev)
                        top = diag1[pos_top];
                    int pos_left = i - A_prev;
                    if (pos_left >= 0 && pos_left < count_prev)
                        left = diag1[pos_left];
                }
                val = max_int(top, left);
            }
            diag_cur[k] = val;
        }

        pthread_barrier_wait(&barrier);

        if (id == 0) {
            int *temp = diag2;
            diag2 = diag1;
            diag1 = diag_cur;
            diag_cur = temp;
            memset(diag_cur, 0, max_diag * sizeof(int));
        }

        pthread_barrier_wait(&barrier);
    }

    pthread_exit(NULL);
    return NULL;
}

//
// main
//
int main(int argc, char *argv[]) {
    if(argc < 2){
        printf("Error: No input file specified.\n");
        return 1;
    }

    FILE *fp = fopen(argv[1], "r");
    if(fp == NULL){
        perror("Error opening file");
        return 1;
    }

    if(fscanf(fp, "%d %d", &len_A, &len_B) != 2){
        printf("Error: Unable to read string lengths.\n");
        fclose(fp);
        return 1;
    }

    if(len_A <= 0 || len_B <= 0){
        printf("Error: Invalid string lengths.\n");
        fclose(fp);
        return 1;
    }

    printf("String A length: %d\nString B length: %d\n", len_A, len_B);

    str_A = malloc((len_A + 1) * sizeof(char));
    str_B = malloc((len_B + 1) * sizeof(char));
    if(str_A == NULL || str_B == NULL){
        printf("Error: Memory allocation failed for strings.\n");
        fclose(fp);
        return 1;
    }

    if(fscanf(fp, "%s %s", str_A, str_B) != 2){
        printf("Error: Unable to read strings.\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    int EventSet = PAPI_NULL;
    if(PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT){
        fprintf(stderr, "Error: PAPI initialization failed.\n");
        return 1;
    }

    max_diag = (len_A < len_B ? len_A : len_B) + 1;
    diag2 = calloc(max_diag, sizeof(int));
    diag1 = calloc(max_diag, sizeof(int));
    diag_cur = calloc(max_diag, sizeof(int));
    if(diag2 == NULL || diag1 == NULL || diag_cur == NULL){
        printf("Error: Memory allocation failed for diagonals.\n");
        return 1;
    }

    if(pthread_barrier_init(&barrier, NULL, NUM_THREADS) != 0){
        fprintf(stderr, "Error initializing barrier.\n");
        return 1;
    }

    pthread_t *threads = malloc(NUM_THREADS * sizeof(pthread_t));
    int thread_ids[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++){
        thread_ids[i] = i;
        if(pthread_create(&threads[i], NULL, lcs_thread_func, &thread_ids[i]) != 0){
            fprintf(stderr, "Error creating thread %d.\n", i);
            exit(EXIT_FAILURE);
        }
    }

    long long papi_time_start = PAPI_get_real_usec();

    for (int i = 0; i < NUM_THREADS; i++){
        pthread_join(threads[i], NULL);
    }

    long long papi_time_stop = PAPI_get_real_usec();

    // ✅ Calcolo corretto dell'indice finale per dp[len_A][len_B]
    int d_final = len_A + len_B;
    int i_min_final = (d_final - len_B > 1) ? d_final - len_B : 1;
    int k = len_A - i_min_final;  // offset nella diagonale finale
    int LCS_length = diag1[k];

    printf("Length of LCS is: %d\n", LCS_length);
    printf("Number of threads: %d\n", NUM_THREADS);
    printf("Total Execution Time: %lld μs\n", papi_time_stop - papi_time_start);

    pthread_barrier_destroy(&barrier);
    free(threads);
    free(diag2);
    free(diag1);
    free(diag_cur);
    free(str_A);
    free(str_B);

    PAPI_shutdown();

    return 0;
}
