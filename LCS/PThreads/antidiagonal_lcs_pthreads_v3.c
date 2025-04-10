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
// Ogni thread processa una porzione degli elementi della diagonale corrente.
//
void *lcs_thread_func(void *arg) {

    int id = *(int *)arg;
    // Loop sulle diagonali: d varia da 2 fino a (len_A + len_B)
    // (Ricorda: la matrice DP è concettualmente 1-indexed; i=0 e j=0 sono i bordi inizializzati a 0)
    for (int d = 2; d <= len_A + len_B; d++) {

        current_d = d;  // Aggiorna la diagonale corrente globale

        // Calcola il range della diagonale corrente:
        //   i_min = max(1, d - len_B)
        //   i_max = min(len_A, d - 1)
        int i_min = (d - len_B > 1) ? d - len_B : 1; //1
        int i_max = (d - 1 < len_A) ? d - 1 : len_A; //2
        int count = i_max - i_min + 1;  // Numero di elementi in questa diagonale //2

        // Per le diagonali precedenti esistono se d>=3 (per diag1) e d>=4 (per diag2)
        int A_prev = 0, count_prev = 0;  // Per diagonale d-1
        if (d >= 3) {
            A_prev = ((d - 1) - len_B > 1) ? (d - 1) - len_B : 1; //1
            int i_max_prev = ((d - 1) - 1 < len_A) ? d - 2 : len_A; //1
            count_prev = (i_max_prev - A_prev + 1 > 0) ? (i_max_prev - A_prev + 1) : 0; //1
        }
        int A_prev2 = 0, count_prev2 = 0;  // Per diagonale d-2
        if (d >= 4) {
            A_prev2 = ((d - 2) - len_B > 1) ? (d - 2) - len_B : 1;
            int i_max_prev2 = ((d - 2) - 1 < len_A) ? d - 3 : len_A;
            count_prev2 = (i_max_prev2 - A_prev2 + 1 > 0) ? (i_max_prev2 - A_prev2 + 1) : 0;
        }

        // Dividi il lavoro (0..count-1) tra i thread
        int chunk = count / NUM_THREADS; //1
        int rem = count % NUM_THREADS;
        int start, end;
        if (id < rem) {
            start = id * (chunk + 1);
            end = start + (chunk + 1);
        } else {
            start = id * chunk + rem; //0 //1
            end = start + chunk; //1 //2
        }

        // Elaborazione parallela degli elementi della diagonale corrente
        for (int k = start; k < end; k++) {
            int i = i_min + k; //1 //2
            int j = d - i;  // Perché i + j = d //2 //1

            int val = 0;
            if (str_A[i - 1] == str_B[j - 1]) {
                // Se i caratteri corrispondono, dp[i][j] = dp[i-1][j-1] + 1.
                // dp[i-1][j-1] si trova nella diagonale d-2.
                // L’indice locale in diag2 corrispondente a cella (i-1, j-1) è: (i-1) - A_prev2.
                int diag_val = 0;
                if (d >= 3 && k - 1 >= 0) { // d>=3 in realtà richiede d>=4 per avere diag2
                    int pos = (i - 1) - A_prev2; //1
                    if (pos >= 0 && pos < count_prev2)
                        diag_val = diag2[pos];
                }
                val = diag_val + 1;
            } else {
                // Altrimenti, dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                int top = 0, left = 0;
                if (d >= 3) {
                    int pos_top = (i - 1) - A_prev;  // in diag1 //-1
                    if (pos_top >= 0 && pos_top < count_prev)
                        top = diag1[pos_top];
                    int pos_left = i - A_prev; //0
                    if (pos_left >= 0 && pos_left < count_prev)
                        left = diag1[pos_left];
                }
                val = max_int(top, left);
            }
            diag_cur[k] = val;
        }

        // Synchronizza tutti i thread sul completamento della diagonale corrente
        pthread_barrier_wait(&barrier);

        // Un solo thread (ad es. id == 0) ruota i buffer per la prossima iterazione.
        if (id == 0) {
            // Ruota: diag2 <- diag1, diag1 <- diag_cur, e riutilizza il buffer diag_cur per il prossimo ciclo.
            int *temp = diag2;
            diag2 = diag1;
            diag1 = diag_cur;
            diag_cur = temp;
            // Azzeriamo diag_cur (che ora sarà usato per scrivere la nuova diagonale)
            memset(diag_cur, 0, max_diag * sizeof(int));
        }

        // Sincronizza di nuovo affinché tutti i thread vedano i buffer aggiornati.
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

    // Legge le lunghezze delle stringhe dal file
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

    // Alloca le stringhe (si assume che non superino la lunghezza indicata)
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

    // Inizializza PAPI (non strettamente necessario per l’algoritmo LCS)
    int EventSet = PAPI_NULL;
    if(PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT){
        fprintf(stderr, "Error: PAPI initialization failed.\n");
        return 1;
    }
    // (Gestione degli eventi PAPI omessa per brevità)

    // Imposta il buffer per le diagonali:
    // La dimensione massima di ciascuna diagonale è min(len_A, len_B) + 1.
    max_diag = (len_A < len_B ? len_A : len_B) + 1;
    diag2 = calloc(max_diag, sizeof(int));
    diag1 = calloc(max_diag, sizeof(int));
    diag_cur = calloc(max_diag, sizeof(int));
    if(diag2 == NULL || diag1 == NULL || diag_cur == NULL){
        printf("Error: Memory allocation failed for diagonals.\n");
        return 1;
    }

    // Inizializza la barriera per NUM_THREADS
    if(pthread_barrier_init(&barrier, NULL, NUM_THREADS) != 0){
        fprintf(stderr, "Error initializing barrier.\n");
        return 1;
    }

    // Crea i thread
    pthread_t *threads = malloc(NUM_THREADS * sizeof(pthread_t));
    int thread_ids[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++){
        thread_ids[i] = i;
        if(pthread_create(&threads[i], NULL, lcs_thread_func, &thread_ids[i]) != 0){
            fprintf(stderr, "Error creating thread %d.\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Avvia il timer PAPI
    long long papi_time_start = PAPI_get_real_usec();

    // Attendi la terminazione dei thread
    for (int i = 0; i < NUM_THREADS; i++){
        pthread_join(threads[i], NULL);
    }

    long long papi_time_stop = PAPI_get_real_usec();

    // Il risultato (LCS) si trova nell'ultima diagonale calcolata (che ora è in diag1)
    // Nella diagonale finale d = len_A + len_B,
    // la cella dp[len_A][len_B] corrisponde a indice = len_A - A_final, dove A_final = max(1, d - len_B).
    int d_final = len_A + len_B;
    int A_final = (d_final - len_B > 1) ? d_final - len_B : 1;
    int idx_result = len_A - A_final;
    int LCS_length = diag1[idx_result];

    printf("Length of LCS is: %d\n", LCS_length);
    printf("Number of threads: %d\n", NUM_THREADS);
    printf("Total Execution Time: %lld μs\n", papi_time_stop - papi_time_start);

    // Qui, eventualmente, stampare altri contatori PAPI...

    // Pulisci le risorse
    pthread_barrier_destroy(&barrier);
    free(threads);
    free(diag2);
    free(diag1);
    free(diag_cur);
    free(str_A);
    free(str_B);
    
    // Pulizia PAPI
    PAPI_shutdown();
    
    return 0;
}