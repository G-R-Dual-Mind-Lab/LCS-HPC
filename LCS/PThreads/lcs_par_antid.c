#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <limits.h>
#include <papi.h>

#define NUM_THREADS 16 // Default number of threads

// Global variables
int len_A, len_B, **DP_matrix;
char *str_A, *str_B;
pthread_barrier_t barrier;


#define DP_MATRIX(i, j, DP_matrix) DP_matrix[i][j]


// Function executed by each thread.
// Each thread processes (in parallel) the assigned elements of each antidiagonal.
void *lcs_thread_func(void *thread_id) {

    int id = *(int *)thread_id;

    // The loop iterates over the antidiagonals of the DP matrix.
    // Each thread processes a portion of the cells in the current antidiagonal.
    // For the computation of the cells, the following observation is made:
    // - The index d (sum of indices) varies from 2 to (len_A+len_B);
    // - For each d, the valid indices are: i = max(1, d-len_B) ... min(len_A, d-1) with j = d - i
    for (int d = 2; d <= len_A + len_B; d++) {

        // Determine the range of rows for this antidiagonal
        int i_min = (d - len_B) > 1 ? (d - len_B) : 1; // max(1, d-len_B)
        int i_max = (d - 1) < len_A ? (d - 1) : len_A; // min(len_A, d-1)
        int count = i_max - i_min + 1; // Number of cells in the current antidiagonal

        // If the diagonal has no cells (rare case, but for safety) the thread waits
        // and moves to the next antidiagonal.
        if (count <= 0) {
            pthread_barrier_wait(&barrier);
            continue;
        }

        // Distiìribution of the cells among the threads.
        // Each thread will process a portion of the cells in the current antidiagonal.
        int chunk_size = count / NUM_THREADS;
        int remainder = count % NUM_THREADS;
        int start_index, end_index;
        if (id < remainder) { // If there are remaining elements, distribute them to the first threads
            start_index = id * (chunk_size + 1);
            end_index = start_index + (chunk_size + 1);
        } else {
            start_index = id * chunk_size + remainder;
            end_index = start_index + chunk_size;
        }

        // Compute the start and end indices on which the thread will work in the antidiagonal d.
        for (int index = start_index; index < end_index; index++) {
            int i = i_min + index;
            int j = d - i;

            int value = 0;
            if (str_A[i - 1] == str_B[j - 1])
            {
                // If the characters match, take the diagonal value and add 1.
                value = DP_MATRIX(i - 1, j - 1, DP_matrix) + 1;
            }
            else
            {
                // If the characters do not match, take the maximum of the left and top cells.
                int top = DP_MATRIX(i - 1, j, DP_matrix);
                int left = DP_MATRIX(i, j - 1, DP_matrix);
                value = (top > left) ? top : left;
            }
            DP_MATRIX(i, j, DP_matrix) = value;
        }
        // Sincronizza tutti i thread: attendi il completamento dell'antidiagonale.
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}

/* Funzione lcs: computa la lunghezza della LCS utilizzando la tecnica delle antidiagonali con PThreads.
   - DP_matrix: matrice DP già allocata (dimensione (len_A+1) x (len_B+1)) ed inizializzata a 0.
   - str_A, str_B: le stringhe di input.
   - len_A, len_B: lunghezze delle due stringhe.
   - t: numero di thread da utilizzare.
   Ritorna il valore memorizzato in DP[len_A][len_B] (lunghezza della LCS).  */
int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("\nError: No input file specified. Please specify the input file, and run again.\n");
        return 1;
    }

    // Check if the input file exists
    FILE *fp = fopen(argv[1], "r");
    if (fp == NULL) {
        perror("Error opening file");
        return 1;
    }
    
    // Declare variables
    long long papi_time_start, papi_time_stop; // Initialize PAPI time variables
    int EventSet = PAPI_NULL;
    long long countCacheMiss[3]; // Array to store cache miss counts

    // Initialize strings lengths
    fscanf(fp, "%d %d", &len_A, &len_B);
    if (len_A <= 0 || len_A >= INT_MAX || len_B <= 0 || len_B >= INT_MAX) {
        printf("Error: Invalid string lengths. Please check the input file.\n");
        fclose(fp);
        return 1;
    }

    // Print string lengths
    printf("String A length: %d\nString B lenght: %d\n", len_A, len_B);

    // Allocate memory for strings and unique characters
    str_A = malloc((len_A+1) * sizeof(char));
    str_B = malloc((len_B+1) * sizeof(char));

    DP_matrix = malloc((len_A + 1) * sizeof(int *));
    if (DP_matrix == NULL) {
        fprintf(stderr, "Errore: allocazione della matrice delle righe fallita!\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i <= len_A; i++) {
        DP_matrix[i] = calloc(len_B + 1, sizeof(int));
        if (DP_matrix[i] == NULL) {
            fprintf(stderr, "Errore: allocazione della riga %d fallita!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    fscanf(fp, "%s %s", str_A, str_B);

    // Check if the strings are empty
    if (str_A[0] == '\0' || str_B[0] == '\0') {
        printf("Error: One of the strings is empty. Please check the input file.\n");
        fclose(fp);
        return 1;
    }

    // Close the file after reading
    fclose(fp);

    // Initialize the PAPI library
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "Error: PAPI library initialization failed.\n");
        return 1;
    }

    // Create an event set for PAPI
    if(PAPI_create_eventset(&EventSet) != PAPI_OK) {
        printf("Error creating PAPI EventSet\n");
        exit(1);
    }

    // Add the L1 cache miss event to the event set
    if(PAPI_add_event(EventSet, PAPI_L1_TCM) != PAPI_OK){
        printf("Error adding event cache miss L1 in PAPI EventSet\n");
        exit(1);
    }
    
    // Add the L2 cache miss event to the event set
    if(PAPI_add_event(EventSet, PAPI_L2_TCM) != PAPI_OK){
        printf("Error adding event cache miss L2 in PAPI EventSet\n");
        exit(1);
    }

    // Add the L3 cache miss event to the event set
    if(PAPI_add_event(EventSet, PAPI_L3_TCM) != PAPI_OK){
        printf("Error adding event cache miss L3 in PAPI EventSet\n");
        exit(1);
    }  

    // Start the counting of events
    if (PAPI_start(EventSet) != PAPI_OK){
        printf("Error starting event counting\n");
        exit(1);
    }

    pthread_t *threads = malloc(NUM_THREADS * sizeof(pthread_t));

    // Inizializza il barrier per "t" thread.
    if (pthread_barrier_init(&barrier, NULL, NUM_THREADS)) {
        fprintf(stderr, "Errore nell'inizializzazione del pthread_barrier\n");
        exit(EXIT_FAILURE);
    }

    papi_time_start = PAPI_get_real_usec();
    
    int thread_ids[NUM_THREADS]; // Array per contenere gli ID dei thread
    // Crea i thread
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i; // Assegniamo l'ID ad ogni indice
        if (pthread_create(&threads[i], NULL, lcs_thread_func, &thread_ids[i])) {
            fprintf(stderr, "Errore nella creazione del thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Attendi il completamento di tutti i thread
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    papi_time_stop = PAPI_get_real_usec();

    // Stop the counting of events
    if (PAPI_stop(EventSet, countCacheMiss) != PAPI_OK) {
        printf("Stop and store counter error\n");
        exit(1);
    }

    printf("Length of LCS is: %d\n", DP_MATRIX(len_A, len_B, DP_matrix));

    printf("Number of threads: %d\n", NUM_THREADS);

    printf("Total Execution Time: %lld μs\n", papi_time_stop - papi_time_start);
    
    printf("Cache miss L1: %lld\n", countCacheMiss[0]);
    printf("Cache miss L2: %lld\n", countCacheMiss[1]);
    printf("Cache miss L3: %lld\n", countCacheMiss[2]);

    // Distruggi il barrier e libera le risorse allocate per i thread
    pthread_barrier_destroy(&barrier);
    free(threads);

    for (int i = 0; i <= len_A; i++) {
        free(DP_matrix[i]);
    }
    free(DP_matrix);
    
    /* Pulizia degli EventSet e della libreria PAPI */
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_shutdown();
    
    return 0;
}