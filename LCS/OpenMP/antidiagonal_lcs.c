#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>
#include <papi.h>

typedef long long ll;

// Macro to access the flattened DP_matrix matrix (on a single line)
// The matrix is of size (m+1) x (n+1)
#define DP_MATRIX(i,j) DP_matrix[(i)*(n+1) + (j)]

/* Funzione che calcola la LCS usando il metodo “antidiagonale” con OpenMP.
   X ed Y sono le stringhe, m ed n le loro lunghezze, t il numero di thread da usare. */
ll lcs(unsigned short *DP_matrix, char *str_A, char *str_B, int m, int n, int t) {

    omp_set_num_threads((int)t);
    
    ll i, j, ii, jj, k_index;
    
    /* Calcolo in antidiagonale */
    for (i = 1, j = 1; j <= n && i <= m; j++) {

        /* Il numero di elementi sulla diagonale corrente è il minimo tra j ed (m-i) */
        ll sz = (j < (m - i)) ? j : (m - i);

        #pragma omp parallel shared(i, j, dp, X, Y, sz) // creazione thread
        {
            #pragma omp for // divido il lavoro tra i thread
            for (k_index = 0; k_index <= sz; ++k_index) {

                ii = i + k_index; 
                jj = j - k_index; 

                if (str_A[ii - 1] == str_B[jj - 1]) // confronto i caratteri
                    DP_MATRIX(ii, jj) = DP_MATRIX(ii - 1, jj - 1) + 1;
                else
                    DP_MATRIX(ii, jj) = (DP_MATRIX(ii - 1, jj) > DP_MATRIX(ii, jj - 1)) ? DP_MATRIX(ii - 1, jj) : DP_MATRIX(ii, jj - 1);
            }
            #pragma omp barrier
        }
        if (j >= n) {
            j = n - 1;
            i++;
        }
    }
    
    return DP_MATRIX(m, n);
}

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
    int len_a, len_b, len_c; // Lengths of strings A, B, and C
    long long papi_time_start, papi_time_stop; // Initialize PAPI time variables
    int EventSet = PAPI_NULL;
    long long countCacheMiss[3]; // Array to store cache miss counts

    // Initialize strings lengths
    fscanf(fp, "%d %d %d", &len_a, &len_b, &len_c);
    if (len_a <= 0 || len_a >= USHRT_MAX || len_b <= 0 || len_b >= USHRT_MAX || len_c <= 0) {
        printf("Error: Invalid string lengths. Please check the input file.\n");
        fclose(fp);
        return 1;
    }

    // Print string lengths
    printf("String A length: %d\nString B lenght: %d\nAlphabet lenght: %d\n", len_a, len_b, len_c);

    // Allocate memory for strings and unique characters
    char *string_A = (char *)malloc((len_a+1) * sizeof(char));
    char *string_B = (char *)malloc((len_b+1) * sizeof(char));
    char *unique_chars_C = (char *)malloc((len_c+1) * sizeof(char));

    fscanf(fp, "%s %s %s", string_A, string_B, unique_chars_C);

    // Check if the strings are empty
    if (string_A[0] == '\0' || string_B[0] == '\0' || unique_chars_C[0] == '\0') {
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
    
    /* Array di thread da utilizzare per ciascuna misurazione */
    int thread_arr[] = {1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64};
    int thread_arr_size = sizeof(thread_arr) / sizeof(thread_arr[0]);
    ll result;
    int k;
    
    for (k = 0; k < thread_arr_size; k++) {
        
        // Allocate memory for DP_matrix matrix
        // DP_matrix matrix is of size (len_a+1) x (len_b+1)
        unsigned short *DP_matrix = (unsigned short *)calloc((len_a + 1) * (len_b + 1), sizeof(unsigned short));

        // Start the counting of events
        if (PAPI_start(EventSet) != PAPI_OK){
            printf("Error starting event counting\n");
            exit(1);
        }

        papi_time_start = PAPI_get_real_usec();
        
        result = lcs(DP_matrix, string_A, string_B, len_a, len_b, thread_arr[k]);
        
        papi_time_stop = PAPI_get_real_usec();

         // Stop the counting of events
        if (PAPI_stop(EventSet, countCacheMiss) != PAPI_OK) {
            printf("Stop and store counter error\n");
            exit(1);
        }

        printf("Length of LCS is %lld\n", result);
        printf("Total Execution Time: %lld μs\n", papi_time_stop - papi_time_start);
        
        printf("Cache miss L1: %lld\n", countCacheMiss[0]);
        printf("Cache miss L2: %lld\n", countCacheMiss[1]);
        printf("Cache miss L3: %lld\n", countCacheMiss[2]);

        free(DP_matrix);
        DP_matrix = NULL;
    }
    
    /* Pulizia degli EventSet e della libreria PAPI */
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_shutdown();
    
    return 0;
}
