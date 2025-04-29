#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <limits.h>
#include <papi.h>

#define NUM_THREADS 16  // Number of threads to use

// Global variables
int len_A, len_B;       // String lengths
char *str_A, *str_B;    // Strings to compare
int **DP_matrix;        // DP matrix for LCS computation

// Barrier for thread synchronization
pthread_barrier_t barrier;

// Macro for calculating the maximum between two values
#define MAX(x, y) ((x) > (y) ? (x) : (y))

/*
 * Function: lcs_thread_func
 * ----------------------
 * Each thread executes this function and, using the wavefront algorithm, for each diagonal
 * computes a subset of the elements in that diagonal.
 * The function uses a barrier to synchronize the threads after each diagonal computation.
 * Then, it swaps the diagonals to prepare for the next iteration.
 */
void *lcs_thread_func (void *thread_id)
{
    // Get the thread ID
    int id = *(int *)thread_id;

    // Variables declaration for the loop
    int i_min, i_max, count, chunk_size, remainder, start_index, end_index, k, i, j, top, left, value;

    // For each diagonal in the DP matrix
    // The diagonal d id processed in parallel by the threads
    // Each thread computes a subset of the elements in that diagonal
    for (int d = 2; d <= len_A + len_B; d++)
    {
        // Calculate the range of rows to consider for the current diagonal
        // i_min and i_max are the start and end indices of the rows to consider
        i_min = (d - len_B) > 1 ? (d - len_B) : 1;
        i_max = (d - 1) < len_A ? (d - 1) : len_A;

        // Number of elements in the current diagonal
        count = i_max - i_min + 1;

        // Number of elements that each thread will process in the current diagonal
        chunk_size = count / NUM_THREADS;

        // Compute the number of remaining elements
        remainder = count % NUM_THREADS;

        // Distribute the remaining elements
        if (id < remainder)
        {
            start_index = id * (chunk_size + 1);
            end_index = start_index + (chunk_size + 1);
        }
        else
        {
            start_index = id * chunk_size + remainder;
            end_index = start_index + chunk_size;
        }

        // Each thread computes its own portion of the current diagonal
        // So, each thread will iterate over the elements in his portion of the diagonal
        // Precisely, the thread will iterate over the elements in his range [start_index, end_index)
        for (k = start_index; k < end_index; k++) {

            // Compute the indices of the element (i,j) on which the thread will work
            i = i_min + k;
            j = d - i;
            
            // Initialize the value to 0
            value = 0;

            // Compute the value of the element (i,j) in the DP matrix
            // If the characters match, take the diagonal value and add 1
            if (str_A[i - 1] == str_B[j - 1])
            {
                value = DP_matrix[i-1][j-1] + 1;
            }
            else
            {
                // Otherwise, take the maximum of the left and top cells
                value = MAX(DP_matrix[i - 1][j], DP_matrix[i][j - 1]);
            }
            DP_matrix[i][j] = value;
        }

        // Synchronize the threads after each diagonal computation
        pthread_barrier_wait(&barrier);
    }

    // Explicitly exit the thread
    pthread_exit(NULL);
}

/*
 * Function: main
 * ----------------------
 * This is the main function of the program.
 * It reads the input file, initializes the PAPI library, and creates the threads to compute the LCS.
 * It also handles the memory allocation and deallocation for the strings and diagonals.
 * Finally, it prints the length of the LCS and the execution time.
 * When the program is executed, it takes the input file name as a command line argument.
 * The input file should contain the lengths of the two strings followed by the strings themselves, in the format:
 * <len_a> <len_b>\n<string_A>\n<string_B>
 */
int main (int argc, char *argv[])
{
    // Check if the input file is provided
    if(argc < 2){
        printf("Error: No input file specified.\n");
        return 1;
    }

    // Check if the input file exists
    FILE *fp = fopen(argv[1], "r");
    if(fp == NULL){
        perror("Error opening file");
        return 1;
    }

    // Initialize strings lengths
    if(fscanf(fp, "%d %d", &len_A, &len_B) != 2){
        printf("Error: Unable to read string lengths.\n");
        fclose(fp);
        return 1;
    }

    // Check if the lengths are valid
    if(len_A <= 0 || len_B <= 0){
        printf("Error: Invalid string lengths.\n");
        fclose(fp);
        return 1;
    }

    printf("String A length: %d\nString B length: %d\n", len_A, len_B);
    
    // Allocate memory for the strings
    // +1 for the null terminator
    str_A = malloc((len_A + 1) * sizeof(char));
    str_B = malloc((len_B + 1) * sizeof(char));

    // Check if memory allocation was successful
    if(str_A == NULL || str_B == NULL){
        printf("Error: Memory allocation failed for strings.\n");
        fclose(fp);
        return 1;
    }

    // Read the strings from the file
    if(fscanf(fp, "%s %s", str_A, str_B) != 2){
        printf("Error: Unable to read strings.\n");
        fclose(fp);
        return 1;
    }

    // Close the file after reading
    fclose(fp);

    // Initialize PAPI library
    if(PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT){
        fprintf(stderr, "Error: PAPI initialization failed.\n");
        return 1;
    }

    // Allocate memory for the pointers to the rows of the DP matrix
    DP_matrix = malloc((len_A + 1) * sizeof(int *));
    if (DP_matrix == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for DP matrix.\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory for each row of the DP matrix
    for (int i = 0; i <= len_A; i++)
    {
        DP_matrix[i] = calloc(len_B + 1, sizeof(int));
        if (DP_matrix[i] == NULL) {
            fprintf(stderr, "Error: Memory allocation failed for DP matrix row %d.\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Initialize the pthread barrier
    if(pthread_barrier_init(&barrier, NULL, NUM_THREADS) != 0){
        fprintf(stderr, "Error initializing barrier.\n");
        return 1;
    }

    // Start the timer to measure elapsed time
    long long papi_time_start = PAPI_get_real_usec();

    // Create NUM_THREADS threads to compute the LCS
    // Each thread will execute the lcs_thread_func function
    pthread_t *threads = malloc(NUM_THREADS * sizeof(pthread_t));   
    int thread_ids[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        if (pthread_create(&threads[i], NULL, lcs_thread_func, &thread_ids[i])) {
            fprintf(stderr, "Errore nella creazione del thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Wait for all threads to finish
    for (int i = 0; i < NUM_THREADS; i++){
        pthread_join(threads[i], NULL);
    }

    // Stop the timer
    long long papi_time_stop = PAPI_get_real_usec();

    printf("Length of LCS is: %d\n", DP_matrix[len_A][len_B]);
    printf("Number of threads: %d\n", NUM_THREADS);
    printf("Total Execution Time: %lld μs\n", papi_time_stop - papi_time_start);

    // Free the allocated memory
    pthread_barrier_destroy(&barrier);
    free(threads);
    
    for (int i = 0; i <= len_A; i++)
    {
        free(DP_matrix[i]);
    }

    free(DP_matrix);

    free(str_A);
    free(str_B);

    // Stop the PAPI library
    PAPI_shutdown();
    
    return 0;
}