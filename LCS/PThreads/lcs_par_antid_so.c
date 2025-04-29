#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <papi.h>

#define NUM_THREADS 16  // Number of threads to use

// Global variables
int len_A, len_B;       // String lengths
char *str_A, *str_B;    // Strings to compare
int max_diag;           // Max length of the diagonals
int *diag_prev2;        // Diagonal two steps behind the current one in the DP matrix
int *diag_prev1;        // Diagonal one step behind the current one in the DP matrix
int *diag_curr;         // Current diagonal on which we are working
int *diagonals;         // Array to hold all the three diagonals

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
void *lcs_thread_func (void *arg)
{
    // Get the thread ID
    int id = *(int *)arg;

    // Variables declaration for the loop
    int i_min, i_max, count, i_min_prev, count_prev, i_min_prev2, count_prev2, i_max_prev, i_max_prev2, chunk, remainder, start_index, end_index, top, left, diag_val, pos_top, pos_left, pos, i, j;

    // For each diagonal in the DP matrix
    // The diagonal d id processed in parallel by the threads
    // Each thread computes a subset of the elements in that diagonal
    for (int d = 2; d <= len_A + len_B; d++)
    {
        // Calculate the range of rows to consider for the current diagonal
        // i_min and i_max are the start and end indices of the rows to consider
        i_min = (d - len_B > 1) ? d - len_B : 1;
        i_max = (d - 1 < len_A) ? d - 1 : len_A;

        // Number of elements in the current diagonal
        count = i_max - i_min + 1;

        // For the first cycle, d = 2, so the previous diagonal has only two elements, both equal to 0.
        // So it's not necessary to compute the indices of the rows for the previous diagonal.
        i_min_prev = 0;
        count_prev = 0;
        if (d >= 3)
        {
            i_min_prev = ((d - 1) - len_B > 1) ? (d - 1) - len_B : 1;
            i_max_prev = ((d - 1) - 1 < len_A) ? d - 2 : len_A;
            count_prev = (i_max_prev - i_min_prev + 1 > 0) ? (i_max_prev - i_min_prev + 1) : 0;
        }

        // For the first and second cycle, d = 2 and 3, so the diagonal two steps back has respectively, one and two elements.
        // So, in these cases, it's not necessary to compute the indices of the rows for the diagonal two steps back.
        i_min_prev2 = 0;
        count_prev2 = 0;
        if (d >= 4)
        {
            i_min_prev2 = ((d - 2) - len_B > 1) ? (d - 2) - len_B : 1;
            i_max_prev2 = ((d - 2) - 1 < len_A) ? d - 3 : len_A;
            count_prev2 = (i_max_prev2 - i_min_prev2 + 1 > 0) ? (i_max_prev2 - i_min_prev2 + 1) : 0;
        }

        // Number of elements that each thread will process in the current diagonal
        chunk = count / NUM_THREADS;

        // Compute the number of remaining elements
        remainder = count % NUM_THREADS;

        // Distribute the remaining elements
        if (id < remainder)
        {
            start_index = id * (chunk + 1);
            end_index = start_index + (chunk + 1);
        }
        else
        {
            start_index = id * chunk + remainder;
            end_index = start_index + chunk;
        }

        // Each thread computes its own portion of the current diagonal
        // So, each thread will iterate over the elements in his portion of the diagonal
        // Precisely, the thread will iterate over the elements in his range [start_index, end_index)
        for (int k = start_index; k < end_index; k++) {
            
            // Compute the indices of the element (i,j) on which the thread will work
            i = i_min + k;
            j = d - i;

            // Compute the value of the element (i,j) in the DP matrix
            // If the characters match, take the diagonal value and add 1
            if (str_A[i - 1] == str_B[j - 1])
            {
                // If this isn't the first or the second iteration (d=2 or d=3), so the diagonal two steps back is not the first diagonal of the DP matrix (only one element = 0)
                // or the second diagonal of the DP matrix (only two elements, both = 0), take the diagonal value and add 1
                if (d >= 4)
                {
                    pos = (i - 1) - i_min_prev2;
                    diag_curr[k] = (pos >= 0 && pos < count_prev2) ? (diag_prev2[pos] + 1) : 1;
                }
                // Otherwise, we already know that the diagonal two steps back has only values equal to 0
                else
                {
                    diag_curr[k] = 1;
                }
            }
            // Otherwise, take the maximum of the left and top cells
            else
            {
                // If this isn't the first iteration (d=2), so the previous diagonal is not the second diagonal of the DP matrix (only two element, both = 0),
                // take the maximum of the left and top cells reading the values from the previous diagonal
                // Otherwise, we already know that the previous diagonal has only values equal to 0, so we take the maximum of 0 and 0
                top = 0, left = 0;
                if (d >= 3)
                {
                    // Compute the position of the top element in the previous diagonal and read its value
                    pos_top = (i - 1) - i_min_prev;
                    if (pos_top >= 0 && pos_top < count_prev) top = diag_prev1[pos_top];

                    // Compute the position of the left element in the previous diagonal and read its value
                    pos_left = i - i_min_prev;
                    if (pos_left >= 0 && pos_left < count_prev) left = diag_prev1[pos_left];
                }
                // Compute the element (i,j) in the DP matrix calculating the maximum of the left and top cells
                diag_curr[k] = MAX(top, left);
            }
        }

        // Synchronize the threads after each diagonal computation
        pthread_barrier_wait(&barrier);

        // Swap the diagonals for the next iteration
        if (id == 0) {
            int *temp = diag_prev2;
            diag_prev2 = diag_prev1;
            diag_prev1 = diag_curr;
            diag_curr = temp;
        }

        // Synchronize the threads after the swap
        pthread_barrier_wait(&barrier);
    }

    // Explicitly exit the thread
    pthread_exit(NULL);
    return NULL;
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

    // Compute the maximum length of the diagonals
    // The maximum length of the diagonals is the minimum of the lengths of the two strings plus one
    max_diag = (len_A < len_B ? len_A : len_B) + 1;

    // Allocate memory for the three diagonals
    diagonals = calloc((max_diag * 3), sizeof(int));

    // Initialize the pointers to each diagonal
    // The diagonals are stored in a single contiguous array, so we need to calculate the offsets
    diag_prev2 = &diagonals[max_diag * 2];
    diag_prev1 = &diagonals[max_diag];
    diag_curr = &diagonals[0];

    // Check if memory allocation was successful
    if(diag_prev2 == NULL || diag_prev1 == NULL || diag_curr == NULL){
        printf("Error: Memory allocation failed for diagonals.\n");
        return 1;
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
    for (int i = 0; i < NUM_THREADS; i++){
        thread_ids[i] = i;
        if(pthread_create(&threads[i], NULL, lcs_thread_func, &thread_ids[i]) != 0){
            fprintf(stderr, "Error creating thread %d.\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Wait for all threads to finish
    for (int i = 0; i < NUM_THREADS; i++){
        pthread_join(threads[i], NULL);
    }

    // Stop the timer
    long long papi_time_stop = PAPI_get_real_usec();

    // After the last iteration, due to the last swap, the length of the longest common subsequence (LCS) is stored in the first element of diag_prev2
    printf("Length of LCS is: %d\n", diag_prev2[0]);
    printf("Number of threads: %d\n", NUM_THREADS);
    printf("Total Execution Time: %lld μs\n", papi_time_stop - papi_time_start);

    // Free the allocated memory
    pthread_barrier_destroy(&barrier);
    free(threads);
    free(diagonals);
    free(str_A);
    free(str_B);

    // Stop the PAPI library
    PAPI_shutdown();

    return 0;
}