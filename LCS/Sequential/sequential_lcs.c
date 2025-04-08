#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <papi.h>

// Macro for calculating the maximum between two values
#define MAX(x, y) ((x) > (y) ? (x) : (y))

// Function to calculate the length of the longest common subsequence (LCS)
unsigned short lcs(unsigned short *DP_matrix, char *A, char *B, int m, int n)
{
    for (int i = 1; i <= m; i++) {
       
        // Access the current row and previous row in the flattened DP_matrix
        // This avoids a lot of cache misses
        unsigned short *currRow = DP_matrix + i * (n + 1);
        unsigned short *prevRow = DP_matrix + (i - 1) * (n + 1);

        // Get the character from string A for the current row only when value of i changes
        char a_i = A[i - 1];

        for (int j = 1; j <= n; j++) {

            if (a_i != B[j - 1])
            {
                currRow[j] = MAX(prevRow[j], currRow[j - 1]);
            }
            else
            {
                currRow[j] = prevRow[j - 1] + 1;
            }
        }
    }

    return DP_matrix[m * (n + 1) + n];
}

int main(int argc, char *argv[])
{
    // Check if the input file is provided
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

    // Allocate memory for DP_matrix matrix
    // DP_matrix matrix is of size (len_a+1) x (len_b+1)
    unsigned short *DP_matrix = (unsigned short *)calloc((len_a + 1) * (len_b + 1), sizeof(unsigned short));

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

    // Start the timer and run the LCS algorithm
    papi_time_start = PAPI_get_real_usec();
    printf("Length of LCS is: %d\n", lcs(DP_matrix, string_A, string_B, len_a, len_b));

    // Stop the timer and print the time taken
    papi_time_stop = PAPI_get_real_usec();
    printf("Time taken by sequential algorithm is: %lld μs\n", papi_time_stop - papi_time_start);

    // Stop the counting of events
    if (PAPI_stop(EventSet, countCacheMiss) != PAPI_OK) {
        printf("Stop and store counter error\n");
        exit(1);
    }

    // Print the cache miss counts
    printf("Cache miss L1: %lld\n", countCacheMiss[0]);
    printf("Cache miss L2: %lld\n", countCacheMiss[1]);
    printf("Cache miss L3: %lld\n", countCacheMiss[2]);

    // Remove the event from the event set and destroy it
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);

    // Shutdown PAPI library
    PAPI_shutdown();

    // Deallocate heap memory
    free(string_A);
    free(string_B);
    free(unique_chars_C);
    free(DP_matrix);
    
    return 0;
}