#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <papi.h>

// Macro for calculating the maximum between two values
#define MAX(x, y) ((x) > (y) ? (x) : (y))

// Function to calculate and return the length of the longest common subsequence (LCS)
// This function uses a space-optimized approach with two rows of the DP matrix.
// The function takes the following parameters:
// - two_rows_dp_matrix: A pointer to the 1D array representing the two rows of the DP matrix;
// - A: The first input string;
// - B: The second input string;
// - len_A: The length of the first string;
// - len_B: The length of the second string.
// The function returns the length of the LCS as an unsigned integer.
unsigned int lcs(unsigned int *two_rows_dp_matrix, char *A, char *B, unsigned int len_A, unsigned int len_B)
{
    char is_copy_possible = 0; // Flag to check for adjacent duplicates in string A
    unsigned int *currRow = &two_rows_dp_matrix[len_B+1]; // Pointer to the current row in the DP matrix
    unsigned int *prevRow = &two_rows_dp_matrix[0]; // Pointer to the previous row in the DP matrix
    unsigned int *temp = NULL; // Temporary pointer for swapping rows

    // The outer loop iterates over the characters of string A (the rows of the DP matrix)
    for (unsigned int i = 1; i <= len_A; i++) {

        // Get the character of string A for the current row
        char a_i = A[i - 1];
        
        // If the current character of A is the same as the previous one,
        // it's possibile to copy the value from the previous row until a match is found.
        // This avoids unnecessary calculations and cache misses.
        is_copy_possible = (i >= 2) && (a_i == A[i - 2]);

        // The inner loop iterates over the characters of string B (the columns of the DP matrix)
        for (unsigned int j = 1; j <= len_B; j++) {
        
            // If the characters of A and B are not equal
            if (a_i != B[j - 1])
            {   
                // If the current character of A is not a duplicate of the previous one, so we can't copy the value from the previous row
                if (!is_copy_possible)
                {   
                    // If we are on the first column, we know that the first column of the DP matrix is always 0
                    // So we can copy the value from the previous row whitout calculating the maximum
                    // between the left and top cells, because the left cell is always 0.
                    // This avoids unnecessary calculations and cache misses.
                    if (j == 1)
                    {
                        currRow[j] = prevRow[j];
                    }
                    else
                    {
                        // Otherwise, we take the maximum of the left and top cells
                        currRow[j] = MAX(prevRow[j], currRow[j - 1]);
                    }
                }
                else
                {
                    // If the current character of A is a duplicate of the previous one, we can copy the value from the previous row
                    // This avoids unnecessary calculations and cache misses.
                    currRow[j] = prevRow[j];
                }
            }
            else // If the characters of A and B are equal
            {   
                // If the current character of A is not a duplicate of the previous one, so we can't copy the value from the previous row
                if(!is_copy_possible)
                {
                    // Take the diagonal value and add 1
                    currRow[j] = prevRow[j - 1] + 1;
                }
                else
                {
                    // If adjacent duplicate is found, the first time that the character is equal, copy the value from the previous row
                    currRow[j] = prevRow[j];
                }

                // Reset the flag if characters are equal, because from now on we can't copy anymore the value from the previous row
                is_copy_possible = 0;
            }
        }

        // Swap the pointers for the next iteration
        temp = prevRow;
        prevRow = currRow;
        currRow = temp;
    }

    // The length of the LCS is stored in the last cell of the previous row
    // This is because we are using two rows of the DP matrix and due to the previous swap operation,
    // the last row is now the previous row.
    return prevRow[len_B];
}

// Main function to read input, initialize PAPI, and run the LCS algorithm
// The file name is passed as a command line argument and neeeds to be in the format:
// <len_a> <len_b>\n<string_A>\n<string_B>
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
    unsigned int len_a, len_b; // Lengths of strings A, B, and C
    long long papi_time_start, papi_time_stop; // Initialize PAPI time variables
    int EventSet = PAPI_NULL; // Event set for PAPI
    long long countCacheMiss[3]; // Array to store cache miss counts

    // Initialize strings lengths
    fscanf(fp, "%d %d", &len_a, &len_b);
    if (len_a == 0 || len_a >= UINT_MAX || len_b == 0 || len_b >= UINT_MAX) {
        printf("Error: Invalid string lengths. Please check the input file.\n");
        fclose(fp);
        return 1;
    }

    // Print string lengths
    printf("String A length: %u\nString B length: %u\n", len_a, len_b);

    // Allocate memory for strings and unique characters
    char *string_A = (char *)malloc((len_a + 1) * sizeof(char));
    char *string_B = (char *)malloc((len_b + 1) * sizeof(char));

    if (string_A == NULL || string_B == NULL) {
        printf("Error: Memory allocation failed for strings.\n");
        fclose(fp);
        return 1;
    }

    // Read the strings from the file
    fscanf(fp, "%s %s", string_A, string_B);

    // Check if the strings are empty
    if (string_A[0] == '\0' || string_B[0] == '\0') {
        printf("Error: One of the strings is empty. Please check the input file.\n");
        fclose(fp);
        return 1;
    }

    // Close the file after reading
    if (fp != NULL) {
        fclose(fp);
    }

    // Allocate memory for two rows DP_matrix matrix
    // In this way, we optimise the space complexity of the algorithm from O(n*m) to O(2*m)
    unsigned int *two_rows_dp_matrix = (int *)calloc((2) * (len_b + 1), sizeof(int));

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

    printf("Length of LCS is: %u\n", lcs(two_rows_dp_matrix, string_A, string_B, len_a, len_b));

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
    free(two_rows_dp_matrix);
    
    return 0;
}