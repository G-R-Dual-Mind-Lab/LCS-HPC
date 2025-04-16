
/****
    Author: Rayhan Shikder,
    email: shikderr@myumanitoba.ca
    MSc Student,
    Department of Computer Science,
    University of Manitoba, Winnipeg, MB, Canada
****/


#include<stdio.h>
#include<string.h>
#include <stdlib.h>
#include <time.h>
#include <papi.h>

//macros
#define max(x,y) ((x)>(y)?(x):(y))


//global variables
char *string_A;
char *string_B;
char *unique_chars_C; //unique alphabets
int c_len;
short **DP_Results; //to store the DP values

//function prototypes
void print_matrix(int **x, int row, int col);
short lcs(short **DP, char *A, char *B, int m, int n);



void print_matrix(int **x, int row, int col)
{
    for(int i=0;i<row;i++)
    {
        for(int j=0;j<col;j++)
        {
            printf("%d ",x[i][j]);
        }
        printf("\n");
    }
}


short lcs(short **DP, char *A, char *B, int m, int n)
{
   // printf("%s %d \n%s %d\n",A,m,B,n );

    for(int i=1;i<(m+1);i++)
    {
        for(int j=1;j<(n+1);j++)
        {
            if(A[i-1] == B[j-1])
            {
                DP[i][j] = DP[i-1][j-1] + 1;
            }
            else
            {
                DP[i][j] = max(DP[i-1][j],DP[i][j-1]);
            }
        }
    }

    return DP[m][n];
}

int main(int argc, char *argv[])
{
    if(argc <= 1){
        printf("Error: No input file specified! Please specify the input file, and run again!\n");
        return 0;
    }
    printf("\nYour input file: %s \n",argv[1]);
    
    FILE *fp;
    int len_a,len_b;
    long long start_time,stop_time;
    int EventSet = PAPI_NULL;
    long long countCacheMiss[3]; // Array to store cache miss counts


    fp = fopen(argv[1], "r");
    fscanf(fp, "%d %d %d", &len_a, &len_b, &c_len);
    printf("Sequence lengths : %d %d %d\n", len_a, len_b, c_len );

    string_A = (char *)malloc((len_a+1) * sizeof(char *));
    string_B = (char *)malloc((len_b+1) * sizeof(char *));
    unique_chars_C = (char *)malloc((c_len+1) * sizeof(char *));

    fscanf(fp, "%s %s %s", string_A,string_B,unique_chars_C);


    //allocate memory for DP Results
    DP_Results = (short **)malloc((len_a+1) * sizeof(short *));
    for(int k=0;k<len_a+1;k++)
    {
        DP_Results[k] = (short *)calloc((len_b+1), sizeof(short));
    }

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

    //start_time = omp_get_wtime();
    start_time = PAPI_get_real_usec();

    printf("Length of LCS is: %d\n",lcs(DP_Results,string_A,string_B,len_a,len_b));

    stop_time = PAPI_get_real_usec();
    //stop_time = omp_get_wtime();

    printf("Time taken by sequential algorithm is: %lld μs\n",stop_time-start_time);
    printf("Cache miss L1: %lld\n", countCacheMiss[0]);
    printf("Cache miss L2: %lld\n", countCacheMiss[1]);
    printf("Cache miss L3: %lld\n", countCacheMiss[2]);

    //deallocate pointers
    free(DP_Results);

    /* Pulizia degli EventSet e della libreria PAPI */
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_shutdown();

    return 0;
}
