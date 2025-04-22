#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <pthread.h>
#include <papi.h>
#include <time.h>
#include "omp.h"

#define NUM_WORKER_THREADS 16                      // Numero di thread
#define TILE_DIM 1000                        // Dimensione della tile (blocco)
#define HOST_BUFF 256                       // Dimensione del buffer per il nome host
#define TAG_TASK 0                          // Tag dei messaggi di tipo "invio task"
#define TAG_RESULT 1                        // Tag dei messaggi di tipo "invio task"

/*
    Definizione della struttura Task, che rappresenta un task da eseguire.
    Ogni task contiene informazioni relative alla porzione di matrice da elaborare,
    come gli indici di partenza per le stringhe A e B, i bordi superiori e sinistri,
    l'angolo del task e i flag per indicare se i bordi sono pronti.
*/
typedef struct {
    int task_id[2];                         // ID univoco del task
    int start_index_sub_a;                  // Inidce iniziale della porzione di stringa A da considerare
    int start_index_sub_b;                  // Inidce iniziale della porzione di stringa B da considerare
    int top_row[TILE_DIM];                  // Riga superiore (bordo superiore)
    int left_col[TILE_DIM];                 // Colonna sinistra (bordo sinistro)
    int angle;                              // Angolo (diagonale) del task
    char top_row_ready;                     // Flag per indicare se la riga superiore è pronta (1) o meno (0)
    char left_col_ready;                    // Flag per indicare se la colonna sinistra è pronta (1) o meno (0)  
    char angle_ready;                       // Flag per indicare se l'angolo è pronto (1) o meno (0) 
    char initialized;                       // Flag per indicare se il task è inizializzato (1) o meno (0)              
} Task;

/*
    Definizione della struttura Result, che rappresenta il risultato di un task.
    Ogni risultato contiene gli ID del task e i bordi destro e inferiore.
    Questi bordi vengono utilizzati per iniettare dipendenze nei task successivi.
*/
typedef struct{
    int task_id[2];   
    int right_col[TILE_DIM];               // Colonna destra (bordo destro)                    
    int bottom_row[TILE_DIM];              // Riga inferiore (bordo inferiore)
} Result;

// MACRO
#define INITIALIZE_TASK(task, task_counter) { \
    task_queue[task_counter].task_id[0] = task.task_id[0]; \
    task_queue[task_counter].task_id[1] = task.task_id[1]; \
    task_queue[task_counter].start_index_sub_a = task.start_index_sub_a; \
    task_queue[task_counter].start_index_sub_b = task.start_index_sub_b; \
    __atomic_thread_fence(__ATOMIC_RELEASE); \
    task_queue[task_counter].initialized = 1; \
}
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DP_MATRIX(i, j, DP_matrix) DP_matrix[i][j]

/* Seguono i prototipi di funzione */
void handle_PAPI_error(int, char*);
Task create_task(int i, int j, int dim, char *A, char *B, int len_a, int len_b);
void carico_fittizio(int iterazioni);
Result lcs_yang_v2(int **DP, int **P, char *A, char *B, char *C, int m, int n, int u, int offset, int flag);

/* Seguono i prototipi di funzione di thread (funzioni eseguibili dai thread) */
void *task_producer(void *args);
void *task_sender(void *args);
void *pending_task_sender(void *args);
void *send_sequences(void *args);
void *lcs_thread_func(void *thread_id);

int num_processes;                          /* Number of processes in MPI_COMM_WORLD */

// AREA DATI GLOBALE: Variabili globali processo master
Task *task_queue;                           // Coda dei task (da inviare ai worker)
int finished_generating;                    /* Flag per indicare se il master ha finito di generare task */
int num_blocks_rows, num_blocks_cols;       /* Number of blocks in rows and columns */
int num_blocks;                             /* Number of blocks in the matrix (after tailing) */
int num_antidiagonals;                      /* Numero totale di antidiagonali */
int *pending_task_index;                    /* Indice del task non inizializzato (da inviare al worker) */
int max_antidiagonal_length;                /* Lunghezza in blocchi della massima antidiagonale */
int rank_worker;                            /* Rank del worker a cui devo inviare il messaggio */
pthread_mutex_t rank_worker_mutex;          /* Mutex per proteggere l'accesso alla variabile rank_worker */
int max_rank_worker;                        /* Massimo rank tra i worker a cui posso inviare un messaggio */
char stop_pending_sender;
int producer_start = 0;                     /* Flag per indicare se il produttore ha iniziato a produrre task */
int lcs_length = 0;                       /* Lunghezza della LCS (longest common subsequence) */

// AREA DATI GLOBALE: Variabili globali master + worker
int rank;                                   /* Current process identifier */
char hn[HOST_BUFF];                         /* Hostname of the machine */
char *string_A, *string_B, *alphabet;       /* Pointers to the two strings and alphabet */
int string_lengths[3];                      /* Array to store the lengths of the two strings */
int offset_A, offset_B;

// AREA DATI GLOBALE: Variabili globali worker
int **DP_matrix;
pthread_barrier_t barrier;
Result result;                          // risultato del task (da inviare al master)

int main(int argc, char *argv[])
{
    /*
        Ogni processo MPI avrà queste variabili (ma attenzione non sono globali, sono private, locali a ciascun processo).
        
        Inoltre, anche ciascun thread (sottoinsieme del processo) di ogni processo vedrà qieste variabili.
        A livello di thread queste variabili sono globali (condivisione di memoria).
    */ 

    int provided;                           /* MPI thread level supported */
    int rc;                                 /* Return code used in error handling */
    long_long time_start, time_stop;        /* To measure execution time */
    long_long num_cache_miss;               /* To measure number of cache misses */
    int event_set = PAPI_NULL;              /* Group of hardware events for PAPI library */

    // Start MPI setup (ogni processo MPI esegue questo codice)
    if((rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided)) != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init error. Return code: %d\n", rc);
        exit(-1);
    } 
    if(provided < MPI_THREAD_SERIALIZED) {
        fprintf(stderr, "Minimum MPI threading level requested: %d (provided: %d)\n", MPI_THREAD_SERIALIZED, provided);
        exit(-1);
    }
    // End MPI setup

    // Start PAPI setup (ogni processo MPI esegue questo codice)
    if((rc = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
        handle_PAPI_error(rc, "Error in library init.");
    if((rc = PAPI_create_eventset(&event_set)) != PAPI_OK)
        handle_PAPI_error(rc, "Error while creating the PAPI eventset.");
    if((rc = PAPI_add_event(event_set, PAPI_L2_TCM)) != PAPI_OK)
        handle_PAPI_error(rc, "Error while adding L2 total cache miss event.");
    if((rc = PAPI_start(event_set)) != PAPI_OK) 
        handle_PAPI_error(rc, "Error in PAPI_start().");
    // End PAPI setup

    // Ogni processo MPI esegue questo codice
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // recupero il rank del processo e la dimensione del comunicatore MPI_COMM_WORLD
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    gethostname(hn, HOST_BUFF); // recupero il nome dell'host (della macchina su cui è eseguito il processo)
    max_rank_worker = num_processes - 1;

    if (!rank) {    // se sei il processo master (rank 0)

        /*
            Quì dentro ci entra solo il processo master (rank 0).
            Le variabili dichiarate in questa sezione di codice sono visibili a tutti i thread del processo master.
        */

        FILE *fp;                                   /* File pointer for input file */
        stop_pending_sender = 0;                     /* Flag per terminare il thread pending_task_sender */

        /* 
            Inizializzo una mutex che sarà visibile a tutti i thread del processo master.
            Le mutex (mutual exclusion) permettono di escludere accesso concorrente a una sezione critica del codice.
            - pthread_mutex_t: tipo di dato che rappresenta la mutex
            - queue_mutex : nome variabile che rappresenta la mutex
            - PTHREAD_MUTEX_INITIALIZER: macro che fornisce inizializzazione di default alla mutex

            Inizializzo una variabile condition.
            Consente mettere i thread in attesa e risvegliarli.
            - pthread_cond_t: tipo di dato che rappresenta la condvar
            - queue_mutex : nome variabile che rappresenta la condvar
            - PTHREAD_MUTEX_INITIALIZER: macro che fornisce inizializzazione di default alla condvar
        */
        
        /* PThreads setup */
        rc = pthread_mutex_init(&rank_worker_mutex, NULL);
        //rc |= pthread_cond_init(&queue_not_empty, NULL);
        if (rc) { 
            printf("PThread elements init error.");
            exit(-1);
        }

        // Dichiaro due variabili di tipo pthread_t che rappresentano i thread nel programma (non sto creando i thread)
        pthread_t producer_thread, sender_thread, sender_thread_2, send_sequences_thread; // thread per la generazione dei task e per l'invio dei task ai worker

        // ******************************************** INIZIO FASE 1: Il master legge il file di input ********************************************
        if(argc <= 1){
            printf("Error: No input file specified! Please specify the input file, and run again!\n");
            return 0;
        }
        //printf("(MASTER %d on %s) (thread %lu) I have read input file: %s \n", rank, hn, (unsigned long)pthread_self(), argv[1]);

        /* Opening input files in dir "argv[1]" */
        if((fp = fopen(argv[1], "r")) == NULL) {
            fprintf(stderr, "Error while opening grid file.\n");
            exit(-1);
        }

        fscanf(fp, "%d %d %d", &string_lengths[0], &string_lengths[1], &string_lengths[2]);

        printf("(MASTER %d on %s) (thread %lu) Sequence lengths: %d %d\n", rank, hn, (unsigned long)pthread_self(), string_lengths[0], string_lengths[1]);

        // alloco memoria per le due stinghe
        string_A = malloc((string_lengths[0] + 1) * sizeof(char));
        string_B = malloc((string_lengths[1] + 1) * sizeof(char));
        alphabet = malloc((string_lengths[2] + 1) * sizeof(char));

        // carico le stringhe in memoria
        fscanf(fp, "%s %s %s", string_A, string_B, alphabet);

        //printf("(MASTER %d on %s) (thread %lu) Sequences read.\n", rank, hn, (unsigned long)pthread_self());

        // chiudo il file di dati di input
        fclose(fp);
        // ******************************************** FINE FASE 1: Il master legge il file di input ********************************************

        time_start = PAPI_get_real_usec(); // inizio misurazione tempo

        // ******************************************** INIZIO FASE 2: Tiling ********************************************
        num_blocks_rows = string_lengths[0] / TILE_DIM;  // numero di blocchi verticali (sulle righe della matrice)
        num_blocks_cols = string_lengths[1]/ TILE_DIM;  // numero di blocchi orizzontali (sulle colonne)
        num_antidiagonals = num_blocks_rows + num_blocks_cols - 1;  // il numero totale di antidiagonali è dato da blocchi_righe + blocchi_colonne - 1
        num_blocks = num_blocks_rows * num_blocks_cols; // numero totale di blocchi (sulle righe e colonne della matrice)
        printf("(MASTER %d on %s) (thread %lu) Terminato il tiling:\n- numero blocchi per ogni riga: %d;\n- numero blocchi per ogni colonna: %d;\n- numero antidiagonali: %d;\n- numero totale di blocchi: %d\n", rank, hn, (unsigned long)pthread_self(), num_blocks_rows, num_blocks_cols, num_antidiagonals, num_blocks);
        // ******************************************** FINE FASE 2: Tiling ********************************************

        max_antidiagonal_length = MIN(num_blocks_rows, num_blocks_cols);
        task_queue = calloc(num_blocks, sizeof(Task)); // alloco memoria per la coda dei task (da inviare ai worker)
        pending_task_index = calloc(max_antidiagonal_length + 1, sizeof(int)); // alloco memoria per tanti interi quanti sono i blocchi sulla antidiagonale massima

        // Invio ai worker le dimensioni delle 2 sequenze
        MPI_Bcast(string_lengths, 3, MPI_INT, 0, MPI_COMM_WORLD);
        printf("(MASTER %d on %s) (thread %lu) Ho inviato l'array contenente le 3 lunghezze.\n", rank, hn, (unsigned long)pthread_self());

        // Invio le due sequenze e l'alfabeto ai worker
        MPI_Bcast(string_A, string_lengths[0] + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(string_B, string_lengths[1] + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(alphabet, string_lengths[2] + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        printf("(MASTER %d on %s) (thread %lu) Ho inviato le stringhe e l'alfabeto.\n", rank, hn, (unsigned long)pthread_self());

        //MPI_Barrier(MPI_COMM_WORLD); // Sincronizzazione tra master e worker
        //printf("(MASTER %d on %s) (thread %lu) Ho superato la barriera.\n", rank, hn, (unsigned long)pthread_self());

        // creazione dei thread
        //pthread_create(&send_sequences_thread, NULL, send_sequences, NULL); // creo thread per invio delle lunghezze delle stringhe e delle stringhe stesse ai worker
        pthread_create(&producer_thread, NULL, task_producer, NULL); // thread per la generazione dei task
        pthread_create(&sender_thread, NULL, task_sender, NULL); // thread per l'invio dei task ai worker
        pthread_create(&sender_thread_2, NULL, pending_task_sender, NULL); // thread per l'invio dei task ai worker

        // attendi entrambi
        pthread_join(producer_thread, NULL); // il main thread del master si sospende e aspetta che il thread producer termini (sincronizzazione su condizione)
        printf("Ha terminato il thread producer.\n");

        //printf("(MASTER %d on %s) (thread %lu) Threads created: producer and sender.\n", rank, hn, (unsigned long)pthread_self());

        pthread_join(sender_thread, NULL); // il main thread del master si sospende e aspetta che il thread sender termini (sincronizzazione su condizione)
        printf("Ha terminato il thread sender.\n");

        //pthread_join(send_sequences, NULL);

        // termina esplicitamente il thread sender_thread_2
        pthread_cancel(sender_thread_2);
        pthread_join(sender_thread_2, NULL);
        printf("Ha terminato il sender 2.\n");

        //printf("(MASTER %d on %s) (thread %lu) Threads terminated: producer and sender.\n", rank, hn, (unsigned long)pthread_self());

        // libero memoria allocata su heap (esegue questo codice solo il main thread del processo master)
 
    } else {    // se sei un worker (rank > 0)

        Task task;
        char stop_worker = 0;                   // flag per terminare il worker

        pthread_t *threads = malloc(NUM_WORKER_THREADS * sizeof(pthread_t));
        int thread_ids[NUM_WORKER_THREADS];            // Array per contenere gli ID dei thread

        // Inizializza il barrier per "t" thread.
        if (pthread_barrier_init(&barrier, NULL, NUM_WORKER_THREADS)) {
            fprintf(stderr, "Errore nell'inizializzazione del pthread_barrier\n");
            exit(EXIT_FAILURE);
        }
                
        // Ricevo lunghezze stringhe dal master
        MPI_Bcast(string_lengths, 3, MPI_INT, 0, MPI_COMM_WORLD); //printf("(WORKER %d on %s) (thread %lu) Ho ricevuto le 3 lunghezze.\n", rank, hn, (unsigned long)pthread_self());

        // Alloco memoria per le due stringhe
        string_A = malloc((string_lengths[0] + 1) * sizeof(char));
        string_B = malloc((string_lengths[1] + 1) * sizeof(char));
        alphabet = malloc((string_lengths[2] + 1) * sizeof(char));

        // Ricevo le due stringhe e le carico in memoria
        MPI_Bcast(string_A, string_lengths[0] + 1, MPI_CHAR, 0, MPI_COMM_WORLD); // len_A
        MPI_Bcast(string_B, string_lengths[1] + 1, MPI_CHAR, 0, MPI_COMM_WORLD); // len_B
        MPI_Bcast(alphabet, string_lengths[2] + 1, MPI_CHAR, 0, MPI_COMM_WORLD); // len_C printf("(WORKER %d on %s) (thread %lu) Ho ricevuto le 3 stringhe.\n", rank, hn, (unsigned long)pthread_self());

        // Alloco memoria per la matrice DP
        DP_matrix = malloc((TILE_DIM + 1) * sizeof(int *));
        if (DP_matrix == NULL) {
            fprintf(stderr, "Errore: allocazione della matrice delle righe fallita!\n");
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i <= TILE_DIM; i++) {
            DP_matrix[i] = calloc(TILE_DIM + 1, sizeof(int));
            if (DP_matrix[i] == NULL) {
                fprintf(stderr, "Errore: allocazione della riga %d fallita!\n", i);
                exit(EXIT_FAILURE);
            }
        }

        while(1) {
            // Ricevo il task dal MASTER
            MPI_Recv(&task, sizeof(Task), MPI_BYTE, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //printf("(WORKER %d on %s) (thread %lu) Ho ricevuto il task del blocco [%d][%d] dal master.\n", rank, hn, (unsigned long)pthread_self(), task.task_id[0], task.task_id[1]);
            
            if(task.angle == -1){ // condizione di uscita
                printf("Il worker ha ricevuto messaggio terminazione.\n");
                break;
            }

            DP_matrix[0][0] = task.angle; // inizializzo DP_Matrix[0][0] con l'angolo del task
            offset_A = task.start_index_sub_a; // offset_A
            offset_B = task.start_index_sub_b; // offset_B

            for(int k=1; k<TILE_DIM + 1; k++)
            {
                DP_matrix[0][k] = task.top_row[k - 1];
                //printf("(WORKER %d on %s) (thread %lu) Blocco [%d][%d] dal master. top_row[%d]:%d.\n", rank, hn, (unsigned long)pthread_self(), task.task_id[0], task.task_id[1], k, task.top_row[k - 1]);
                DP_matrix[k][0] = task.left_col[k - 1];
                //printf("(WORKER %d on %s) (thread %lu) Blocco [%d][%d] dal master. left_col[%d]:%d.\n", rank, hn, (unsigned long)pthread_self(), task.task_id[0], task.task_id[1], k, task.left_col[k - 1]);
            }

            //printf("(WORKER %d on %s) (thread %lu) Blocco [%d][%d] dal master. La prima lettera di A che considero è %c.\n", rank, hn, (unsigned long)pthread_self(), task.task_id[0], task.task_id[1], string_A[task.start_index_sub_a]);
            //printf("(WORKER %d on %s) (thread %lu) Blocco [%d][%d] dal master. La prima lettera di B che considero è %c.\n", rank, hn, (unsigned long)pthread_self(), task.task_id[0], task.task_id[1], string_B[task.start_index_sub_b]);

            if(!(task.task_id[0] % 100) || !(task.task_id[1] % 100)) {
                printf("(WORKER %d on %s) (thread %lu) Sono arrivato al blocco [%d][%d] dal master.\n", rank, hn, (unsigned long)pthread_self(), task.task_id[0], task.task_id[1]);
            }

            // Crea i thread
            for (int i = 0; i < NUM_WORKER_THREADS; i++) {
                thread_ids[i] = i; // Assegniamo l'ID ad ogni indice
                if (pthread_create(&threads[i], NULL, lcs_thread_func, &thread_ids[i])) {
                    fprintf(stderr, "Errore nella creazione del thread %d\n", i);
                    exit(EXIT_FAILURE);
                }
            }

            // Attendi il completamento di tutti i thread
            for (int i = 0; i < NUM_WORKER_THREADS; i++) {
                pthread_join(threads[i], NULL);
            }

            // Preparo messaggio di ripsosta
            memcpy(result.task_id, task.task_id, sizeof(task.task_id)); // ID del task

            /*for(int k=1; k<TILE_DIM + 1; k++)
            {
                printf("(WORKER %d on %s) (thread %lu) Ho preparato il risultato del blocco [%d][%d] dal master. bottom_row[%d]:%d.\n", rank, hn, (unsigned long)pthread_self(), task.task_id[0], task.task_id[1], k, result.bottom_row[k - 1]);
                printf("(WORKER %d on %s) (thread %lu) Ho preparato il risultato del blocco [%d][%d] dal master. right_col[%d]:%d.\n", rank, hn, (unsigned long)pthread_self(), task.task_id[0], task.task_id[1], k, result.right_col[k - 1]);
            }*/

            // Invio il risultato del task al MASTER
            MPI_Send(&result, sizeof(Result), MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
        }

        // Distruggi il barrier e libera le risorse allocate per i thread
        pthread_barrier_destroy(&barrier);
        free(threads);

        for (int i = 0; i <= TILE_DIM; i++) {
            free(DP_matrix[i]);
        }
        free(DP_matrix);
    }

    // libero memoria allocata su heap per ciascun processo (tutti i processi MPI eseguono questo codice)
    free(string_A);
    free(string_B);
    free(alphabet);
   
    // Stop the count (il sequente codice è eseguito da tutti i processi MPI)
    if ((rc = PAPI_stop(event_set, &num_cache_miss)) != PAPI_OK)
        handle_PAPI_error(rc, "Error in PAPI_stop().");
    printf("Rank: %d, total L2 cache misses:%lld\n", rank, num_cache_miss);

    if(!rank) {     // se sei il processo master (rank 0)
        time_stop = PAPI_get_real_usec(); // arresta misurazione tempo
        printf("(MASTER %d on %s) (thread %lu) (PAPI) Elapsed time: %lld us\n", rank, hn, (unsigned long)pthread_self(), (time_stop - time_start));
        printf("(MASTER %d on %s) (thread %lu) (PAPI) LCS: %d\n", rank, hn, (unsigned long)pthread_self(), lcs_length);
        free(task_queue);
        free(pending_task_index);
    }

    MPI_Finalize(); // termino ambiente MPI 

    return 0;
}

void *task_producer(void *args) { // funzione di thread

    MPI_Request recv_requests; // richiesta di ricezione

    Task task;
    int task_counter = 0; // Contatore globale per i task

    // OpenMP
    for (int d = 0; d < num_antidiagonals; d++) { // per ogni diagonale

        // determino intervallo di righe da considerare per l'antidiagonale corrente
        int i_start = (d < num_blocks_cols) ? 0 : d - num_blocks_cols + 1; // 0
        int i_end = (d < num_blocks_rows) ? d : num_blocks_rows - 1; // 1

        for (int i = i_start; i <= i_end; i++) { // itero l'intervallo di righe da considerare 
            int j = d - i; // siccome i + j = d 

            // Ora (i, j) è un blocco valido sulla diagonale d

            //carico_fittizio(100000); // carico fittizio per evitare che il master sia troppo veloce rispetto ai worker

            task.task_id[0] = i; task.task_id[1] = j; // task_id è un array di due interi (i, j) che rappresentano la posizione del blocco nella matrice

            task.start_index_sub_a = i * TILE_DIM; // Indice iniziale della porzione di stringa A da considerare
            task.start_index_sub_b = j * TILE_DIM; // Indice iniziale della porzione di stringa B da considerare
            INITIALIZE_TASK(task, task_counter); // Ensure task_counter is properly defined in the surrounding scope
            task_counter++; // Incrementa il contatore
            //printf("(MASTER %d on %s) (thread %lu) Task %d enqueued.\n", rank, hn, (unsigned long)pthread_self(), (i+ i_end)); // DEBUG
        }

        // Invio il primo task al worker 1 (avvio il sistema degli invii/ricezioni)
        if(!d) {
            MPI_Send(&task_queue[0], sizeof(Task), MPI_BYTE, 1, TAG_TASK, MPI_COMM_WORLD);
        }


    }

    printf("Ho finito di generare i task.\n");
    pthread_exit(NULL); // termino esplicitamente il thread corrente
}

void *send_sequences(void *args) { // funzione di thread 

    // Invio ai worker le dimensioni delle 2 sequenze
    MPI_Bcast(string_lengths, 3, MPI_INT, 0, MPI_COMM_WORLD);

    // Invio le due sequenze e l'alfabeto ai worker
    MPI_Bcast(string_A, string_lengths[0] + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(string_B, string_lengths[1] + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(alphabet, string_lengths[2] + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    pthread_exit(NULL); // termino esplicitamente il thread corrente
}

void *pending_task_sender(void *args) { // funzione di thread 

    int c = 0;

    int blocchi_inviati = 0; // contatore per i task non inizializzati

    while(1) {

        while(!pending_task_index[c]) { // finchè l'indice è zero entra
            if (stop_pending_sender) {
                goto exit;
            }
        } // fino a quando non è stato inserito un indice valido cicla a vuoto

        while(!task_queue[pending_task_index[c]].initialized) { } // fino a quando il task non è inizializzato dal produttore cicla a vuoto

        printf("----------------------------\n--------------------------------------\n-------------------------------------------\n----------------------------------------------------\n");
        pthread_mutex_lock(&rank_worker_mutex);  // Entra nella sezione critica
        MPI_Send(&task_queue[pending_task_index[c]], sizeof(Task), MPI_BYTE, rank_worker, TAG_TASK, MPI_COMM_WORLD);
        //printf("(MASTER %d on %s) (thread %lu) PTS Ho inviato il blocco (%d, %d) al worker %d.\n", rank, hn, (unsigned long)pthread_self(), task_queue[pending_task_index[c]].task_id[0], task_queue[pending_task_index[c]].task_id[1], rank_worker);
        blocchi_inviati++;
        rank_worker = (rank_worker == max_rank_worker) ? 1 : ++rank_worker; // incremento del rank del worker a cui inviare il messaggio
        pthread_mutex_unlock(&rank_worker_mutex); // Esce dalla sezione critica

        if(c == (max_antidiagonal_length)) { // se sono arrivato all'ultimo task da inviare
            printf("PST Ho raggiunto la massima dimensione dell'array.\n");
        }

        pending_task_index[c] = 0;

        c = (c == (max_antidiagonal_length)) ? 0 : ++c;

    }

    exit:

    printf("(MASTER %d on %s) (thread %lu) PTS Ho inviato %d blocchi.\n", rank, hn, (unsigned long)pthread_self(), blocchi_inviati);

    pthread_exit(NULL); // termino esplicitamente il thread corrente
}

void *task_sender(void *args) { // funzione di thread 

    Result result;

    int i;
    int j;

    int blocchi_inviati = 0; // contatore per i task non inizializzati

    int index_right;
    int index_down;
    int index_right_down;

    MPI_Status status; // Variabile per lo stato del messaggio

    int count_pending_task = 0;

    char stop_sender = 0;

    while(!stop_sender) { // per ogni worker (rank > 0) while(stop_sender == false)

        //printf("(MASTER %d on %s) (thread %lu) Mi preparo a ricevere il risultato.\n", rank, hn, (unsigned long)pthread_self());
        MPI_Recv(&result, sizeof(Result), MPI_BYTE, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);

        i = result.task_id[0]; // recupero i valori di i e j dal task ricevuto
        j = result.task_id[1];
        
        pthread_mutex_lock(&rank_worker_mutex);  // Entra nella sezione critica
        rank_worker = status.MPI_SOURCE; // recupero il rank del worker che mi ha inviato il messaggio
        //printf("(MASTER %d on %s) (thread %lu) Ho ricevuto il risultato del blocco [%d][%d] dal worker %d.\n", rank, hn, (unsigned long)pthread_self(), result.task_id[0], result.task_id[1], rank_worker);
        pthread_mutex_unlock(&rank_worker_mutex); // Esce dalla sezione critica

        //printf("(MASTER %d on %s) (thread %lu) siamo arrivati al blocco %d %d.\n", rank, hn, (unsigned long)pthread_self(), i, j);
        //printf("(MASTER %d on %s) (thread %lu) siamo arrivati al blocco %d %d.\n", rank, hn, (unsigned long)pthread_self(), i, j);

        if (j < num_blocks_cols - 1) { // controllo se non sono l'ultima colonna (j == N - 1)

            index_right = block_index(i, j + 1, num_blocks_rows, num_blocks_cols); // calcolo l'indice dell'array task_queue nel quale è presente il blocco a destra di quello ricevuto
            //printf("(MASTER %d on %s) (thread %lu) Il blocco a destra, (%d, %d), ha indice %d.\n", rank, hn, (unsigned long)pthread_self(), task_queue[index_right].task_id[0], task_queue[index_right].task_id[1], index_right);
            
            memcpy(task_queue[index_right].left_col, result.right_col, sizeof(result.right_col)); // inietto dipendenza nel task
            task_queue[index_right].left_col_ready = 1; // setto il flag a 1 (pronta)


            if(!i || (task_queue[index_right].top_row_ready && task_queue[index_right].angle_ready)) {
                if(task_queue[index_right].initialized) {
                    //printf("(MASTER %d on %s) (thread %lu) Mi preparo ad inviare il blocco (%d, %d) con indice %d.\n", rank, hn, (unsigned long)pthread_self(), task_queue[index_right].task_id[0], task_queue[index_right].task_id[1], index_right);

                    pthread_mutex_lock(&rank_worker_mutex);  // Entra nella sezione critica
                    MPI_Send(&task_queue[index_right], sizeof(Task), MPI_BYTE, rank_worker, TAG_TASK, MPI_COMM_WORLD);
                    blocchi_inviati++;
                    //printf("(MASTER %d on %s) (thread %lu) 1 Ho inviato il blocco (%d, %d) con indice %d al worker %d.\n", rank, hn, (unsigned long)pthread_self(), task_queue[index_right].task_id[0], task_queue[index_right].task_id[1], index_right, rank_worker);
                    rank_worker = (rank_worker == max_rank_worker) ? 1 : ++rank_worker; // incremento del rank del worker a cui inviare il messaggio
                    pthread_mutex_unlock(&rank_worker_mutex); // Esce dalla sezione critica

                }
                else {
                    pending_task_index[count_pending_task] = index_right;
                    //printf("(MASTER %d on %s) (thread %lu) 1 Ho inserito blocco (%d, %d) alla posizione %d di pendig_tasq_index. index_right %d.\n",  rank, hn, (unsigned long)pthread_self(), task_queue[index_right].task_id[0], task_queue[index_right].task_id[1], count_pending_task, index_right);
                    count_pending_task = (count_pending_task == (max_antidiagonal_length)) ? 0 : ++count_pending_task;
                }
            }
            
        }
    
        // Blocco sotto: (x+1, y)
        if (i < num_blocks_rows - 1) { // controllo se non sono l'ultima riga (i == M - 1)

            index_down = block_index(i + 1, j, num_blocks_rows, num_blocks_cols); // calcolo l'indice dell'array task_queue nel quale è presente il blocco sotto di quello ricevuto
            //printf("(MASTER %d on %s) (thread %lu) Il blocco sotto, (%d, %d), ha indice %d.\n", rank, hn, (unsigned long)pthread_self(), task_queue[index_down].task_id[0], task_queue[index_down].task_id[1], index_down);
            
            memcpy(task_queue[index_down].top_row, result.bottom_row, sizeof(result.bottom_row)); // inietto dipendenza nel task
            task_queue[index_down].top_row_ready = 1; // setto il flag a 1 (pronta)

            if(!j || (task_queue[index_down].left_col_ready && task_queue[index_down].angle_ready)) {
                if(task_queue[index_down].initialized) {
                    //printf("(MASTER %d on %s) (thread %lu) Mi preparo ad inviare il blocco (%d, %d) con indice %d.\n", rank, hn, (unsigned long)pthread_self(), task_queue[index_down].task_id[0], task_queue[index_down].task_id[1], index_down);
                    
                    pthread_mutex_lock(&rank_worker_mutex);  // Entra nella sezione critica
                    MPI_Send(&task_queue[index_down], sizeof(Task), MPI_BYTE, rank_worker, TAG_TASK, MPI_COMM_WORLD);
                    blocchi_inviati++;
                    //printf("(MASTER %d on %s) (thread %lu) 2 Ho inviato il blocco (%d, %d) con indice %d al worker %d.\n", rank, hn, (unsigned long)pthread_self(), task_queue[index_down].task_id[0], task_queue[index_down].task_id[1], index_down, rank_worker);
                    rank_worker = (rank_worker == max_rank_worker) ? 1 : ++rank_worker; // incremento del rank del worker a cui inviare il messaggio
                    pthread_mutex_unlock(&rank_worker_mutex); // Esce dalla sezione critica

                }
                else {
                    pending_task_index[count_pending_task] = index_down;
                    //printf("(MASTER %d on %s) (thread %lu) 2 Ho inserito blocco (%d, %d) alla posizione %d di pendig_tasq_index.\n", rank, hn, (unsigned long)pthread_self(),  task_queue[index_down].task_id[0], task_queue[index_down].task_id[1], count_pending_task);
                    count_pending_task = (count_pending_task == (max_antidiagonal_length)) ? 0 : ++count_pending_task;
                }
            }
        }

        if(j != (num_blocks_cols - 1) && i != (num_blocks_rows - 1)) { // basta che sono o all'ultima riga o all'ultima colonna e non entro nell'if
            
            index_right_down = block_index(i + 1, j + 1, num_blocks_rows, num_blocks_cols);
            //printf("(MASTER %d on %s) (thread %lu) Il blocco in basso a destra (angolo), (%d, %d), ha indice %d.\n", rank, hn, (unsigned long)pthread_self(), task_queue[index_right_down].task_id[0], task_queue[index_right_down].task_id[1], index_right_down);

            task_queue[index_right_down].angle = result.bottom_row[TILE_DIM - 1]; // inietto dipendenza nel task
            task_queue[index_right_down].angle_ready = 1; // setto il flag a 1 (pronta)

            if(task_queue[index_right_down].left_col_ready && task_queue[index_right_down].top_row_ready) {
                if(task_queue[index_right_down].initialized) {
                    //printf("(MASTER %d on %s) (thread %lu) Mi preparo ad inviare il blocco (%d, %d) con indice %d.\n", rank, hn, (unsigned long)pthread_self(), task_queue[index_right_down].task_id[0], task_queue[index_right_down].task_id[1], index_right_down);
                    
                    pthread_mutex_lock(&rank_worker_mutex);  // Entra nella sezione critica
                    MPI_Send(&task_queue[index_right_down], sizeof(Task), MPI_BYTE, rank_worker, TAG_TASK, MPI_COMM_WORLD);
                    blocchi_inviati++;
                    //printf("(MASTER %d on %s) (thread %lu) 3 Ho inviato il blocco (%d, %d) con indice %d al worker %d.\n", rank, hn, (unsigned long)pthread_self(), task_queue[index_right_down].task_id[0], task_queue[index_right_down].task_id[1], index_right_down, rank_worker);
                    rank_worker = (rank_worker == max_rank_worker) ? 1 : ++rank_worker; // incremento del rank del worker a cui inviare il messaggio
                    pthread_mutex_unlock(&rank_worker_mutex); // Esce dalla sezione critica

                }
                else {
                    pending_task_index[count_pending_task] = index_right_down;
                    //printf("(MASTER %d on %s) (thread %lu) 3 Ho inserito blocco (%d, %d) alla posizione %d di pendig_tasq_index.\n", rank, hn, (unsigned long)pthread_self(), task_queue[index_right_down].task_id[0], task_queue[index_right_down].task_id[1], count_pending_task);
                    count_pending_task = (count_pending_task == (max_antidiagonal_length)) ? 0 : ++count_pending_task;
                }
            }
        } else if (j == num_blocks_cols - 1 && i == num_blocks_rows - 1) {
            lcs_length = result.bottom_row[TILE_DIM - 1]; // salvo la lunghezza della LCS
            stop_sender = 1;
        }
        
        /*if(blocchi_inviati%1000 == 0) { // ogni 1000 invii di task stampo un messaggio
            printf("Sono il sender principale. Ho inviato %d blocchi.\n", blocchi_inviati);
        }*/
    }

    printf("(MASTER %d on %s) (thread %lu) Ho inviato %d blocchi.\n", rank, hn, (unsigned long)pthread_self(), blocchi_inviati);

    printf("Il master ha ricevuto l'ultimo blocco.\n");

    stop_pending_sender = 1;

    Task task;
    task.angle = -1; // invio un task con angolo -1 per terminare i worker
    printf("Il master ha inviato messaggio broadcast.\n");

    for (int k=1; k<=max_rank_worker; k++) {
        MPI_Send(&task, sizeof(Task), MPI_BYTE, k, TAG_TASK, MPI_COMM_WORLD); // invio il messaggio di stop a tutti i worker
    }

    pthread_exit(NULL); // termino esplicitamente il thread corrente
}

/* Function to handle PAPI errors. It prints the error message and exits the program. */
void handle_PAPI_error(int rc, char *msg) {
    char error_str[PAPI_MAX_STR_LEN];
    memset(error_str, 0, sizeof(char)*PAPI_MAX_STR_LEN);
  
    fprintf(stderr, "%s\nReturn code: %d - PAPI error message:\n", msg, rc);
    PAPI_perror(error_str); PAPI_strerror(rc);
    exit(-1);
}

// Calcola la lunghezza dell'antidiagonale d in una matrice MxN
int length_of_diagonal(int d, int M, int N) {
    int start = MAX(0, d - (N - 1));
    int end   = MIN(d, M - 1);
    if (end >= start)
        return end - start + 1;
    else
        return 0;
}

// Calcola la somma cumulativa dei blocchi nelle antidiagonali da 0 a d-1
int sum_of_lengths(int d, int M, int N) {
    int sum = 0;
    for (int k = 0; k < d; k++) {
        sum += length_of_diagonal(k, M, N);
    }
    return sum;
}

// Calcola l'indice del blocco (x,y) nell'array 
// memorizzato per antidiagonali in una matrice MxN.
int block_index(int x, int y, int M, int N) {
    int d = x + y;
    int before = sum_of_lengths(d, M, N);        // blocchi nelle antidiagonali precedenti
    int r_start = MAX(0, d - (N - 1));            // riga iniziale valida per la diagonale d
    int offset = x - r_start;                     // offset all'interno della diagonale
    return before + offset;
}

// seguono funzioni di DEBUG
void carico_fittizio(int iterazioni) {
    volatile double dummy = 0.0; // volatile per evitare ottimizzazione
    for (int i = 0; i < iterazioni; i++) {
        dummy += i * 0.0001;
        dummy = dummy / 1.000001;
    }
}

void *lcs_thread_func(void *thread_id) {

    int id = *(int *)thread_id;

    // The loop iterates over the antidiagonals of the DP matrix.
    // Each thread processes a portion of the cells in the current antidiagonal.
    // For the computation of the cells, the following observation is made:
    // - The index d (sum of indices) varies from 2 to (len_A+len_B);
    // - For each d, the valid indices are: i = max(1, d-len_B) ... min(len_A, d-1) with j = d - i
    for (int d = 2; d <= TILE_DIM + TILE_DIM; d++) {

        // Determine the range of rows for this antidiagonal
        int i_min = (d - TILE_DIM) > 1 ? (d - TILE_DIM) : 1; // max(1, d-len_B)     1
        int i_max = (d - 1) < TILE_DIM ? (d - 1) : TILE_DIM; // min(len_A, d-1)     2
        int count = i_max - i_min + 1; // Number of cells in the current antidiagonal   2

        // If the diagonal has no cells (rare case, but for safety) the thread waits
        // and moves to the next antidiagonal.
        if (count <= 0) {
            pthread_barrier_wait(&barrier);
            continue;
        }

        // Distiìribution of the cells among the threads.
        // Each thread will process a portion of the cells in the current antidiagonal.
        int chunk_size = count / NUM_WORKER_THREADS;    //0
        int remainder = count % NUM_WORKER_THREADS;     //2
        int start_index, end_index;
        if (id < remainder) { // If there are remaining elements, distribute them to the first threads
            start_index = id * (chunk_size + 1);    // 0    1
            end_index = start_index + (chunk_size + 1); // 1    2
        } else {
            start_index = id * chunk_size + remainder;
            end_index = start_index + chunk_size;
        }

        // Compute the start and end indices on which the thread will work in the antidiagonal d.
        for (int index = start_index; index < end_index; index++) {
            int i = i_min + index; // 1
            int j = d - i; // 2

            int value = 0;
            if (string_A[offset_A + i - 1] == string_B[offset_B + j - 1])
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

            //printf("(WORKER %d on %s) (thread %lu) DP_MATRIX[%d][%d] = %d\n", rank, hn, (unsigned long)pthread_self(), i, j, value); // DEBUG
            DP_MATRIX(i, j, DP_matrix) = value;

            if(i == TILE_DIM) {
                result.bottom_row[j -1] = value; // salvo il valore della cella in basso a destra (angolo) dell'ultimo blocco
            }

            if(j == TILE_DIM) {
                result.right_col[i - 1] = value; // salvo il valore della cella in basso a destra (angolo) dell'ultimo blocco
            }

        }
        // Sincronizza tutti i thread: attendi il completamento dell'antidiagonale.
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}