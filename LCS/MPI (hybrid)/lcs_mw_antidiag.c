#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <pthread.h>
#include <papi.h>
#include <time.h>
#include "omp.h"

#define NUM_WORKER_THREADS 8                // Numero di thread
#define INNER_TILE_DIM 125                  // dimensione del sotto‑blocco
#define TILE_DIM 5000                       // Dimensione della tile (blocco)
#define HOST_BUFF 256                       // Dimensione del buffer per il nome host
#define TAG_TASK 0                          // Tag dei messaggi di tipo "invio task"
#define TAG_RESULT 1                        // Tag dei messaggi di tipo "invio task"

/*
 * Struttura Task: rappresenta un'unità di lavoro per l'elaborazione di una porzione di matrice.
 * Contiene:
 * - Indici di partenza nelle stringhe A e B.
 * - Riferimenti ai bordi superiore e sinistro della matrice.
 * - L'angolo di elaborazione del task.
 * - Flag che indicano la disponibilità dei bordi.
 */
typedef struct {
    int task_id[2];                         // ID univoco del task
    int start_index_sub_a;                  // Inidce iniziale della porzione di stringa A da considerare
    int start_index_sub_b;                  // Inidce iniziale della porzione di stringa B da considerare
    int  block_h;                           // new: number of rows in *this* tile
    int  block_w;                           // new: number of cols in *this* tile
    int top_row[TILE_DIM];                  // Riga superiore (bordo superiore)
    int left_col[TILE_DIM];                 // Colonna sinistra (bordo sinistro)
    int angle;                              // Angolo (diagonale) del task
    char top_row_ready;                     // Flag per indicare se la riga superiore è pronta (1) o meno (0)
    char left_col_ready;                    // Flag per indicare se la colonna sinistra è pronta (1) o meno (0)  
    char angle_ready;                       // Flag per indicare se l'angolo è pronto (1) o meno (0) 
    char initialized;                       // Flag per indicare se il task è inizializzato (1) o meno (0)              
} Task;

/*
 * Struttura Result: rappresenta l'output generato da un task.
 * Contiene:
 * - ID del task (e del task precedente, se necessario).
 * - Bordo destro (right_col) e bordo inferiore (bottom_row) della sottosezione elaborata.
 * 
 * Questi bordi sono usati per propagare i dati necessari ai task dipendenti,
 * abilitando l'esecuzione parallela con dipendenze corrette.
 */
typedef struct{
    int task_id[2];   
    int right_col[TILE_DIM];               // Colonna destra (bordo destro)                    
    int bottom_row[TILE_DIM];              // Riga inferiore (bordo inferiore)
} Result;

/*
 * Macro: INITIALIZE_TASK
 *
 * Inizializza un task nella task_queue alla posizione indicata da task_counter.
 * Copia i campi principali dal task sorgente, tra cui ID, indici di partenza e dimensioni del blocco.
 * Inserisce una barriera di memoria (__atomic_thread_fence) per assicurare che
 * tutti i dati siano visibili ad altri thread prima di impostare il flag "initialized".
 *
 * Parametri:
 *  - task: struttura sorgente contenente i dati del task da copiare.
 *  - task_counter: indice nella task_queue dove inserire il nuovo task.
 *
 * Nota:
 *  L'uso della barriera __ATOMIC_RELEASE è cruciale in contesti concorrenti per
 *  evitare che il flag "initialized" venga letto prima che gli altri dati siano consistenti.
 */
#define INITIALIZE_TASK(task, task_counter) { \
    task_queue[task_counter].task_id[0] = task.task_id[0]; \
    task_queue[task_counter].task_id[1] = task.task_id[1]; \
    task_queue[task_counter].start_index_sub_a = task.start_index_sub_a; \
    task_queue[task_counter].start_index_sub_b = task.start_index_sub_b; \
    task_queue[task_counter].block_h        = task.block_h;      \
    task_queue[task_counter].block_w        = task.block_w;      \
    __atomic_thread_fence(__ATOMIC_RELEASE); \
    task_queue[task_counter].initialized = 1; \
}
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DP_MATRIX(i,j) DP_matrix[i][j]

/* =========================================================================
 * Prototipi di funzione
 * =========================================================================
 */
void handle_PAPI_error(int, char*);
void carico_fittizio(int iterazioni);
Task create_task(int i, int j, int dim, char *A, char *B, int len_a, int len_b);
Result lcs_yang_v2(int **DP, int **P, char *A, char *B, char *C, int m, int n, int u, int offset, int flag);

/* =========================================================================
 * Prototipi delle funzioni eseguibili dai thread
 * =========================================================================
 *
 * Queste funzioni rappresentano i punti di ingresso per i thread. 
 */
void *task_producer(void *args);
void *task_sender(void *args);
void *pending_task_sender(void *args);
void *send_sequences(void *args);
void lcs_block_wavefront(Task *received_task_ptr);

/*
 * =========================================================================
 * AREA DATI GLOBALE - Variabili globali utilizzate dal processo master
 * =========================================================================
 * Questa sezione raccoglie tutte le variabili globali accessibili dal 
 * processo master. Esse vengono utilizzate per il coordinamento dei task, 
 * la gestione della concorrenza e la comunicazione tra thread e processi 
 * (es. tramite MPI e pthreads).
 * =========================================================================
 */
int finished_generating;                    // Flag per indicare se il master ha finito di generare task
int num_blocks;                             // Number of blocks in the matrix (after tailing)
int num_antidiagonals;                      // Numero totale di antidiagonali
int *pending_task_index;                    // Indice del task non inizializzato (da inviare al worker)
int rank_worker;                            // Rank del worker a cui devo inviare il messaggio
int max_rank_worker;                        // Massimo rank tra i worker a cui posso inviare un messaggio
int producer_start = 0;                     // Flag per indicare se il produttore ha iniziato a produrre task
int lcs_length = 0;                         // Lunghezza della LCS (longest common subsequence) 
int num_processes;                          // Number of processes in MPI_COMM_WORLD 
char stop_pending_sender;
pthread_mutex_t rank_worker_mutex;          // Mutex per proteggere l'accesso alla variabile rank_worker
Task *task_queue;                           // Coda dei task (da inviare ai worker)

/*
 * =========================================================================
 * AREA DATI GLOBALE - Variabili globali per il processo master e i worker
 * =========================================================================
 * Questa sezione contiene tutte le variabili globali utilizzate sia dal 
 * processo master che dai worker.
 * =========================================================================
 */
int rank;                                   // Current process identifier
int offset_A, offset_B;
int max_antidiagonal_length;                // Lunghezza in blocchi della massima antidiagonale
int num_blocks_rows, num_blocks_cols;       // Number of blocks in rows and columns
int string_lengths[2];                      // Array to store the lengths of the two strings
char *string_A, *string_B;                  // Pointers to the two strings and alphabet
char hn[HOST_BUFF];                         // Hostname of the machine

/*
 * =========================================================================
 * AREA DATI GLOBALE - Variabili globali per i worker
 * =========================================================================
 * Questa sezione contiene le variabili globali specifiche per i worker 
 * all'interno del sistema parallelo. Queste variabili sono utilizzate per 
 * gestire lo stato e il comportamento dei worker, inclusi la sincronizzazione 
 * tra di essi e il coordinamento con il processo master.
 * =========================================================================
 */
int *DP_data;
int **DP_matrix;
int inner_block_num;                                     // Numero sotto‑blocchi che occorrono per coprire TILE_DIM
Result result;                              // Risultato del task (da inviare al master)
MPI_Request *send_requests;
//Task received_task;

int main(int argc, char *argv[])
{
    /*
        Ogni processo MPI avrà queste variabili (ma attenzione non sono globali, sono private, locali a ciascun processo).
        
        Inoltre, anche ciascun thread (sottoinsieme del processo) di ogni processo vedrà qieste variabili.
        A livello di thread queste variabili sono globali (condivisione di memoria).
    */ 

    int provided;                           // MPI thread level supported
    int rc;                                 // Return code used in error handling
    int event_set = PAPI_NULL;              // Group of hardware events for PAPI library
    long_long time_start, time_stop;        // To measure execution time
    long_long num_cache_miss;               // To measure number of cache misses

    // Start MPI setup (ogni processo MPI esegue questo codice)
    if((rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided)) != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init error. Return code: %d\n", rc);
        exit(EXIT_FAILURE);
    } 
    if(provided < MPI_THREAD_SERIALIZED) {
        fprintf(stderr, "Minimum MPI threading level requested: %d (provided: %d)\n", MPI_THREAD_SERIALIZED, provided);
        exit(EXIT_FAILURE);
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

    if (!rank) {    // Se sei il processo master (rank 0)

        /*
            Quì dentro ci entra solo il processo master (rank 0).
            Le variabili dichiarate in questa sezione di codice sono visibili a tutti i thread del processo master.
        */

        FILE *fp;                                   // File pointer for input file
        stop_pending_sender = 0;                    // Flag per terminare il thread pending_task_sender
        
        // PThreads setup
        rc = pthread_mutex_init(&rank_worker_mutex, NULL);
        if (rc) { 
            printf("PThread elements init error.\n");
            exit(-1);
        }

        pthread_t producer_thread, sender_thread, sender_thread_2, send_sequences_thread; // Dichiaro variabili di tipo pthread_t che rappresentano i thread nel programma (non sto creando i thread)

        // ========================== INIZIO FASE 1: Il master legge il file di input ==========================
        if(argc <= 1){
            fprintf(stderr, "Error: No input file specified! Please specify the input file, and run again!\n");
            return 0;
        }
        if((fp = fopen(argv[1], "r")) == NULL) { // Opening input files in dir "argv[1]"
            fprintf(stderr, "Error while opening file.");
            exit(-1);
        }
        fscanf(fp, "%d %d", &string_lengths[0], &string_lengths[1]);
        printf("(MASTER %d on %s) (thread %lu) Sequence lengths: %d %d\n", rank, hn, (unsigned long)pthread_self(), string_lengths[0], string_lengths[1]);
        string_A = malloc((string_lengths[0] + 1) * sizeof(char));
        string_B = malloc((string_lengths[1] + 1) * sizeof(char));
        fscanf(fp, "%s %s", string_A, string_B); // Carico le stringhe in memoria
        fclose(fp); // chiudo il file di dati di input
        // ========================== FINE FASE 1: Il master legge il file di input ==========================

        time_start = PAPI_get_real_usec(); // Inizio misurazione tempo

        // ========================== INIZIO FASE 2: Tiling e setup relativi ==========================
        num_blocks_rows = (string_lengths[0] + TILE_DIM - 1) / TILE_DIM; // numero di blocchi verticali (sulle righe della matrice)
        num_blocks_cols = (string_lengths[1] + TILE_DIM - 1) / TILE_DIM; // numero di blocchi orizzontali (sulle colonne)
        num_antidiagonals = num_blocks_rows + num_blocks_cols - 1; // il numero totale di antidiagonali è dato da blocchi_righe + blocchi_colonne - 1
        num_blocks = num_blocks_rows * num_blocks_cols; // numero totale di blocchi (sulle righe e colonne della matrice)
        //printf("(MASTER %d on %s) (thread %lu) Terminato il tiling:\n- numero blocchi per ogni riga: %d;\n- numero blocchi per ogni colonna: %d;\n- numero antidiagonali: %d;\n- numero totale di blocchi: %d\n", rank, hn, (unsigned long)pthread_self(), num_blocks_rows, num_blocks_cols, num_antidiagonals, num_blocks);
        max_antidiagonal_length = MIN(num_blocks_rows, num_blocks_cols);
        task_queue = calloc(num_blocks, sizeof(Task)); // alloco memoria per la coda dei task (da inviare ai worker)
        pending_task_index = calloc(max_antidiagonal_length + 1, sizeof(int)); // alloco memoria per tanti interi quanti sono i blocchi sulla antidiagonale massima
        // ========================== FINE FASE 2: Tiling  e setup relativi ==========================

        // Invio ai worker le dimensioni delle 2 sequenze e le sequenze stesse
        MPI_Bcast(string_lengths, 2, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(string_A, string_lengths[0] + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(string_B, string_lengths[1] + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Creazione dei thread
        rc = pthread_create(&producer_thread, NULL, task_producer, NULL); // Thread per la generazione dei task
        rc |= pthread_create(&sender_thread, NULL, task_sender, NULL); // Thread per l'invio dei task ai worker
        rc |= pthread_create(&sender_thread_2, NULL, pending_task_sender, NULL); // Thread per l'invio dei task ai worker
        if (rc) { 
            fprintf(stderr, "PThread creation error. Return code: %d\n", rc);
            exit(EXIT_FAILURE);
        }

        pthread_join(producer_thread, NULL); // Il main thread del master si sospende e aspetta che il thread producer termini (sincronizzazione su condizione)
        pthread_join(sender_thread, NULL); // Il main thread del master si sospende e aspetta che il thread sender termini (sincronizzazione su condizione)
        pthread_cancel(sender_thread_2); // Termina esplicitamente il thread sender_thread_2
        pthread_join(sender_thread_2, NULL);

    } else { // Se sei un worker (rank > 0)
        
        // Ricevo le lunghezze e poi alloco i buffer per le stringhe
        MPI_Bcast(string_lengths, 2, MPI_INT, 0, MPI_COMM_WORLD);
    
        // Alloco memoria per le due stringhe
        string_A = malloc((string_lengths[0] + 1) * sizeof(char));
        string_B = malloc((string_lengths[1] + 1) * sizeof(char));
        if (!string_A || !string_B) {
            perror("Alloc string buffers");
            exit(EXIT_FAILURE);
        }
    
        // Ricevo effettivamente le stringhe
        MPI_Bcast(string_A, string_lengths[0] + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(string_B, string_lengths[1] + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Calcolo quanti sotto‑blocchi occorrono per coprire TILE_DIM
        inner_block_num = (TILE_DIM + INNER_TILE_DIM - 1) / INNER_TILE_DIM; // numero blocchi per dimensione

        // Alloca la memoria contigua per la DP
        size_t size = (TILE_DIM + 1) * (TILE_DIM + 1) * sizeof(int);
        if (posix_memalign((void**)&DP_data, 64, size)) {
            perror("Posix_memalign");
            exit(EXIT_FAILURE);
        }

        // Costruisci un array di puntatori per l’indicizzazione DP_matrix[i][j]
        DP_matrix = malloc((TILE_DIM + 1) * sizeof(int*));
        if (!DP_matrix) {
            perror("Alloc DP_matrix pointers");
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < TILE_DIM + 1; ++i) {
            DP_matrix[i] = DP_data + i * (TILE_DIM + 1);
        }

        // Inizializza la matrice a zero con first-touch
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < TILE_DIM + 1; ++i) {
            memset(DP_matrix[i], 0, (TILE_DIM + 1) * sizeof(int));
        }
    
        // Ciclo principale: ricevo un Task, lo processa con OpenMP, invio Result
        Task received_task;
        MPI_Status status;
        while (1) {
            MPI_Recv(&received_task, sizeof(Task), MPI_BYTE,
                     0, TAG_TASK, MPI_COMM_WORLD, &status);
            if (received_task.angle == -1)
                break;
            lcs_block_wavefront(&received_task);
        }
        
        free(DP_data);
        free(DP_matrix); 
    }

    free(string_A);
    free(string_B);
   
    // Stop the count (il sequente codice è eseguito da tutti i processi MPI)
    if ((rc = PAPI_stop(event_set, &num_cache_miss)) != PAPI_OK)
        handle_PAPI_error(rc, "Error in PAPI_stop().");
    printf("Rank: %d, total L2 cache misses:%lld\n", rank, num_cache_miss);

    if(!rank) { // Se sei il processo master (rank 0)
        time_stop = PAPI_get_real_usec(); // arresta misurazione tempo
        printf("(MASTER %d on %s) (thread %lu) (PAPI) Elapsed time: %lld us\n", rank, hn, (unsigned long)pthread_self(), (time_stop - time_start));
        printf("(MASTER %d on %s) (thread %lu) (PAPI) LCS: %d\n", rank, hn, (unsigned long)pthread_self(), lcs_length);
        free(task_queue);
        free(pending_task_index);
    }

    MPI_Finalize(); // termino ambiente MPI 

    return 0;
}

void *task_producer(void *args) { // Funzione di thread

    MPI_Request send_request; // Richiesta di ricezione

    Task task;
    int task_counter = 0; // Contatore globale per i task

    // OpenMP
    for (int d = 0; d < num_antidiagonals; d++) { // Per ogni diagonale

        // Determino intervallo di righe da considerare per l'antidiagonale corrente
        int i_start = (d < num_blocks_cols) ? 0 : d - num_blocks_cols + 1;
        int i_end = (d < num_blocks_rows) ? d : num_blocks_rows - 1;

        for (int i = i_start; i <= i_end; i++) { // Itero l'intervallo di righe da considerare 
            int j = d - i; // Siccome i + j = d 

            // Ora (i, j) è un blocco valido sulla diagonale d

            task.task_id[0] = i; task.task_id[1] = j; // task_id è un array di due interi (i, j) che rappresentano la posizione del blocco nella matrice
            task.start_index_sub_a = i * TILE_DIM; // Indice iniziale della porzione di stringa A da considerare
            task.start_index_sub_b = j * TILE_DIM; // Indice iniziale della porzione di stringa B da considerare
            // how many rows/cols this tile really has:
            task.block_h = MIN( TILE_DIM, string_lengths[0] - task.start_index_sub_a );
            task.block_w = MIN( TILE_DIM, string_lengths[1] - task.start_index_sub_b );
            INITIALIZE_TASK(task, task_counter); // Ensure task_counter is properly defined in the surrounding scope
            task_counter++; // Incrementa il contatore
        }

        // Invio il primo task al worker 1 (avvio il sistema degli invii/ricezioni)
        if(!d) {
            MPI_Isend(&task_queue[0], sizeof(Task), MPI_BYTE, 1, TAG_TASK, MPI_COMM_WORLD, &send_request);
        }
    }

    MPI_Wait(&send_request, MPI_STATUS_IGNORE);

    pthread_exit(NULL); // Termino esplicitamente il thread corrente
}

void *send_sequences(void *args) { // Funzione di thread 

    // Invio ai worker le dimensioni delle 2 sequenze
    MPI_Bcast(string_lengths, 3, MPI_INT, 0, MPI_COMM_WORLD);

    // Invio le due sequenze e l'alfabeto ai worker
    MPI_Bcast(string_A, string_lengths[0] + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(string_B, string_lengths[1] + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    pthread_exit(NULL); // Termino esplicitamente il thread corrente
}

void *pending_task_sender(void *args) { // Funzione di thread 

    int c = 0;

    while(1) {

        while(!pending_task_index[c]) { // Finchè l'indice è zero entra
            if (stop_pending_sender) {
                goto exit;
            }
        } // Fino a quando non è stato inserito un indice valido cicla a vuoto

        while(!task_queue[pending_task_index[c]].initialized) { } // Fino a quando il task non è inizializzato dal produttore cicla a vuoto

        pthread_mutex_lock(&rank_worker_mutex);  // Entra nella sezione critica
        MPI_Send(&task_queue[pending_task_index[c]], sizeof(Task), MPI_BYTE, rank_worker, TAG_TASK, MPI_COMM_WORLD);
        rank_worker = (rank_worker == max_rank_worker) ? 1 : ++rank_worker; // Incremento del rank del worker a cui inviare il messaggio
        pthread_mutex_unlock(&rank_worker_mutex); // Esce dalla sezione critica

        pending_task_index[c] = 0;

        c = (c == (max_antidiagonal_length)) ? 0 : ++c;

    }

    exit:
    pthread_exit(NULL); // Termino esplicitamente il thread corrente
}

void *task_sender(void *args) { // Funzione di thread 

    Result result;

    int i;
    int j;

    int index_right;
    int index_down;
    int index_right_down;

    MPI_Status status;                      // Variabile per lo stato del messaggio
    MPI_Request send_requests[max_antidiagonal_length];

    int count_requests = 0;                 // Contatore per le richieste di invio
    int count_pending_task = 0;

    char stop_sender = 0;

    while(!stop_sender) { // Per ogni worker (rank > 0) while(stop_sender == false)

        MPI_Recv(&result, sizeof(Result), MPI_BYTE, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);

        i = result.task_id[0]; // Recupero i valori di i e j dal task ricevuto
        j = result.task_id[1];
        
        pthread_mutex_lock(&rank_worker_mutex); // Entra nella sezione critica
        rank_worker = status.MPI_SOURCE; // Recupero il rank del worker che mi ha inviato il messaggio
        pthread_mutex_unlock(&rank_worker_mutex); // Esce dalla sezione critica

        if (j < num_blocks_cols - 1) { // Controllo se non sono l'ultima colonna (j == N - 1)

            index_right = block_index(i, j + 1, num_blocks_rows, num_blocks_cols); // Calcolo l'indice dell'array task_queue nel quale è presente il blocco a destra di quello ricevuto
            
            memcpy(task_queue[index_right].left_col, result.right_col, sizeof(result.right_col)); // Inietto dipendenza nel task
            task_queue[index_right].left_col_ready = 1; // Setto il flag a 1 (pronta)


            if(!i || (task_queue[index_right].top_row_ready && task_queue[index_right].angle_ready)) {
                if(task_queue[index_right].initialized) {

                    pthread_mutex_lock(&rank_worker_mutex);  // Entra nella sezione critica
                    rank_worker = (rank_worker == max_rank_worker) ? 1 : ++rank_worker; // Incremento del rank del worker a cui inviare il messaggio
                    MPI_Isend(&task_queue[index_right], sizeof(Task), MPI_BYTE, rank_worker, TAG_TASK, MPI_COMM_WORLD, &send_requests[count_requests]);
                    pthread_mutex_unlock(&rank_worker_mutex); // Esce dalla sezione critica
                    count_requests = (count_requests == (max_antidiagonal_length - 1)) ? 0 : ++count_requests; // Incremento del contatore delle richieste di invio

                }
                else {
                    pending_task_index[count_pending_task] = index_right;
                    count_pending_task = (count_pending_task == (max_antidiagonal_length)) ? 0 : ++count_pending_task;
                }
            }
        }
    
        // Blocco sotto: (x+1, y)
        if (i < num_blocks_rows - 1) { // Controllo se non sono l'ultima riga (i == M - 1)

            index_down = block_index(i + 1, j, num_blocks_rows, num_blocks_cols); // Calcolo l'indice dell'array task_queue nel quale è presente il blocco sotto di quello ricevuto
            
            memcpy(task_queue[index_down].top_row, result.bottom_row, sizeof(result.bottom_row)); // Inietto dipendenza nel task
            task_queue[index_down].top_row_ready = 1; // Setto il flag a 1 (pronta)

            if(!j || (task_queue[index_down].left_col_ready && task_queue[index_down].angle_ready)) {
                if(task_queue[index_down].initialized) {
                    pthread_mutex_lock(&rank_worker_mutex);  // Entra nella sezione critica
                    rank_worker = (rank_worker == max_rank_worker) ? 1 : ++rank_worker; // Incremento del rank del worker a cui inviare il messaggio
                    MPI_Isend(&task_queue[index_down], sizeof(Task), MPI_BYTE, rank_worker, TAG_TASK, MPI_COMM_WORLD, &send_requests[count_requests]);
                    pthread_mutex_unlock(&rank_worker_mutex); // Esce dalla sezione critica
                    count_requests = (count_requests == (max_antidiagonal_length - 1)) ? 0 : ++count_requests; // Incremento del contatore delle richieste di invio
                }
                else {
                    pending_task_index[count_pending_task] = index_down;
                    count_pending_task = (count_pending_task == (max_antidiagonal_length)) ? 0 : ++count_pending_task;
                }
            }
        }

        if(j != (num_blocks_cols - 1) && i != (num_blocks_rows - 1)) { // Basta che sono o all'ultima riga o all'ultima colonna e non entro nell'if
            index_right_down = block_index(i + 1, j + 1, num_blocks_rows, num_blocks_cols);
            int parent_w = MIN(TILE_DIM, string_lengths[1] - j * TILE_DIM);
            task_queue[index_right_down].angle = result.bottom_row[parent_w - 1];
            task_queue[index_right_down].angle_ready = 1; // Setto il flag a 1 (pronta)

        } else if (j == num_blocks_cols - 1 && i == num_blocks_rows - 1) {
            int W_last = string_lengths[1] - j * TILE_DIM;
            lcs_length = result.bottom_row[W_last - 1];
            stop_sender = 1;
        }

    }

    stop_pending_sender = 1;

    Task task;
    task.angle = -1; // Invio un task con angolo -1 per terminare i worker
    for (int k=1; k<=max_rank_worker; k++) {
        MPI_Send(&task, sizeof(Task), MPI_BYTE, k, TAG_TASK, MPI_COMM_WORLD); // Invio il messaggio di stop a tutti i worker
    }

    pthread_exit(NULL); // Termino esplicitamente il thread corrente
}

void lcs_block_wavefront(Task *t) {

    int i0, i1;
    int j0, j1;
    int up, left;
    int gi, gj;
    int bi, bj;
    int i, j;

    // Tile dimensione tile_height * tile_width
    int tile_height = t->block_h; // altezza del tile ricevuto
    int tile_width = t->block_w; // larghezza del tile ricevuto

    DP_MATRIX(0,0) = t->angle;
    for (int k = 1; k <= tile_width; ++k)
        DP_MATRIX(0, k) = t->top_row[k-1];
    for (int k = 1; k <= tile_height; ++k)
        DP_MATRIX(k, 0) = t->left_col[k-1];

    // Il tile di dimensione tile_height×tile_width viene ulteriormente diviso in blocchi di lato INNER_TILE_DIM
    int block_count_vertical = (tile_height + INNER_TILE_DIM - 1) / INNER_TILE_DIM; // Numero di blocchi in altezza rispetto al tile ricevuto
    int block_count_horizontal = (tile_width + INNER_TILE_DIM - 1) / INNER_TILE_DIM; // Numero di blocchi in larghezza rispetto al tile ricevuto
    int antidiagonal_count = block_count_vertical + block_count_horizontal - 1; //Quante antidiagonali ha il blocco che dobbiamo calcolare ed stato diviso ulteriormente in blocchi

    for (int d = 0; d < antidiagonal_count; ++d) { // Itero le antidiagonali di blocchi

        // Per ciascuna antidiagonale d si calcolano i blocchi (bi,bj) che giacciono su quella antidiagonale di blocchi

        /*
        *Calcolo, per la antidiagonale di blocchi “d”, quali sono i limiti inferiori e superiori dell’indice di riga del blocco (bi)
        */ 
        int min_row_index = MAX(0, d - (block_count_horizontal - 1));
        int max_row_index = MIN(d, block_count_vertical - 1);

        /*
         * Inizio della regione parallela OpenMP:
         * - Per il blocco (0,0) ci sarà un solo thread, min_row_index = max_row_index = 0
        */
        //omp_set_num_threads(NUM_WORKER_THREADS); // Setto il numero di thread da usare
        #pragma omp parallel for schedule(dynamic, 1) private(i0, i1, j0, j1, up, left, gi, gj, i, j, bi)
        for (int bi = min_row_index; bi <= max_row_index; ++bi) { // Itero le righe dell'antidiagonale 
            int bj = d - bi;
            
            // Blocco (0, 1) -> bi = 0, bj = d - 0 = 1 - 0 = 1

            // Coordinate locali in [1..tile_height]×[1..tile_width]
            int i0 = bi * INNER_TILE_DIM + 1; // Indice di riga inziiale
            int i1 = MIN((bi + 1) * INNER_TILE_DIM, tile_height); // Indice di riga finale

            int j0 = bj * INNER_TILE_DIM + 1; // Indice di colonna iniziale
            int j1 = MIN((bj + 1) * INNER_TILE_DIM, tile_width); // Indice di colonna finale

            //          j0                        j1
            //          |                         |
            //          v                         v
            // i0 ->   +-----------------------------+   
            //         |                             |   
            //         |                             |   
            //         |        TILE BLOCK           |   --> dimensione: INNER_TILE_DIM × INNER_TILE_DIM
            //         |                             |   
            //         |                             |   
            // i1 ->   +-----------------------------+
            //
            //         ^                             ^
            //         |                             |
            //    bi * INNER_TILE_DIM        MIN((bi+1)*INNER_TILE_DIM, tile_height)

            for (int i = i0; i <= i1; ++i) { // Itero le righe del blochetto interno (INNER_TILE_DIM)
                for (int j = j0; j <= j1; ++j) { // Itero le colonne del blochetto interno (INNER_TILE_DIM)
                    up   = DP_MATRIX(i-1, j);
                    left = DP_MATRIX(i, j-1);
                    gi   = t->start_index_sub_a + i - 1; // Indice della stringa A da considerare
                    gj   = t->start_index_sub_b + j - 1; // Indice della stringa B da considerare
                    DP_MATRIX(i, j) = (string_A[gi] == string_B[gj]) // Se string_A[*] == string_B[*] allora
                        ? DP_MATRIX(i-1, j-1) + 1 // Caso in cui sono uguali (diagonale + 1)
                        : MAX(up, left); // Altrimenti prendo il massimo tra su e sinistra (max(up, left))
                }
            }
       } // Barriera implicita di OpenMP
    }

    // Il thread master MPI prepara e invia il risultato del tile
    Result res;
    res.task_id[0] = t->task_id[0];
    res.task_id[1] = t->task_id[1];

    // Send only the actual borders of this tile
    for (int i = 1; i <= tile_height; ++i)
        res.right_col[i-1] = DP_MATRIX(i, tile_width);
    for (int j = 1; j <= tile_width; ++j)
        res.bottom_row[j-1] = DP_MATRIX(tile_height, j);
    
    MPI_Send(&res, sizeof(res), MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
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

// Calcola l'indice del blocco (x,y) nell'array (memorizzato per antidiagonali in una matrice MxN)
int block_index(int x, int y, int M, int N) {
    int d = x + y;
    int before = sum_of_lengths(d, M, N); // Blocchi nelle antidiagonali precedenti
    int r_start = MAX(0, d - (N - 1)); //Riga iniziale valida per la diagonale d
    int offset = x - r_start; // Offset all'interno della diagonale
    return before + offset;
}

// Function to handle PAPI errors. It prints the error message and exits the program.
void handle_PAPI_error(int rc, char *msg) {
    char error_str[PAPI_MAX_STR_LEN];
    memset(error_str, 0, sizeof(char)*PAPI_MAX_STR_LEN);
  
    fprintf(stderr, "%s\nReturn code: %d - PAPI error message:\n", msg, rc);
    PAPI_perror(error_str); PAPI_strerror(rc);
    exit(EXIT_FAILURE);
}