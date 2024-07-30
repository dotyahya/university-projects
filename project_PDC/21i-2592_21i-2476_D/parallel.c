/*
    Authors: Muhammad Yahya, Muhammad Omer Nasir
    Roll#: 21i-2592, 21i-2476
    Code File: Top K Shortest Path Problem with MPI and OpenMP

    Description: This code implements parallel algorithms using MPI and OpenMP to 
                find the top K shortest paths between pairs of nodes in a graph. 
                It reads the graph from a CSV file, distributes the computation of finding shortest paths 
                across MPI processes, and utilizes OpenMP for parallelization within each process. 
                The program then calculates the average time taken and prints it out. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

#define INF 1000000000

struct Edge {
    char *source;
    char *target;
    int weight;
};

struct Node {
    char *name;
    int index;
};

struct QueueNode {
    int distance;
    int vertex;
};

struct NodePair {
    int srcIndex;
    int destIndex;
};

int compare(const void *a, const void *b) {
    const struct QueueNode *nodeA = (const struct QueueNode *)a;
    const struct QueueNode *nodeB = (const struct QueueNode *)b;
    return (nodeA->distance - nodeB->distance);
}

int findNodeIndex(struct Node *nodes, int node_count, char *name) {
    for (int i = 0; i < node_count; i++) {
        if (strcmp(nodes[i].name, name) == 0) {
            return i;
        }
    }
    return -1;  // node not present
}

void readGraph(char *filename, int *N, int *M, struct Edge **edges, struct Node **nodes) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file\n");
        exit(EXIT_FAILURE);
    }

    char header[256];
    if (fgets(header, sizeof(header), file) == NULL) {
        printf("Error reading header\n");
        exit(EXIT_FAILURE);
    }

    *M = 0;
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        (*M)++;
    }

    *edges = (struct Edge *)malloc(*M * sizeof(struct Edge));
    if (*edges == NULL) {
        printf("Error allocating memory for edges\n");
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_SET);
    fgets(header, sizeof(header), file);  // skip headers

    int node_count = 0, max_nodes = 10;
    *nodes = malloc(max_nodes * sizeof(struct Node));
    int edge_index = 0;

    while (fgets(line, sizeof(line), file)) {
        char *source = strtok(line, ",");
        char *target = strtok(NULL, ",");
        char *weight_str = strtok(NULL, ",");

        (*edges)[edge_index].source = strdup(source);
        (*edges)[edge_index].target = strdup(target);
        (*edges)[edge_index].weight = atoi(weight_str);
        edge_index++;

        int srcIndex = findNodeIndex(*nodes, node_count, source);
        if (srcIndex == -1) {
            if (node_count >= max_nodes) {
                max_nodes *= 2;
                *nodes = realloc(*nodes, max_nodes * sizeof(struct Node));
            }
            (*nodes)[node_count].name = strdup(source);
            (*nodes)[node_count].index = node_count;
            srcIndex = node_count++;
        }

        int destIndex = findNodeIndex(*nodes, node_count, target);
        if (destIndex == -1) {
            if (node_count >= max_nodes) {
                max_nodes *= 2;
                *nodes = realloc(*nodes, max_nodes * sizeof(struct Node));
            }
            (*nodes)[node_count].name = strdup(target);
            (*nodes)[node_count].index = node_count;
            destIndex = node_count++;
        }
    }
    *N = node_count;
    fclose(file);
}

void freeResources(struct Edge *edges, struct Node *nodes, int M, int N) {
    for (int i = 0; i < M; i++) {
        free(edges[i].source);
        free(edges[i].target);
    }
    free(edges);

    for (int i = 0; i < N; i++) {
        free(nodes[i].name);
    }
    free(nodes);
}

void findKShortest(struct Node *nodes, struct Edge *edges, int N, int M, int k, int sourceIndex, int destinationIndex, int rank) {
    int *dis = (int *)malloc(N * k * sizeof(int));
    if (dis == NULL) {
        printf("Memory allocation failed.\n");
        return;
    }

    #pragma omp parallel for
    for (int i = 0; i < N * k; i++) {
        dis[i] = INF;
    }
    dis[sourceIndex * k] = 0;

    struct QueueNode *pq = (struct QueueNode *)malloc(N * k * sizeof(struct QueueNode));
    if (pq == NULL) {
        printf("Memory allocation failed.\n");
        free(dis);
        return;
    }

    int pqSize = 0;
    pq[pqSize++] = (struct QueueNode){0, sourceIndex};

    while (pqSize > 0) {
        struct QueueNode top = pq[--pqSize];

        int u = top.vertex;
        int d = top.distance;
        if (dis[u * k + k - 1] < d) continue;

        // parallelizing loops using OpenMP
        #pragma omp parallel for
        for (int j = 0; j < M; j++) {
            if (strcmp(nodes[u].name, edges[j].source) == 0) {
                int dest = findNodeIndex(nodes, N, edges[j].target);
                int cost = edges[j].weight;
                if (d + cost < dis[dest * k + k - 1]) {
                    dis[dest * k + k - 1] = d + cost;
                    pq[pqSize++] = (struct QueueNode){d + cost, dest};
                }
            }
        }

        qsort(pq, pqSize, sizeof(struct QueueNode), compare);
    }

    printf("\nProcess %d is finding the shortest paths from node %s to node %s\n", rank, nodes[sourceIndex].name, nodes[destinationIndex].name);
    printf("Paths found by Process %d:\n", rank);
    for (int i = 0; i < k; i++) {
        if (dis[destinationIndex * k + i] == INF)
            printf("Path %d: Distance = INF\n", i + 1);
        else
            printf("Path %d: Distance = %d\n", i + 1, dis[destinationIndex * k + i]);
    }

    free(dis);
    free(pq);
}

// driver code
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            printf("Usage format: %s <filename.csv>\n", argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    int N, M, K;
    if (rank == 0) {
        printf("\nEnter the number of shortest paths (K): ");
        fflush(stdout);
        scanf("%d", &K);
        if (K <= 0) {
            printf("K must be a positive integer.\n");
            fflush(stdout);
            MPI_Finalize();
            return 0;
        }
    }

    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    struct Edge *edges;
    struct Node *nodes;

    readGraph(argv[1], &N, &M, &edges, &nodes);

    struct NodePair pairs[10];
    double times[10]; // time array for each proc

    for (int i = 0; i < 10; i++) {
        pairs[i].srcIndex = rand() % N;
        do {
            pairs[i].destIndex = rand() % N;
        } while (pairs[i].srcIndex == pairs[i].destIndex);
    }

    // determining the num of node pairs each process has to handle
    int num_pairs_per_process = ceil(10.0 / size);
    int start_index = rank * num_pairs_per_process;
    int end_index = fmin((rank + 1) * num_pairs_per_process, 10);

    double local_total_time = 0.0;
    for (int i = start_index; i < end_index; i++) {
        clock_t start, end;
        double cpu_time_used;

        start = clock();
        findKShortest(nodes, edges, N, M, K, pairs[i].srcIndex, pairs[i].destIndex, rank);
        end = clock();

        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        times[i] = cpu_time_used;
        local_total_time += cpu_time_used;

        printf("\nTime taken by process %d for nodes %d to %d: %f seconds\n", rank, pairs[i].srcIndex + 1, pairs[i].destIndex + 1, cpu_time_used);
    }

    // ensuring all processes reach this point before average is calc and displayded
    MPI_Barrier(MPI_COMM_WORLD);

    double global_total_time;
    MPI_Reduce(&local_total_time, &global_total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double average_time;
    if (rank == 0) {
        average_time = global_total_time / 10;
    }

    MPI_Bcast(&average_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nAverage time taken: %f seconds\n", average_time);
    }

    freeResources(edges, nodes, M, N);

    MPI_Finalize();
    return 0;
}

