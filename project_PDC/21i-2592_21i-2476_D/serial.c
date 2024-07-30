/*
    Authors: Muhammad Yahya, Muhammad Omer Nasir
    Roll#: 21i-2592, 21i-2476
    Code File: Top K Shortest Path Problem with MPI and OpenMP

    Description: This C program finds the top K shortest paths between randomly 
                selected pairs of nodes in a graph. It reads the graph from a CSV file, 
                where each line represents an edge with source, target, and weight. 
                The graph is represented using adjacency lists. It then generates 10 random pairs 
                of source and destination nodes and calculates the K shortest paths between each pair 
                using Dijkstra's algorithm. The program measures the execution time for each pair and 
                calculates the average time taken for all pairs. 
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

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

int findNodeIndex(struct Node *nodes, int node_count, char *name) {
    for (int i = 0; i < node_count; i++) {
        if (strcmp(nodes[i].name, name) == 0) {
            return i;
        }
    }
    return -1;  // return -1 if the node is not found
}

void readGraph(char *filename, int *N, int *M, struct Edge **edges, struct Node **nodes) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file");
        exit(EXIT_FAILURE);
    }

    char header[256];
    if (fgets(header, sizeof(header), file) == NULL) {
        printf("Error reading header");
        exit(EXIT_FAILURE);
    }

    *M = 0;
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        (*M)++;
    }

    *edges = (struct Edge *)malloc(*M * sizeof(struct Edge));
    if (*edges == NULL) {
        printf("Error allocating memory for edges");
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_SET);
    fgets(header, sizeof(header), file); // skip the headers

    int node_count = 0;
    int max_nodes = 10; // initially we suppose that we have arbitrarily 10 max nodes so that we dont 
                        // allocate a very large memory without knowing the actual dataset  
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

        // source and dest nodes to nodes array
        if (findNodeIndex(*nodes, node_count, source) == -1) {
            if (node_count >= max_nodes) {  // dynamically resize array if num of nodes greater than arbitrary max nodes
                max_nodes *= 2;
                *nodes = realloc(*nodes, max_nodes * sizeof(struct Node));
            }
            (*nodes)[node_count].name = strdup(source);
            (*nodes)[node_count].index = node_count;
            node_count++;
        }
        if (findNodeIndex(*nodes, node_count, target) == -1) {
            if (node_count >= max_nodes) {
                max_nodes *= 2;
                *nodes = realloc(*nodes, max_nodes * sizeof(struct Node));
            }
            (*nodes)[node_count].name = strdup(target);
            (*nodes)[node_count].index = node_count;
            node_count++;
        }
    }
    *N = node_count;
    fclose(file);
}

// deallocating dynamic resources
void freeResources(struct Edge *edges, struct Node *nodes, int *matrix, int M, int N) {
    for (int i = 0; i < M; i++) {
        free(edges[i].source);
        free(edges[i].target);
    }
    free(edges);

    for (int i = 0; i < N; i++) {
        free(nodes[i].name);
    }
    free(nodes);
    free(matrix);
}

int compare(const void *a, const void *b) {
    int l = ((struct QueueNode *)a)->distance;
    int r = ((struct QueueNode *)b)->distance;
    return (l - r);
}

void findKShortest(struct Node *nodes, struct Edge *edges, int N, int M, int k, int sourceIndex, int destinationIndex) {
    int *dis = malloc(N * k * sizeof(int));
    for (int i = 0; i < N * k; i++) {
        dis[i] = INF;
    }
    dis[sourceIndex * k] = 0; // start from the source node

    struct QueueNode *pq = malloc(N * k * sizeof(struct QueueNode));
    int pqSize = 0;
    pq[pqSize++] = (struct QueueNode){0, sourceIndex};

    while (pqSize > 0) {
        qsort(pq, pqSize, sizeof(struct QueueNode), compare);
        struct QueueNode top = pq[--pqSize];

        int u = top.vertex;
        int d = top.distance;
        if (dis[u * k + k - 1] < d) 
            continue;

        for (int i = 0; i < M; i++) {
            if (strcmp(nodes[u].name, edges[i].source) == 0) {
                int dest = findNodeIndex(nodes, N, edges[i].target);
                int cost = edges[i].weight;
                if (d + cost < dis[dest * k + k - 1]) {
                    dis[dest * k + k - 1] = d + cost;
                    qsort(&dis[dest * k], k, sizeof(int), compare);
                    pq[pqSize++] = (struct QueueNode){d + cost, dest};
                }
            }
        }
    }

    // K shortest paths for the specified destination node from the source node
    printf("\nTop %d shortest paths from node %s to node %s:\n\n", k, nodes[sourceIndex].name, nodes[destinationIndex].name);
    for (int i = 0; i < k; i++) {
        if (dis[destinationIndex * k + i] == INF)
            printf("Path %d: Distance = INF\n", i + 1);
        else
            printf("Path %d: Distance = %d\n", i + 1, dis[destinationIndex * k + i]);
    }
    // printf("\n");

    free(dis);
    free(pq);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage format: %s <filename.csv>\n", argv[0]);
        return 0;
    }

    int N, M, K;
    printf("\nEnter the number of shortest paths (K): ");
    fflush(stdout); // clearing the buffer
    scanf("%d", &K);
    if (K <= 0) {
        printf("K must be a positive integer.\n");
        fflush(stdout);
        return 0;
    }

    struct Edge *edges;
    struct Node *nodes;

    readGraph(argv[1], &N, &M, &edges, &nodes);

    /* DEBUGGING PURPOSES */
    // // init distance matrix
    // int *matrix = malloc(N * N * sizeof(int));
    // for (int i = 0; i < N * N; i++) {
    //     matrix[i] = INF;
    // }
    // for (int i = 0; i < N; i++) {
    //     matrix[i * N + i] = 0;  // set diagonal to zero
    // }
    // for (int i = 0; i < M; i++) {
    //     int u = findNodeIndex(nodes, N, edges[i].source);
    //     int v = findNodeIndex(nodes, N, edges[i].target);
    //     matrix[u * N + v] = edges[i].weight;
    // }

    // printf("\nAdjacency Matrix:\n");

    // // printing the adj matrix - row and col headers
    // printf("     ");
    // for (int i = 0; i < N; i++) {
    //     printf("%5d ", i + 1);
    // }
    // printf("\n");

    // for (int i = 0; i < N; i++) {
    //     printf("%5d ", i + 1);
    //     for (int j = 0; j < N; j++) {
    //         if (matrix[i * N + j] == INF)
    //             printf("  INF ");
    //         else
    //             printf("%5d ", matrix[i * N + j]);
    //     }
    //     printf("\n");
    // }

    // printf("\n");

    srand(time(NULL));

    struct NodePair pairs[10];
    for (int i = 0; i < 10; i++) {
        pairs[i].srcIndex = rand() % N; // selecting random source node
        do {
            pairs[i].destIndex = rand() % N; // random dest node
        } while (pairs[i].srcIndex == pairs[i].destIndex); // source can not be the same as dest
    }

    clock_t start, end;
    double total_time = 0.0, cpu_time_used;

    for (int i = 0; i < 10; i++) {
        int src = pairs[i].srcIndex;
        int dest = pairs[i].destIndex;

        start = clock();
        findKShortest(nodes, edges, N, M, K, src, dest);
        end = clock();

        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        total_time += cpu_time_used;
        printf("\nTime taken for nodes %d to %d: %f seconds\n", src + 1, dest + 1, cpu_time_used);
        printf("\n-----------------------------------------------------------------------------------------");
    }

    double average_time = total_time / 10;
    printf("\n\nMaximum Number of Nodes: %d\n", N);
    printf("Maximum Number of Edges: %d\n", M);
    printf("\nAverage time taken: %f seconds\n", average_time);

    freeResources(edges, nodes, NULL, M, N);

    return 0;
}
