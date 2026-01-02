#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define FEATURES 5

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    FILE *file = fopen("benchmarks/inference_input.csv", "r");
    if (!file) {
        if (rank == 0)
            printf("Failed to open inference_input.csv\n");
        MPI_Finalize();
        return 1;
    }

    char buffer[1024];
    int total_rows = 0;

    fgets(buffer, sizeof(buffer), file); // skip header
    while (fgets(buffer, sizeof(buffer), file)) {
        total_rows++;
    }

    rewind(file);
    fgets(buffer, sizeof(buffer), file);

    int rows_per_proc = total_rows / size;
    int start = rank * rows_per_proc;
    int end = (rank == size - 1) ? total_rows : start + rows_per_proc;

    double **data = malloc((end - start) * sizeof(double *));
    for (int i = 0; i < end - start; i++) {
        data[i] = malloc(FEATURES * sizeof(double));
    }

    int current = 0;
    int local_index = 0;

    while (fgets(buffer, sizeof(buffer), file)) {
        if (current >= start && current < end) {
            sscanf(
                buffer,
                "%lf,%lf,%lf,%lf,%lf",
                &data[local_index][0],
                &data[local_index][1],
                &data[local_index][2],
                &data[local_index][3],
                &data[local_index][4]
            );
            local_index++;
        }
        current++;
    }

    fclose(file);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int i = 0; i < local_index; i++) {
        double score = 0.0;
        for (int j = 0; j < FEATURES; j++) {
            score += data[i][j] * 0.001;
        }
        score = score > 0.5 ? 1.0 : 0.0;
    }

    double local_time = MPI_Wtime() - start_time;
    double max_time;

    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("MPI distributed inference time: %f seconds\n", max_time);
        printf("Processes used: %d\n", size);
        printf("Total records: %d\n", total_rows);
    }

    for (int i = 0; i < local_index; i++) {
        free(data[i]);
    }
    free(data);

    MPI_Finalize();
    return 0;
}
