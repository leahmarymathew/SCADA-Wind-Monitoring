#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define FEATURES 5

int main() {
    FILE *file = fopen("benchmarks/inference_input.csv", "r");
    if (!file) {
        printf("Failed to open inference_input.csv\n");
        return 1;
    }

    char buffer[1024];
    int count = 0;

    // Skip header
    fgets(buffer, sizeof(buffer), file);

    // Count rows
    while (fgets(buffer, sizeof(buffer), file)) {
        count++;
    }

    rewind(file);
    fgets(buffer, sizeof(buffer), file); // skip header again

    double **data = malloc(count * sizeof(double *));
    for (int i = 0; i < count; i++) {
        data[i] = malloc(FEATURES * sizeof(double));
    }

    for (int i = 0; i < count; i++) {
        fscanf(
            file,
            "%lf,%lf,%lf,%lf,%lf",
            &data[i][0],
            &data[i][1],
            &data[i][2],
            &data[i][3],
            &data[i][4]
        );
    }

    fclose(file);

    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < count; i++) {
        double score = 0.0;
        for (int j = 0; j < FEATURES; j++) {
            score += data[i][j] * 0.001;
        }
        score = score > 0.5 ? 1.0 : 0.0;
    }

    double end = omp_get_wtime();

    printf("OpenMP inference time: %f seconds\n", end - start);
    printf("Records processed: %d\n", count);
    printf("Threads used: %d\n", omp_get_max_threads());

    for (int i = 0; i < count; i++) {
        free(data[i]);
    }
    free(data);

    return 0;
}
