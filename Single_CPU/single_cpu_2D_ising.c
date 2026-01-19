#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float energy_2D(int **lattice, int size_x, int size_y, float J, float h) {
    float energy = 0.0;
    for (int i = 0; i < size_x; i++) {
        for (int j = 0; j < size_y; j++) {
            int spin = lattice[i][j];
            int right_neighbor = lattice[i][(j + 1) % size_y];
            if (i == size_x-1){
                right_neighbor = 0;
            }
            int bottom_neighbor = lattice[(i + 1) % size_x][j];
            if (j == size_y-1){
                bottom_neighbor = 0;
            }
            energy -= J * spin * (right_neighbor + bottom_neighbor);
            energy -= h * spin;
        }
    }
    return energy;
}

int main(int argc, char *argv[]) {
    // Check command line arguments
    if (argc != 3) {
        printf("Usage: %s <lattice_size_x> <lattice_size_y>\n", argv[0]);
        printf("Example: %s 10 10\n", argv[0]);
        return 1;
    }

    // Parse command line arguments
    int lattice_size_x = atoi(argv[1]);
    int lattice_size_y = atoi(argv[2]);

    float J = 1.0; // Interaction strength
    float h = 1.0 ; // External magnetic field
    
    // Allocate memory for the lattice
    int **lattice = (int **)malloc(lattice_size_x * sizeof(int *));
    for (int i = 0; i < lattice_size_x; i++) {
        lattice[i] = (int *)malloc(lattice_size_y * sizeof(int));
    }

    int c = 1;

    switch(c){
        // Switch for different initializations cases
        case 1:
        // Initialize to all spins up (+1)
        for (int i = 0; i < lattice_size_x; i++) {
            for (int j = 0; j < lattice_size_y; j++) {
                lattice[i][j] = 1;
            }
        }

        case 2:
        // Initialize to all spins down (-1)
        for (int i = 0; i < lattice_size_x; i++) {
            for (int j = 0; j < lattice_size_y; j++) {
                lattice[i][j] = -1;
            }
        }
    
        case 3:
        // Initialize to random spins (+1 or -1)
        srand(42); // Seed for reproducibility
        for (int i = 0; i < lattice_size_x; i++) {
            for (int j = 0; j < lattice_size_y; j++) {
                lattice[i][j] = (rand() % 2) * 2 - 1; // Randomly assign +1 or -1
            }

        }
    }
        

    //print the lattice
    if (lattice_size_x <=4 && lattice_size_y <=4){
        printf("2D Ising Lattice:\n");
        for (int i = 0; i < lattice_size_x; i++) {
            for (int j = 0; j < lattice_size_y; j++) {
                printf("%d ", lattice[i][j]);
            }
            printf("\n");
        }
    }  
    // Calculate and print the energy
    float total_energy = energy_2D(lattice, lattice_size_x, lattice_size_y, J, h);
    printf("Total Energy of the 2D Ising Lattice: %f\n", total_energy);

    return 0;

}