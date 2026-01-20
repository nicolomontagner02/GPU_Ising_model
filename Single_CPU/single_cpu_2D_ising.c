#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int DEBUG = 0;

float energy_2D(int **lattice, int size_x, int size_y, float J, float h) {
    float energy = 0.0;
    for (int i = 0; i < size_x; i++) {
        for (int j = 0; j < size_y; j++) {
            int spin = lattice[i][j];
            int right_neighbor = 0;
            int bottom_neighbor = 0;
            if (j != size_y-1){
                right_neighbor = lattice[i][j+1];
            }
            if (i != size_x-1){
                bottom_neighbor = lattice[i+1][j];
            }
            if (DEBUG){
                printf("Right neighbor spin of (%i,%i): %i\n",i,j, right_neighbor);
                printf("Bottom neighbor spin of (%i,%i): %i\n",i,j, bottom_neighbor);
            }
            energy -= J * spin * (right_neighbor + bottom_neighbor);
            energy -= h * spin;
        }
    }
    return energy;
}

float magnetisation_2D(int **lattice, int size_x, int size_y){

    int N = size_x*size_y;

    int magnetisation = 0;

    for (int i = 0; i < size_x; i++){
        for (int j = 0; j < size_y; j++){
            magnetisation += lattice[i][j];
        }
    }

    float avg_magnetisation = magnetisation / N;

    return avg_magnetisation;
}

float d_energy_2D(int **lattice, int i, int j, int size_x, int size_y, float J, float h){

    int spin = lattice[i][j];
    float d_energy = 2*h*spin;

    if (i != 0){
        d_energy += 2 * J * spin * lattice[i-1][j];
    }
    if (j != 0){
        d_energy += 2 * J * spin * lattice[i][j-1];
    }
    if (i != size_x-1){
        d_energy += 2 * J * spin * lattice[i+1][j];
    }
    if (i != size_y-1){
        d_energy += 2 * J * spin * lattice[i][j+1];
    }

    return d_energy;
}

float energy_density_2D(float energy, int size_x, int size_y){

    int N = size_x*size_y;

    float e_density = energy / N;

    return e_density;
}

void MH_step(int **lattice, int size_x, int size_y, float J, float h, float kB, float T){
    
    // flip site
    int i_s = rand() % size_x;
    int j_s = rand() % size_y;
    if (DEBUG){
        printf("i: %i, j: %i\n", i_s, j_s);
    }

    lattice[i_s][j_s] *= -1;

    float d_energy = d_energy_2D(lattice, i_s, j_s, size_x, size_y, J,h);

    if (d_energy > 0){
        float p_step = exp(-d_energy/kB/T);
        float u = (float)rand() / (float)RAND_MAX;
        if (p_step > u){
            lattice[i_s][j_s] *= -1;
            if (DEBUG){
                printf("Step not executed.\n");
            }
        }
    }
}

void print_lattice(int **lattice, int size_x, int size_y){

    printf("2D Ising Lattice:\n");
        for (int i = 0; i < size_x; i++) {
            for (int j = 0; j < size_y; j++) {
                printf("%d ", lattice[i][j]);
            }
            printf("\n");
        }

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
    float kB = 1.0;
    float T = 1.0;
    
    // Allocate memory for the lattice
    int **lattice = (int **)malloc(lattice_size_x * sizeof(int *));
    for (int i = 0; i < lattice_size_x; i++) {
        lattice[i] = (int *)malloc(lattice_size_y * sizeof(int));
    }

    int c = 3;

    switch(c){
        // Switch for different initializations cases
        case 1:
            // Initialize to all spins up (+1)
            for (int i = 0; i < lattice_size_x; i++) {
                for (int j = 0; j < lattice_size_y; j++) {
                    lattice[i][j] = 1;
                }
            }
            break;

        case 2:
            // Initialize to all spins down (-1)
            for (int i = 0; i < lattice_size_x; i++) {
                for (int j = 0; j < lattice_size_y; j++) {
                    lattice[i][j] = -1;
                }
            }
            break;

        case 3:
            // Initialize to random spins (+1 or -1)
            srand(2); // Seed for reproducibility
            for (int i = 0; i < lattice_size_x; i++) {
                for (int j = 0; j < lattice_size_y; j++) {
                    lattice[i][j] = (rand() % 2) * 2 - 1; // Randomly assign +1 or -1
                }

            }
            break;

        default:
            printf("Wrong initialization case submitted!\n");
            return 1;

    }
        
    //print the lattice (only small size)
    if (lattice_size_x <=4 && lattice_size_y <=4){
        print_lattice(lattice, lattice_size_x, lattice_size_y);
    }  
    // Calculate and print the energy
    float total_energy = energy_2D(lattice, lattice_size_x, lattice_size_y, J, h);
    printf("Total Energy of the 2D Ising Lattice: %f\n", total_energy);

    float e_density = energy_density_2D(total_energy, lattice_size_x, lattice_size_y);
    printf("Energy density of the 2D Ising Lattice: %f\n", e_density);

    float magnetisation = magnetisation_2D(lattice, lattice_size_x, lattice_size_y);
    printf("Total Magnetisation of the 2D Ising Lattice: %f\n", magnetisation);

    for (int i = 0; i < 4000; i++){
        MH_step(lattice, lattice_size_x, lattice_size_y, J ,h, kB, T);
        if (DEBUG){
            printf("MH step %i executed.\n",i);
        }
    }

    if (lattice_size_x <=4 && lattice_size_y <=4){
        print_lattice(lattice, lattice_size_x, lattice_size_y);
    }  

    return 0;

}