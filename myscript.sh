#!/bin/bash
module load openmpi/4.0.5-gnu-pmi2
srun --mpi=pmi2 -n 16 ./allgather
srun --mpi=pmi2 -n 16 ./pairAllgather
