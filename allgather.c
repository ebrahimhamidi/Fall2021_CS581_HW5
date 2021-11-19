/* Sample program to illustrate the scatter operation using
 * non-blocking point-to-point primitives.  
 *
 * To Compile: mpicc -Wall -o myscatter myscatter.c
 * To Run: On your local system: mpiexec -n 4 ./myscatter 
 *         On DMC cluster, see instructions provided in README.md
 */

/*   
   The scatter operation using MPI sample program (provided by the instructor) was modified to Collective Communication Implementation.
   Modified by: Ebrahim Hamidi
   Date: Nov. 19, 2021
   Email: shamidi1@crimson.ua.edu
   Course Section: CS 581
   Homework-5 Collective Communication Implementation 
   gcc compiler:   mpicc -g -Wall -O -o allgather allgather.c
   to run: srun --mpi=pmi2 -n 8 ./allgather
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>

#define MAXN 262144
#define NTIMES 10

void allgather (void *sendbuf, int sendcount, MPI_Datatype sendtype,
		void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

int main (int argc, char **argv) {
  int i, j, rank, size, msgsize, offset;
  int *sendbuf = NULL, *recvbuf;
  double t1, t2;

  MPI_Init (&argc, &argv);

  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  sendbuf = (int *) malloc (sizeof (int) * MAXN);
  for (i = 0; i < MAXN; i++){
		sendbuf[i] = rank * MAXN + i;
		//printf ("[%d]sendbuf[%d] = %d \n", rank, i, sendbuf[i]);
	}
  recvbuf = (int *) malloc (sizeof (int) * MAXN * size);

  for (msgsize = 8; msgsize <= MAXN; msgsize = msgsize << 1) {
      /* setup a synchronization point */
      MPI_Barrier (MPI_COMM_WORLD);
      t1 = MPI_Wtime ();
      /* include rest of the program here */
      for (i = 0; i < NTIMES; i++)
	        allgather (sendbuf, msgsize, MPI_INT, recvbuf, msgsize, MPI_INT, MPI_COMM_WORLD);
      /* program end here */
      t2 = MPI_Wtime () - t1;

      MPI_Reduce (&t2, &t1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      /* check if everyone got the correct results */
      for (i = 0; i < size; i++) {
          for (j = 0, offset = i*MAXN; j < msgsize; j++){
              assert(recvbuf[i*msgsize + j] == (offset + j));			
		  }
	  }

      if (rank == 0) {
	       printf ("Message size = %ld bytes, Maximum time = %g\n",
		             msgsize * sizeof (int), t1 / 100.0);
      }	  
  }
  
  MPI_Finalize ();

  return 0;
}

void allgather (void *sendbuf, int sendcount, MPI_Datatype sendtype,
	        void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
  int rank, size, i, offset;
  MPI_Status *status;
  MPI_Request *request;
  MPI_Aint lb, sizeofsendtype, sizeofrecvtype;

  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);

  status = malloc (sizeof (MPI_Status) * (size + 1));
  request = malloc (sizeof (MPI_Request) * (size + 1));
  
  MPI_Type_get_extent (sendtype, &lb, &sizeofsendtype);
  MPI_Isend (sendbuf, sizeofsendtype * sendcount, MPI_CHAR, 0, 0, comm, &request[0]);

  if (rank == 0){
	for (i = 0; i < size; i++) {
      MPI_Type_get_extent (sendtype, &lb, &sizeofsendtype);
	  offset = sizeofsendtype * sendcount * i;

	  char *bufptr = recvbuf + offset;
      MPI_Irecv (bufptr, sizeofsendtype * sendcount, MPI_CHAR, i, 0, comm, &request[i+1]);
	}
	MPI_Waitall (size + 1, request, status);

  }else
	  MPI_Wait (&request[0], MPI_STATUS_IGNORE);
    
  if (rank == 0) {	
      for (i = 1; i < size; i++) {
		    MPI_Type_get_extent (recvtype, &lb, &sizeofrecvtype);
	        MPI_Isend (recvbuf, sizeofrecvtype * recvcount * (size), MPI_CHAR, i, 0, comm, &request[i-1]);
      }
      MPI_Waitall (size - 1, request, status);
  } else {   
      MPI_Type_get_extent (recvtype, &lb, &sizeofrecvtype);
      MPI_Irecv (recvbuf, sizeofrecvtype * recvcount * (size), MPI_CHAR, 0, 0, comm, &request[0]);	  
      MPI_Wait (&request[0], MPI_STATUS_IGNORE);
  }
 
  free (request);
  free (status);
}