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
   Date: Nov. 21, 2021
   Email: shamidi1@crimson.ua.edu
   Course Section: CS 581
   Homework-5 Collective Communication Implementation _ Version 2: pair-wise exchange
   gcc compiler:   mpicc -g -Wall -O -o pairAllgather pairAllgather.c
   to run: srun --mpi=pmi2 -n 8 ./pairAllgather
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>

#define MAXN 262144
#define NTIMES 100

int power(int base, int exponent) {
    long long power = 1;
    int i;

    for(i=1; i<=exponent; i++)
    {
        power = power * base;
    }

    return power;
}

void allgather (void *sendbuf, int sendcount, MPI_Datatype sendtype,
		void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

int main (int argc, char **argv) {
  int i, j, rank, size, msgsize, offset;
  int *sendbuf = NULL, *recvbuf;
  double t1, t2;  

  MPI_Init (&argc, &argv);

  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  sendbuf = (int *) calloc (sizeof (int), MAXN);
  for (i = 0; i < MAXN; i++){
		sendbuf[i] = rank * MAXN + i;
	}
  recvbuf = (int *) calloc (sizeof (int), MAXN * size);

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
	       printf ("(pair)Message size = %ld bytes, Maximum time = %g\n",
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
  MPI_Aint lb, sizeofsendtype; //, sizeofrecvtype;

  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);

  status = malloc (sizeof (MPI_Status) * (2));
  request = malloc (sizeof (MPI_Request) * (2));
  
  MPI_Type_get_extent (sendtype, &lb, &sizeofsendtype);
//  MPI_Isend (sendbuf, sizeofsendtype * sendcount, MPI_CHAR, rank, 0, comm, &request[0]);
  
  MPI_Type_get_extent (sendtype, &lb, &sizeofsendtype);
  offset = sizeofsendtype * sendcount * rank;
  char *bufptr = recvbuf + offset;
 // MPI_Irecv (bufptr, sizeofsendtype * sendcount, MPI_CHAR, rank, 0, comm, &request[1]);
  
//  MPI_Waitall (2, request, MPI_STATUSES_IGNORE);
//}


  	MPI_Sendrecv(sendbuf, sizeofsendtype * sendcount, MPI_CHAR, rank, 0, 
		      bufptr, sizeofsendtype * sendcount, MPI_CHAR, rank, 0, 
		      comm, status);
  
  int offset1, offset2;
  int offset_send, offset_recv;
  int ZdbT, NofStep = 0;
  ZdbT = size;
  
  while (ZdbT > 1)
  {
	ZdbT = ZdbT/2;
	NofStep += 1;
  }

  int DirPair, NextPair, pairOffset;
  pairOffset = 1;
  DirPair = rank;
  for (i = 0; i < NofStep; i++) {
	
	int StepSize;
    StepSize = power((double)2, (double)i);
	
		if ((DirPair & 1) == 0){
		NextPair = rank + pairOffset;
		offset_send = DirPair*StepSize ;
		offset_recv = offset_send + pairOffset;
	}else{
		NextPair = rank - pairOffset;
		offset_send = DirPair*StepSize;
		offset_recv = offset_send - pairOffset;
	}
		
	DirPair = DirPair/2;
	pairOffset = pairOffset*2;

  MPI_Type_get_extent (sendtype, &lb, &sizeofsendtype);
  offset1 = sizeofsendtype * sendcount * offset_send;
  char *bufptr1 = recvbuf + offset1;
 // MPI_Isend (bufptr1, sizeofsendtype * sendcount * StepSize, MPI_CHAR, NextPair, 0, comm, &request[0]);
  
  MPI_Type_get_extent (sendtype, &lb, &sizeofsendtype);
  offset2 = sizeofsendtype * sendcount * offset_recv;
  char *bufptr2 = recvbuf + offset2;
//  MPI_Irecv (bufptr2, sizeofsendtype * sendcount * StepSize, MPI_CHAR, NextPair, 0, comm, &request[1]);
  
 // MPI_Waitall(2, request, status);
  
  
  	MPI_Sendrecv(bufptr1, sizeofsendtype * sendcount * StepSize, MPI_CHAR, NextPair, 0, 
	      bufptr2, sizeofsendtype * sendcount * StepSize, MPI_CHAR, NextPair, 0, 
		      comm, status);
}

  free (request);
  free (status);
}