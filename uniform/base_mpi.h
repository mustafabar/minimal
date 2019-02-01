#ifndef base_mpi_h
#define base_mpi_h
#include <mpi.h>
#include <iostream>
#include "types.h"
#include <unistd.h>

namespace exafmm {
  //! Custom MPI utilities
  class BaseMPI {
  private:
    int external;                                               //!< Flag to indicate external MPI_Init/Finalize

  protected:
    const int wait;                                             //!< Waiting time between output of different ranks

  public:
    int mpirank;                                                //!< Rank of MPI communicator
    int mpisize;                                                //!< Size of MPI communicator

  public:
    //! Constructor
    BaseMPI() : external(0), wait(100) {                        // Initialize variables
      int argc(0);                                              // Dummy argument count
      char **argv;                                              // Dummy argument value
      MPI_Initialized(&external);                               // Check if MPI_Init has been called
      if (!external) MPI_Init(&argc, &argv);                    // Initialize MPI communicator
      MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);                  // Get rank of current MPI process
      MPI_Comm_size(MPI_COMM_WORLD, &mpisize);                  // Get number of MPI processes
    }

    //! Destructor
    ~BaseMPI() {
      if (!external) MPI_Finalize();                            // Finalize MPI communicator
    }

    //! Allreduce int type from all ranks
    int allreduceInt(int send) {
      int recv;                                                 // Receive buffer
      MPI_Allreduce(&send, &recv, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);// Communicate values
      return recv;                                              // Return received values
    }

    //! Allreduce vec3 type from all ranks
    vec3 allreduceVec3(vec3 send) {
      fvec3 fsend, frecv;                                       // Single precision buffers
      for (int d=0; d<3; d++) fsend[d] = send[d];               // Copy to send buffer
      MPI_Allreduce(fsend, frecv, 3, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);// Communicate values
      vec3 recv;                                                // Receive buffer
      for (int d=0; d<3; d++) recv[d] = frecv[d];               // Copy from recv buffer
      return recv;                                              // Return received values
    }

    //! Allreduce bounds type from all ranks
    Bounds allreduceBounds(Bounds local) {
      fvec3 localXmin, localXmax, globalXmin, globalXmax;
      for (int d=0; d<3; d++) {                                 // Loop over dimensions
	localXmin[d] = local.Xmin[d];                           //  Convert Xmin to float
	localXmax[d] = local.Xmax[d];                           //  Convert Xmax to float
      }                                                         // End loop over dimensions
      MPI_Allreduce(localXmin, globalXmin, 3, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);// Reduce domain Xmin
      MPI_Allreduce(localXmax, globalXmax, 3, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);// Reduce domain Xmax
      Bounds global;
      for (int d=0; d<3; d++) {                                 // Loop over dimensions
	real_t leeway = (globalXmax[d] - globalXmin[d]) * 1e-6; //  Adding a bit of leeway to global domain
	global.Xmin[d] = globalXmin[d] - leeway;                //  Convert Xmin to real_t
	global.Xmax[d] = globalXmax[d] + leeway;                //  Convert Xmax to real_t
      }                                                         // End loop over dimensions
      return global;                                            // Return global bounds
    }

    //! Allreduce waves type from all ranks
    Waves allreduceWaves(Waves waves) {
      int numWaves = waves.size();
      std::vector<float> fsend(2*numWaves);                     // Single precision buffers
      int w = 0;                                                // Wave counter
      for (W_iter W=waves.begin(); W!=waves.end(); W++, w++) {  // Loop over waves
        fsend[2*w+0] = W->REAL;                                 //  Real part
        fsend[2*w+1] = W->IMAG;                                 //  Imag part
      }                                                         // End loop over waves
      std::vector<float> frecv(2*numWaves);                     // Single precision buffers
      MPI_Allreduce(&fsend[0], &frecv[0], 2*numWaves, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);// Communicate values
      w = 0;                                                    // Wave counter
      for (W_iter W=waves.begin(); W!=waves.end(); W++, w++) {  // Loop over waves
        W->REAL = frecv[2*w+0];                                 //  Real part
        W->IMAG = frecv[2*w+1];                                 //  Imag part
      }                                                         // End loop over waves
      return waves;                                             // Return received values
    }

    //! Send bodies to next rank (round robin)
    void shiftBodies(Bodies & bodies) {
      int newSize;                                              // New number of bodies
      int oldSize = bodies.size();                              // Current number of bodies
      const int word = sizeof(bodies[0]) / 4;                   // Word size of body structure
      const int isend = (mpirank + 1          ) % mpisize;      // Send to next rank (wrap around)
      const int irecv = (mpirank - 1 + mpisize) % mpisize;      // Receive from previous rank (wrap around)
      MPI_Request sreq,rreq;                                    // Send, receive request handles

      MPI_Isend(&oldSize, 1, MPI_INT, irecv, 0, MPI_COMM_WORLD, &sreq);// Send current number of bodies
      MPI_Irecv(&newSize, 1, MPI_INT, isend, 0, MPI_COMM_WORLD, &rreq);// Receive new number of bodies
      MPI_Wait(&sreq, MPI_STATUS_IGNORE);                       // Wait for send to complete
      MPI_Wait(&rreq, MPI_STATUS_IGNORE);                       // Wait for receive to complete

      Bodies recvBodies(newSize);                               // Resize buffer to new number of bodies
      MPI_Isend((int*)&bodies[0], oldSize*word, MPI_INT, irecv, // Send bodies to next rank
		1, MPI_COMM_WORLD, &sreq);
      MPI_Irecv((int*)&recvBodies[0], newSize*word, MPI_INT, isend,// Receive bodies from previous rank
		1, MPI_COMM_WORLD, &rreq);
      MPI_Wait(&sreq, MPI_STATUS_IGNORE);                       // Wait for send to complete
      MPI_Wait(&rreq, MPI_STATUS_IGNORE);                       // Wait for receive to complete
      bodies = recvBodies;                                      // Copy bodies from buffer
    }

    //! Print a scalar value on all ranks
    template<typename T>
    void print(T data) {
      for (int irank=0; irank<mpisize; irank++ ) {              // Loop over ranks
	MPI_Barrier(MPI_COMM_WORLD);                            //  Sync processes
	usleep(wait);                                           //  Wait "wait" milliseconds
	if (mpirank == irank) std::cout << data << " ";         //  If it's my turn print "data"
      }                                                         // End loop over ranks
      MPI_Barrier(MPI_COMM_WORLD);                              // Sync processes
      usleep(wait);                                             // Wait "wait" milliseconds
      if (mpirank == mpisize-1) std::cout << std::endl;         // New line
    }

    //! Print a scalar value on irank
    template<typename T>
    void print(T data, const int irank) {
      MPI_Barrier(MPI_COMM_WORLD);                              // Sync processes
      usleep(wait);                                             // Wait "wait" milliseconds
      if( mpirank == irank ) std::cout << data;                 // If it's my rank print "data"
    }

    //! Print a vector value on all ranks
    template<typename T>
    void print(T * data, const int begin, const int end) {
      for (int irank=0; irank<mpisize; irank++) {               // Loop over ranks
	MPI_Barrier(MPI_COMM_WORLD);                            //  Sync processes
	usleep(wait);                                           //  Wait "wait" milliseconds
	if (mpirank == irank) {                                 //  If it's my turn to print
	  std::cout << mpirank << " : ";                        //   Print rank
	  for (int i=begin; i<end; i++) {                       //   Loop over data
	    std::cout << data[i] << " ";                        //    Print data[i]
	  }                                                     //   End loop over data
	  std::cout << std::endl;                               //   New line
	}                                                       //  Endif for my turn
      }                                                         // End loop over ranks
    }

    //! Print a vector value on irank
    template<typename T>
    void print(T * data, const int begin, const int end, const int irank) {
      MPI_Barrier(MPI_COMM_WORLD);                              // Sync processes
      usleep(wait);                                             // Wait "wait" milliseconds
      if (mpirank == irank) {                                   // If it's my rank
	std::cout << mpirank << " : ";                          //  Print rank
	for (int i=begin; i<end; i++) {                         //  Loop over data
	  std::cout << data[i] << " ";                          //   Print data[i]
	}                                                       //  End loop over data
	std::cout << std::endl;                                 //  New line
      }                                                         // Endif for my rank
    }
  };
}
#endif
