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

  };
}
#endif
