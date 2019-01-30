#ifndef tree_mpi_h
#define tree_mpi_h
#include "base_mpi.h"
#include "logger.h"
#include "namespace.h"
#include "types.h"

namespace EXAFMM_NAMESPACE {
  //! Handles all the communication of local essential trees
  class TreeMPI {
  private:
    Kernel & kernel;                                            //!< Kernel class
    BaseMPI & baseMPI;                                          //!< MPI utils
    const int mpirank;                                          //!< Rank of MPI communicator
    const int mpisize;                                          //!< Size of MPI communicator
    const real_t theta;                                         //!< Multipole acceptance criteria
    const int images;                                           //!< Number of periodic image sublevels
    std::vector<fvec3> allBoundsXmin;                           //!< Array for local Xmin for all ranks
    std::vector<fvec3> allBoundsXmax;                           //!< Array for local Xmax for all ranks
    Bodies sendBodies;                                          //!< Send buffer for bodies
    Bodies recvBodies;                                          //!< Receive buffer for bodies
    Cells sendCells;                                            //!< Send buffer for cells
    Cells recvCells;                                            //!< Receive buffer for cells
    std::vector<int> sendBodyCount;                             //!< Send count
    std::vector<int> sendBodyDispl;                             //!< Send displacement
    std::vector<int> recvBodyCount;                             //!< Receive count
    std::vector<int> recvBodyDispl;                             //!< Receive displacement
    std::vector<int> sendCellCount;                             //!< Send count
    std::vector<int> sendCellDispl;                             //!< Send displacement
    std::vector<int> recvCellCount;                             //!< Receive count
    std::vector<int> recvCellDispl;                             //!< Receive displacement

  public:
    //! Constructor
    TreeMPI(Kernel & _kernel, BaseMPI & _baseMPI, real_t _theta, int _images) :
      kernel(_kernel), baseMPI(_baseMPI), mpirank(baseMPI.mpirank), mpisize(baseMPI.mpisize),
      theta(_theta), images(_images) { // Initialize variables
      allBoundsXmin.resize(mpisize);                      // Allocate array for minimum of local domains
      allBoundsXmax.resize(mpisize);                      // Allocate array for maximum of local domains
      sendBodyCount.resize(mpisize);                        // Allocate send count
      sendBodyDispl.resize(mpisize);                        // Allocate send displacement
      recvBodyCount.resize(mpisize);                        // Allocate receive count
      recvBodyDispl.resize(mpisize);                        // Allocate receive displacement
      sendCellCount.resize(mpisize);                        // Allocate send count
      sendCellDispl.resize(mpisize);                        // Allocate send displacement
      recvCellCount.resize(mpisize);                        // Allocate receive count
      recvCellDispl.resize(mpisize);                        // Allocate receive displacement
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

      recvBodies.resize(newSize);                               // Resize buffer to new number of bodies
      MPI_Isend((int*)&bodies[0], oldSize*word, MPI_INT, irecv, // Send bodies to next rank
		1, MPI_COMM_WORLD, &sreq);
      MPI_Irecv((int*)&recvBodies[0], newSize*word, MPI_INT, isend,// Receive bodies from previous rank
		1, MPI_COMM_WORLD, &rreq);
      MPI_Wait(&sreq, MPI_STATUS_IGNORE);                       // Wait for send to complete
      MPI_Wait(&rreq, MPI_STATUS_IGNORE);                       // Wait for receive to complete
      bodies = recvBodies;                                      // Copy bodies from buffer
    }

  };
}
#endif
