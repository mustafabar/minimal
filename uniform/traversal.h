#ifndef traversal_h
#define traversal_h
#include "logger.h"
#include "namespace.h"
#include "thread.h"
#include "types.h"

#if EXAFMM_COUNT_KERNEL
#define countKernel(N) N++
#else
#define countKernel(N)
#endif

namespace EXAFMM_NAMESPACE {
  class Traversal {
  private:
    Kernel & kernel;                                            //!< Kernel class
    const real_t theta;                                         //!< Multipole acceptance criteria
    const int nspawn;                                           //!< Threshold of NBODY for spawning new threads
    const int images;                                           //!< Number of periodic image sublevels
    const char * path;                                          //!< Path to save files
#if EXAFMM_COUNT_KERNEL
    real_t numP2P;                                              //!< Number of P2P kernel calls
    real_t numM2L;                                              //!< Number of M2L kernel calls
#endif
    C_iter Ci0;                                                 //!< Iterator of first target cell
    C_iter Cj0;                                                 //!< Iterator of first source cell

  private:
#if EXAFMM_COUNT_LIST
    //! Accumulate interaction list size of cells
    void countList(C_iter Ci, C_iter Cj, bool isP2P) {
      if (isP2P) Ci->numP2P++;                                  // If P2P, increment P2P counter of target cell
      else Ci->numM2L++;                                        // Else, increment M2L counter of target cell
    }
#else
    void countList(C_iter, C_iter, bool) {}
#endif

#if EXAFMM_USE_WEIGHT
    //! Accumulate interaction weights of cells
    void countWeight(C_iter Ci, C_iter Cj, real_t weight) {
      Ci->WEIGHT += weight;                                     // Increment weight of target cell
    }
#else
    void countWeight(C_iter, C_iter, real_t) {}
#endif

    //! Get level from key
    int getLevel(uint64_t key) {
      int level = -1;                                           // Initialize level
      while( int(key) >= 0 ) {                                  // While key has level offsets to subtract
	level++;                                                //  Increment level
	key -= 1 << 3*level;                                    //  Subtract level offset
      }                                                         // End while loop for level offsets
      return level;                                             // Return level
    }

    //! Get 3-D index from key
    ivec3 getIndex(uint64_t key) {
      int level = -1;                                           // Initialize level
      while( int(key) >= 0 ) {                                  // While key has level offsets to subtract
	level++;                                                //  Increment level
	key -= 1 << 3*level;                                    //  Subtract level offset
      }                                                         // End while loop for level offsets
      key += 1 << 3*level;                                      // Compensate for over-subtraction
      level = 0;                                                // Initialize level
      ivec3 iX = 0;                                             // Initialize 3-D index
      int d = 0;                                                // Initialize dimension
      while( key > 0 ) {                                        // While key has bits to shift
	iX[d] += (key % 2) * (1 << level);                      //  Deinterleave key bits to 3-D bits
	key >>= 1;                                              //  Shift bits in key
	d = (d+1) % 3;                                          //  Increment dimension
	if( d == 0 ) level++;                                   //  Increment level
      }                                                         // End while loop for key bits to shift
      return iX;                                                // Return 3-D index
    }

    //! Get 3-D index from periodic key
    ivec3 getPeriodicIndex(int key) {
      ivec3 iX;                                                 // Initialize 3-D periodic index
      iX[0] = key % 3;                                          // x periodic index
      iX[1] = (key / 3) % 3;                                    // y periodic index
      iX[2] = key / 9;                                          // z periodic index
      iX -= 1;                                                  // {0,1,2} -> {-1,0,1}
      return iX;                                                // Return 3-D periodic index
    }

  public:
    //! Constructor
    Traversal(Kernel & _kernel, real_t _theta, int _nspawn, int _images, const char * _path) : // Constructor
      kernel(_kernel), theta(_theta), nspawn(_nspawn), images(_images), path(_path) // Initialize variables
#if EXAFMM_COUNT_KERNEL
      , numP2P(0), numM2L(0)
#endif
    {}

#if EXAFMM_COUNT_LIST
    //! Initialize size of P2P and M2L interaction lists per cell
    void initListCount(Cells & cells) {
      for (C_iter C=cells.begin(); C!=cells.end(); C++) {       // Loop over cells
	C->numP2P = C->numM2L = 0;                              //  Initialize size of interaction list
      }                                                         // End loop over cells
    }
#else
    void initListCount(Cells) {}
#endif

#if EXAFMM_USE_WEIGHT
    //! Initialize interaction weights of bodies and cells
    void initWeight(Cells & cells) {
      for (C_iter C=cells.begin(); C!=cells.end(); C++) {       // Loop over cells
	C->WEIGHT = 0;                                          //  Initialize cell weights
	if (C->NCHILD==0) {                                     //  If leaf cell
	  for (B_iter B=C->BODY; B!=C->BODY+C->NBODY; B++) {    //   Loop over bodies in cell
	    B->WEIGHT = 0;                                      //    Initialize body weights
	  }                                                     //   End loop over bodies in cell
	}                                                       //  End if for leaf cell
      }                                                         // End loop over cells
    }
#else
    void initWeight(Cells) {}
#endif

    //! Write G matrix to file
    void writeMatrix(Bodies & bodies, Bodies & jbodies) {
      std::stringstream name;                                   // File name
      name << path << "matrix.dat";                             // Create file name for matrix
      std::ofstream matrixFile(name.str().c_str());             // Open matrix log file
      for (B_iter Bi=bodies.begin(); Bi!=bodies.end(); Bi++) {  // Loop over target bodies
	for (B_iter Bj=jbodies.begin(); Bj!=jbodies.end(); Bj++) {//  Loop over source bodies
	  vec3 dX = Bi->X - Bj->X;                              //   Distance vector
	  real_t R2 = norm(dX) + kernel.eps2;                   //   Distance squared
	  real_t G = R2 == 0 ? 0.0 : 1.0 / sqrt(R2);            //   Green's function
	  matrixFile << G << " ";                               //   Write to matrix data file
	}                                                       //  End loop over source bodies
	matrixFile << std::endl;                                //  Line break
      }                                                         // End loop over target bodies
    }

    //! Print traversal statistics
    void printTraversalData() {
#if EXAFMM_COUNT_KERNEL
      if (logger::verbose) {                                    // If verbose flag is true
	std::cout << "--- Traversal stats --------------" << std::endl// Print title
		  << std::setw(logger::stringLength) << std::left //  Set format
		  << "P2P calls"  << " : "                      //  Print title
		  << std::setprecision(0) << std::fixed         //  Set format
		  << numP2P << std::endl                        //  Print number of P2P calls
		  << std::setw(logger::stringLength) << std::left //  Set format
		  << "M2L calls"  << " : "                      //  Print title
		  << std::setprecision(0) << std::fixed         //  Set format
		  << numM2L << std::endl;                       //  Print number of M2L calls
      }                                                         // End if for verbose flag
#endif
    }
#if EXAFMM_COUNT_LIST
    void writeList(Cells cells, int mpirank) {
      std::stringstream name;                                   // File name
      name << path << "list" << std::setfill('0') << std::setw(6) // Set format
	   << mpirank << ".dat";                                // Create file name for list
      std::ofstream listFile(name.str().c_str());               // Open list log file
      for (C_iter C=cells.begin(); C!=cells.end(); C++) {       // Loop over all lists
	listFile << std::setw(logger::stringLength) << std::left//  Set format
		 << C->ICELL << " " << C->numP2P << " " << C->numM2L << std::endl; // Print list size
      }                                                         // End loop over all lists
    }
#else
    void writeList(Cells, int) {}
#endif
  };
}
#endif
