#ifndef verify_h
#define verify_h
#include "timer.h"
#include "types.h"

namespace exafmm {
  //! Verify results
  class Verify {
    typedef std::map<uint64_t,double> Record;                   //!< Map of regression key value pair
    typedef Record::iterator R_iter;                            //!< Iterator of regression map

  private:
    const char * path;                                          //!< Path to save files

  public:
    bool verbose;                                               //!< Print to screen
    double average, average2;                                   //!< Average for regression

    //! Constructor
    Verify() : path("./"), average(0), average2(0) {}

    //! Constructor with argument
    Verify(const char * _path) : path(_path), average(0), average2(0) {}

    //! Get sum of scalar component of a vector of target bodies
    double getSumScalar(Bodies & bodies) {
      double v = 0;                                             // Initialize difference
      for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {     // Loop over bodies
	v += B->TRG[0] * B->SRC;                                //  Sum of scalar component for Laplace
      }                                                         // End loop over bodies
      return v;                                                 // Return difference
    }

    //! Get norm of scalar component of a vector of target bodies
    double getNrmScalar(Bodies & bodies) {
      double v = 0;                                             // Initialize norm
      for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {     // Loop over bodies
	v += std::abs(B->TRG[0] * B->TRG[0]);                   //  Norm of scalar component
      }                                                         // End loop over bodies
      return v;                                                 // Return norm
    }

    //! Get difference between scalar component of two vectors of target bodies
    double getDifScalar(Bodies & bodies, Bodies & bodies2) {
      double v = 0;                                             // Initialize difference
      B_iter B2 = bodies2.begin();                              // Set iterator of bodies2
      for (B_iter B=bodies.begin(); B!=bodies.end(); B++, B2++) { // Loop over bodies & bodies2
	v += std::abs((B->TRG[0] - B2->TRG[0]) * (B->TRG[0] - B2->TRG[0])); //  Difference of scalar component
      }                                                         // End loop over bodies & bodies2
      return v;                                                 // Return difference
    }

    //! Get difference between scalar component of two vectors of target bodies
    double getRelScalar(Bodies & bodies, Bodies & bodies2) {
      double v = 0;                                             // Initialize difference
      B_iter B2 = bodies2.begin();                              // Set iterator of bodies2
      for (B_iter B=bodies.begin(); B!=bodies.end(); B++, B2++) { // Loop over bodies & bodies2
	v += std::abs(((B->TRG[0] - B2->TRG[0]) * (B->TRG[0] - B2->TRG[0]))
		      / (B2->TRG[0] * B2->TRG[0]));             //  Difference of scalar component
      }                                                         // End loop over bodies & bodies2
      return v;                                                 // Return difference
    }

    //! Get norm of scalar component of a vector of target bodies
    double getNrmVector(Bodies & bodies) {
      double v = 0;                                             // Initialize norm
      for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {     // Loop over bodies
	v += std::abs(B->TRG[1] * B->TRG[1] +                   //  Norm of vector x component
		      B->TRG[2] * B->TRG[2] +                   //  Norm of vector y component
		      B->TRG[3] * B->TRG[3]);                   //  Norm of vector z component
      }                                                         // End loop over bodies
      return v;                                                 // Return norm
    }

    //! Get difference between scalar component of two vectors of target bodies
    double getDifVector(Bodies & bodies, Bodies & bodies2) {
      double v = 0;                                             // Initialize difference
      B_iter B2 = bodies2.begin();                              // Set iterator of bodies2
      for (B_iter B=bodies.begin(); B!=bodies.end(); B++, B2++) { // Loop over bodies & bodies2
	v += std::abs((B->TRG[1] - B2->TRG[1]) * (B->TRG[1] - B2->TRG[1]) + //  Difference of vector x component
		      (B->TRG[2] - B2->TRG[2]) * (B->TRG[2] - B2->TRG[2]) + //  Difference of vector y component
		      (B->TRG[3] - B2->TRG[3]) * (B->TRG[3] - B2->TRG[3])); //  Difference of vector z component
      }                                                         // End loop over bodies & bodies2
      return v;                                                 // Return difference
    }

    //! Get difference between scalar component of two vectors of target bodies
    double getRelVector(Bodies & bodies, Bodies & bodies2) {
      double v = 0;                                             // Initialize difference
      B_iter B2 = bodies2.begin();                              // Set iterator of bodies2
      for (B_iter B=bodies.begin(); B!=bodies.end(); B++, B2++) { // Loop over bodies & bodies2
	v += std::abs(((B->TRG[1] - B2->TRG[1]) * (B->TRG[1] - B2->TRG[1]) +//  Difference of vector x component
		       (B->TRG[2] - B2->TRG[2]) * (B->TRG[2] - B2->TRG[2]) +//  Difference of vector y component
		       (B->TRG[3] - B2->TRG[3]) * (B->TRG[3] - B2->TRG[3]))//   Difference of vector z component
		      / (B2->TRG[1] * B2->TRG[1] +              //  Norm of vector x component
			 B2->TRG[2] * B2->TRG[2] +              //  Norm of vector y component
			 B2->TRG[3] * B2->TRG[3]));             //  Norm of vector z component
      }                                                         // End loop over bodies & bodies2
      return v;                                                 // Return difference
    }

    //! Print relative L2 norm scalar error
    void print(std::string title, double v) {
      if (VERBOSE) {                                            // If verbose flag is true
	std::cout << std::setw(stringLength) << std::left       //  Set format
		  << title << " : " << std::setprecision(decimal) << std::scientific // Set title
		  << v << std::endl;                            //  Print potential error
      }                                                         // End if for verbose flag
    }
  };
}
#endif
