#ifndef verify_h
#define verify_h
#include "timer.h"
#include "types.h"

namespace exafmm {
  //! Verify results
  class Verify {
    typedef std::map<uint64_t,double> Record;                   //!< Map of regression key value pair
    typedef Record::iterator R_iter;                            //!< Iterator of regression map

  public:
    bool verbose;                                               //!< Print to screen
    double average, average2;                                   //!< Average for regression

    //! Constructor
    Verify() : average(0), average2(0) {}

    //! Get sum of scalar component of a vector of target bodies
    double getSumScalar(std::vector<vec4> & bodies) {
      double v = 0;                                             // Initialize difference
      for (size_t b=0; b<bodies.size(); b++) {                  // Loop over bodies
	v += bodies[b][0];                                      //  Sum of scalar component for Laplace
      }                                                         // End loop over bodies
      return v;                                                 // Return difference
    }

    //! Get norm of scalar component of a vector of target bodies
    double getNrmVector(std::vector<vec4> & bodies) {
      double v = 0;                                             // Initialize norm
      for (size_t b=0; b<bodies.size(); b++) {                  // Loop over bodies
	v += std::abs(bodies[b][1] * bodies[b][1] +             //  Norm of vector x component
		      bodies[b][2] * bodies[b][2] +             //  Norm of vector y component
		      bodies[b][3] * bodies[b][3]);             //  Norm of vector z component
      }                                                         // End loop over bodies
      return v;                                                 // Return norm
    }

    //! Get difference between scalar component of two vectors of target bodies
    double getDifVector(std::vector<vec4> & bodies, std::vector<vec4> & bodies2) {
      double v = 0;                                             // Initialize difference
      for (size_t b=0; b<bodies.size(); b++) {                  // Loop over bodies
	v += std::abs((bodies[b][1] - bodies2[b][1]) *
                      (bodies[b][1] - bodies2[b][1]) +          //  Difference of vector x component
		      (bodies[b][2] - bodies2[b][2]) *
                      (bodies[b][2] - bodies2[b][2]) +          //  Difference of vector y component
		      (bodies[b][3] - bodies2[b][3]) *
                      (bodies[b][3] - bodies2[b][3]));          //  Difference of vector z component
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
