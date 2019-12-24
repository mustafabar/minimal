#ifndef types_h
#define types_h
#include <assert.h>                                             // Some compilers don't have cassert
#include <complex>
#include <stdint.h>
#include <vector>
#include "vec.h"

namespace exafmm {
  // Basic type definitions
  typedef double real_t;                                        //!< Floating point type is double precision
  const real_t EPS = 1e-16;                                     //!< Double precision epsilon
  typedef std::complex<real_t> complex_t;                       //!< Complex type
  const complex_t I(0.,1.);                                     //!< Imaginary unit

  const int P = 12;                                             //!< Order of expansion
  const int NTERM = P*(P+1)/2;                                  //!< Number of expansion terms
  
  typedef vec<3,int> ivec3;                                     //!< Vector of 3 int types
  typedef vec<3,real_t> vec3;                                   //!< Vector of 3 real_t types
  typedef vec<4,real_t> vec4;                                   //!< Vector of 4 real_t types
  typedef vec<3,complex_t> cvec3;                               //!< Vector of 3 complex_t types
  typedef vec<NTERM,complex_t> cvecP;                           //!< Vector of NTERM complex_t types

  //! Range of indices
  struct Range {
    int begin;                                                  //!< Begin index
    int end;                                                    //!< End index
  };

  //! Wave structure for Ewald summation
  struct Wave {
    vec3   K;                                                 //!< 3-D wave number vector
    real_t REAL;                                              //!< real part of wave
    real_t IMAG;                                              //!< imaginary part of wave
  };
  typedef std::vector<Wave> Waves;                            //!< Vector of Wave types
  typedef typename Waves::iterator W_iter;                    //!< Iterator of Wave types
}
#endif
