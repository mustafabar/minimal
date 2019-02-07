#ifndef ewald_h
#define ewald_h
#include "timer.h"
#include "types.h"

namespace exafmm {
  class Ewald {
  private:
    const int ksize;                                            //!< Number of waves in Ewald summation
    const real_t alpha;                                         //!< Scaling parameter for Ewald summation
    const real_t sigma;                                         //!< Scaling parameter for Ewald summation
    const real_t cutoff;                                        //!< Cutoff distance
    const real_t cycle;                                         //!< Periodic cycle

  public:
    //! Constructor
    Ewald(int _ksize, real_t _alpha, real_t _sigma, real_t _cutoff, real_t _cycle) :
      ksize(_ksize), alpha(_alpha), sigma(_sigma), cutoff(_cutoff), cycle(_cycle) {} // Initialize variables

    //! Forward DFT
    void dft(Waves & waves, std::vector<vec4> & Jbodies) const {
      vec3 scale;
      for (int d=0; d<3; d++) scale[d]= 2 * M_PI / cycle;       // Scale conversion
#pragma omp parallel for
      for (int w=0; w<int(waves.size()); w++) {                 // Loop over waves
	W_iter W=waves.begin()+w;                               //  Wave iterator
	W->REAL = W->IMAG = 0;                                  //  Initialize waves
	for (int b=0; b<int(Jbodies.size()); b++) {             //  Loop over bodies
	  real_t th = 0;                                        //   Initialize phase
	  for (int d=0; d<3; d++) th += W->K[d] * Jbodies[b][d] * scale[d];//  Determine phase
	  W->REAL += Jbodies[b][3] * std::cos(th);              //   Accumulate real component
	  W->IMAG += Jbodies[b][3] * std::sin(th);              //   Accumulate imaginary component
	}                                                       //  End loop over bodies
      }                                                         // End loop over waves
    }

    //! Inverse DFT
    void idft(Waves & waves, std::vector<vec4> & Ibodies, std::vector<vec4> & Jbodies) const {
      vec3 scale;
      for (int d=0; d<3; d++) scale[d] = 2 * M_PI / cycle;      // Scale conversion
#pragma omp parallel for
      for (int b=0; b<int(Ibodies.size()); b++) {               // Loop over bodies
	vec4 TRG = 0;                                           //  Initialize target values
	for (W_iter W=waves.begin(); W!=waves.end(); W++) {     //   Loop over waves
	  real_t th = 0;                                        //    Initialzie phase
	  for (int d=0; d<3; d++) th += W->K[d] * Jbodies[b][d] * scale[d];// Determine phase
	  real_t dtmp = W->REAL * std::sin(th) - W->IMAG * std::cos(th);// Temporary value
	  TRG[0]     += W->REAL * std::cos(th) + W->IMAG * std::sin(th);// Accumulate potential
	  for (int d=0; d<3; d++) TRG[d+1] -= dtmp * W->K[d];   //    Accumulate force
	}                                                       //   End loop over waves
	for (int d=0; d<3; d++) TRG[d+1] *= scale[d];           //   Scale forces
	Ibodies[b] += TRG;                                      //  Copy results to bodies
      }                                                         // End loop over bodies
    }

    //! Initialize wave vector
    Waves initWaves() const {
      Waves waves;                                              // Initialzie wave vector
      int kmaxsq = ksize * ksize;                               // kmax squared
      int kmax = ksize;                                         // kmax as integer
      for (int l=0; l<=kmax; l++) {                             // Loop over x component
	int mmin = -kmax;                                       //  Determine minimum y component
	if (l==0) mmin = 0;                                     //  Exception for minimum y component
	for (int m=mmin; m<=kmax; m++) {                        //  Loop over y component
	  int nmin = -kmax;                                     //   Determine minimum z component
	  if (l==0 && m==0) nmin=1;                             //   Exception for minimum z component
	  for (int n=nmin; n<=kmax; n++) {                      //   Loop over z component
	    real_t ksq = l * l + m * m + n * n;                 //    Wave number squared
	    if (ksq <= kmaxsq) {                                //    If wave number is below kmax
	      Wave wave;                                        //     Initialzie wave structure
	      wave.K[0] = l;                                    //     x component of k
	      wave.K[1] = m;                                    //     y component of k
	      wave.K[2] = n;                                    //     z component of k
	      wave.REAL = wave.IMAG = 0;                        //     Initialize amplitude
	      waves.push_back(wave);                            //     Push wave to vector
	    }                                                   //    End if for wave number
	  }                                                     //   End loop over z component
	}                                                       //  End loop over y component
      }                                                         // End loop over x component
      return waves;                                             // Return wave vector
    }

    //! Ewald wave part
    void wavePart(Waves &waves) {
      vec3 scale;
      for (int d=0; d<3; d++) scale[d] = 2 * M_PI / cycle;      // Scale conversion
      real_t coef = 2 / sigma / cycle / cycle / cycle;          // First constant
      real_t coef2 = 1 / (4 * alpha * alpha);                   // Second constant
      for (W_iter W=waves.begin(); W!=waves.end(); W++) {       // Loop over waves
	vec3 K = W->K * scale;                                  //  Wave number scaled
        real_t K2 = norm(K);                                    //  Wave number squared
	real_t factor = coef * std::exp(-K2 * coef2) / K2;      //  Wave factor
	W->REAL *= factor;                                      //  Apply wave factor to real part
	W->IMAG *= factor;                                      //  Apply wave factor to imaginary part
      }                                                         // End loop over waves
    }

    //! Subtract self term
    void selfTerm(std::vector<vec4> & Ibodies, std::vector<vec4> & Jbodies) {
      for (int b=0; b<int(Ibodies.size()); b++) {               // Loop over all bodies
	Ibodies[b][0] -= M_2_SQRTPI * Jbodies[b][3] * alpha;    //  Self term of Ewald real part
      }                                                         // End loop over all bodies in cell
    }

  };
}
#endif
