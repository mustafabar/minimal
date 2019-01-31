#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <omp.h>

#define EXAFMM_PP 6
const int P = 6;
const int NTERM = P*(P+1)/2;
const int DP2P = 1; // Use 1 for parallel
const int DM2L = 1; // Use 1 for parallel
const int MTERM = EXAFMM_PP*(EXAFMM_PP+1)*(EXAFMM_PP+2)/6;
const int LTERM = (EXAFMM_PP+1)*(EXAFMM_PP+2)*(EXAFMM_PP+3)/6;

#include "core.h"

#define for_3d for (int d=0; d<3; d++)
#define for_4d for (int d=0; d<4; d++)
#define for_m for (int m=0; m<MTERM; m++)
#define for_l for (int l=0; l<LTERM; l++)
#define EXAFMM_MAX(a,b) (((a) > (b)) ? (a) : (b))
#define EXAFMM_MIN(a,b) (((a) < (b)) ? (a) : (b))

namespace exafmm {
  class Kernel {
  private:
    std::vector<real_t> prefactor;
    std::vector<real_t> Anm;
    std::vector<complex_t> Cnm;
  public:
    static vec3 Xperiodic;
    ivec3 numPartition[10];
    int maxLevel;
    int maxGlobLevel;
    int numBodies;
    int numImages;
    int numCells;
    int numLeafs;
    int numGlobCells;
    int globLevelOffset[10];
    int numSendBodies;
    int numSendCells;
    int numSendLeafs;
    int MPISIZE;
    int MPIRANK;

    vec3 X0;
    real_t R0;
    vec3 RGlob;
    int *Index;
    int *Rank;
    int *sendIndex;
    int *recvIndex;
    int (*Leafs)[2];
    int (*sendLeafs)[2];
    int (*recvLeafs)[2];
    vec4 *Ibodies;
    vec4 *Jbodies;
    real_t (*Multipole)[MTERM];
    real_t (*Local)[LTERM];
    real_t (*globMultipole)[MTERM];
    real_t (*globLocal)[LTERM];
    vec4 *sendJbodies;
    vec4 *recvJbodies;
    float (*sendMultipole)[MTERM];
    float (*recvMultipole)[MTERM];

  private:
    inline int oddOrEven(int n) {
      return (((n) & 1) == 1) ? -1 : 1;
    }

    void cart2sph(vec3 dX, real_t & r, real_t & theta, real_t & phi) {
      r = sqrt(norm(dX));
      theta = r == 0 ? 0 : acos(dX[2] / r);
      phi = atan2(dX[1], dX[0]);
    }

    void sph2cart(real_t r, real_t theta, real_t phi, vec3 spherical, vec3 & cartesian) {
      cartesian[0] = std::sin(theta) * std::cos(phi) * spherical[0]
        + std::cos(theta) * std::cos(phi) / r * spherical[1]
        - std::sin(phi) / r / std::sin(theta) * spherical[2];
      cartesian[1] = std::sin(theta) * std::sin(phi) * spherical[0]
        + std::cos(theta) * std::sin(phi) / r * spherical[1]
        + std::cos(phi) / r / std::sin(theta) * spherical[2];
      cartesian[2] = std::cos(theta) * spherical[0]
        - std::sin(theta) / r * spherical[1];
    }

    void evalMultipole(real_t rho, real_t alpha, real_t beta, complex_t * Ynm, complex_t * YnmTheta) {
      real_t x = std::cos(alpha);
      real_t y = std::sin(alpha);
      real_t fact = 1;
      real_t pn = 1;
      real_t rhom = 1;
      for (int m=0; m<P; m++) {
        complex_t eim = std::exp(I * real_t(m * beta));
        real_t p = pn;
        int npn = m * m + 2 * m;
        int nmn = m * m;
        Ynm[npn] = rhom * p * prefactor[npn] * eim;
        Ynm[nmn] = std::conj(Ynm[npn]);
        real_t p1 = p;
        p = x * (2 * m + 1) * p1;
        YnmTheta[npn] = rhom * (p - (m + 1) * x * p1) / y * prefactor[npn] * eim;
        rhom *= rho;
        real_t rhon = rhom;
        for (int n=m+1; n<P; n++) {
          int npm = n * n + n + m;
          int nmm = n * n + n - m;
          Ynm[npm] = rhon * p * prefactor[npm] * eim;
          Ynm[nmm] = std::conj(Ynm[npm]);
          real_t p2 = p1;
          p1 = p;
          p = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
          YnmTheta[npm] = rhon * ((n - m + 1) * p - (n + 1) * x * p1) / y * prefactor[npm] * eim;
          rhon *= rho;
        }
        pn = -pn * fact * y;
        fact += 2;
      }
    }

    void evalLocal(real_t rho, real_t alpha, real_t beta, complex_t * Ynm2) {
      real_t x = std::cos(alpha);
      real_t y = std::sin(alpha);
      real_t fact = 1;
      real_t pn = 1;
      real_t rhom = 1.0 / rho;
      for (int m=0; m<2*P; m++) {
        complex_t eim = std::exp(I * real_t(m * beta));
        real_t p = pn;
        int npn = m * m + 2 * m;
        int nmn = m * m;
        Ynm2[npn] = rhom * p * prefactor[npn] * eim;
        Ynm2[nmn] = std::conj(Ynm2[npn]);
        real_t p1 = p;
        p = x * (2 * m + 1) * p1;
        rhom /= rho;
        real_t rhon = rhom;
        for (int n=m+1; n<2*P; n++) {
          int npm = n * n + n + m;
          int nmm = n * n + n - m;
          Ynm2[npm] = rhon * p * prefactor[npm] * eim;
          Ynm2[nmm] = std::conj(Ynm2[npm]);
          real_t p2 = p1;
          p1 = p;
          p = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
          rhon /= rho;
        }
        pn = -pn * fact * y;
        fact += 2;
      }
    }
    
  protected:
    inline int getGlobKey(int *iX, int level) const {
      return iX[0] + (iX[1] + iX[2] * numPartition[level][1]) * numPartition[level][0];
    }
 
    inline void getIndex2(ivec3 &iX, int index) const {
      iX = 0;
      int d = 0, level = 0;
      while (index != 0) {
	iX[d] += (index % 2) * (1 << level);
	index >>= 1;
	d = (d+1) % 3;
	if (d == 0) level++;
      }
    }

    void getCenter(vec3 &dX, int index, int level) const {
      real_t R = R0 / (1 << level);
      ivec3 iX = 0;
      getIndex2(iX, index);
      for_3d dX[d] = X0[d] - R0 + (2 * iX[d] + 1) * R;
    }

    void P2P(int ibegin, int iend, int jbegin, int jend, vec3 periodic) const {
      for (int i=ibegin; i<iend; i++) {
	real_t Po = 0, Fx = 0, Fy = 0, Fz = 0;
	for (int j=jbegin; j<jend; j++) {
	  vec3 dX;
	  for_3d dX[d] = Jbodies[i][d] - Jbodies[j][d] - periodic[d];
	  real_t R2 = norm(dX);
	  real_t invR2 = 1.0 / R2;
	  if (R2 == 0) invR2 = 0;
	  real_t invR = Jbodies[j][3] * sqrt(invR2);
	  real_t invR3 = invR2 * invR;
	  Po += invR;
	  Fx += dX[0] * invR3;
	  Fy += dX[1] * invR3;
	  Fz += dX[2] * invR3;
	}
	Ibodies[i][0] += Po;
	Ibodies[i][1] -= Fx;
	Ibodies[i][2] -= Fy;
	Ibodies[i][3] -= Fz;
      }
    }

    void P2M(vec3 dX, real_t SRC, real_t *Mj) const {
      real_t M[MTERM];
      M[0] = SRC;
      powerM(M,dX);
      for_m Mj[m] += M[m];
    }

    void M2M() const {
    }

    void M2L() const {
    }

    void L2L() const {
    }

    void L2P() const {
    }

    void P2P() const {
    }

  public:
    Kernel() : MPISIZE(1), MPIRANK(0) {}
    ~Kernel() {}

    inline int getKey(int *iX, int level, bool levelOffset=true) const {
      int id = 0;
      if (levelOffset) id = ((1 << 3 * level) - 1) / 7;
      for(int lev=0; lev<level; ++lev ) {
	for_3d id += iX[d] % 2 << (3 * lev + d);
	for_3d iX[d] >>= 1;
      }
      return id;
    }

    inline void getGlobIndex(int *iX, int index, int level) const {
      iX[0] = index % numPartition[level][0];
      iX[1] = index / numPartition[level][0] % numPartition[level][1];
      iX[2] = index / numPartition[level][0] / numPartition[level][1];
    }
  };
}
