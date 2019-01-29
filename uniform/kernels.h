#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "namespace.h"
#include <omp.h>

#define EXAFMM_PP 6
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

namespace EXAFMM_NAMESPACE {
  class UniformKernel {
  private:
    std::vector<real_t> prefactor;
    std::vector<real_t> Anm;
    std::vector<complex_t> Cnm;
  public:
    const int P;
    const int NTERM;
    static vec3 Xperiodic;
    int maxLevel;
    int maxGlobLevel;
    int numBodies;
    int numImages;
    int numCells;
    int numLeafs;
    int numGlobCells;
    int numPartition[10][3];
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
    inline void getIndex(int *iX, int index) const {
      for_3d iX[d] = 0;
      int d = 0, level = 0;
      while (index != 0) {
	iX[d] += (index % 2) * (1 << level);
	index >>= 1;
	d = (d+1) % 3;
	if (d == 0) level++;
      }
    }

    void getCenter(real_t *dX, int index, int level) const {
      real_t R = R0 / (1 << level);
      ivec3 iX = 0;
      getIndex(iX, index);
      for_3d dX[d] = X0[d] - R0 + (2 * iX[d] + 1) * R;
    }

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

    void P2P() const {
      int iXc[3];
      getGlobIndex(iXc,MPIRANK,maxGlobLevel);
      int nunit = 1 << maxLevel;
      int nunitGlob[3];
      for_3d nunitGlob[d] = nunit * numPartition[maxGlobLevel][d];
      int nxmin[3], nxmax[3];
      for_3d nxmin[d] = -iXc[d] * nunit;
      for_3d nxmax[d] = nunitGlob[d] + nxmin[d] - 1;
      if (numImages != 0) {
	for_3d nxmin[d] -= nunitGlob[d];
	for_3d nxmax[d] += nunitGlob[d];
      }
#pragma omp parallel for
      for (int i=0; i<numLeafs; i++) {
	ivec3 iX = 0;
	getIndex(iX,i);
	int jxmin[3], jxmax[3];
	for_3d jxmin[d] = std::max(nxmin[d],iX[d] - DP2P);
	for_3d jxmax[d] = std::min(nxmax[d],iX[d] + DP2P);
	int jx[3];
	for (jx[2]=jxmin[2]; jx[2]<=jxmax[2]; jx[2]++) {
	  for (jx[1]=jxmin[1]; jx[1]<=jxmax[1]; jx[1]++) {
	    for (jx[0]=jxmin[0]; jx[0]<=jxmax[0]; jx[0]++) {
	      int jxp[3];
	      for_3d jxp[d] = (jx[d] + nunit) % nunit;
	      int j = getKey(jxp,maxLevel,false);
	      for_3d jxp[d] = (jx[d] + nunit) / nunit;
#if EXAFMM_SERIAL
	      int rankOffset = 13 * numLeafs;
#else
	      int rankOffset = (jxp[0] + 3 * jxp[1] + 9 * jxp[2]) * numLeafs;
#endif
	      j += rankOffset;
	      rankOffset = 13 * numLeafs;
	      vec3 periodic = 0;
	      for_3d jxp[d] = (jx[d] + iXc[d] * nunit + nunitGlob[d]) / nunitGlob[d];
	      for_3d periodic[d] = (jxp[d] - 1) * 2 * RGlob[d];
	      P2P(Leafs[i+rankOffset][0],Leafs[i+rankOffset][1],Leafs[j][0],Leafs[j][1],periodic);
	    }
	  }
	}
      }
    }

    void P2M() const {
      int rankOffset = 13 * numLeafs;
      int levelOffset = ((1 << 3 * maxLevel) - 1) / 7 + 13 * numCells;
#pragma omp parallel for
      for (int i=0; i<numLeafs; i++) {
	real_t center[3];
	getCenter(center,i,maxLevel);
	for (int j=Leafs[i+rankOffset][0]; j<Leafs[i+rankOffset][1]; j++) {
	  real_t dX[3];
	  for_3d dX[d] = center[d] - Jbodies[j][d];
	  real_t M[MTERM];
	  M[0] = Jbodies[j][3];
	  powerM(M,dX);
	  for_m Multipole[i+levelOffset][m] += M[m];
	}
      }
    }

    void M2M() const {
      int rankOffset = 13 * numCells;
      for (int lev=maxLevel; lev>0; lev--) {
	int childOffset = ((1 << 3 * lev) - 1) / 7 + rankOffset;
	int parentOffset = ((1 << 3 * (lev - 1)) - 1) / 7 + rankOffset;
	real_t radius = R0 / (1 << lev);
#pragma omp parallel for schedule(static, 8)
	for (int i=0; i<(1 << 3 * lev); i++) {
	  int c = i + childOffset;
	  int p = (i >> 3) + parentOffset;
	  int iX[3];
	  iX[0] = 1 - (i & 1) * 2;
	  iX[1] = 1 - ((i / 2) & 1) * 2;
	  iX[2] = 1 - ((i / 4) & 1) * 2;
	  real_t dX[3];
	  for_3d dX[d] = iX[d] * radius;
	  real_t M[MTERM];
	  real_t C[LTERM];
	  C[0] = 1;
	  powerM(C,dX);
	  for_m M[m] = Multipole[c][m];
	  for_m Multipole[p][m] += C[m] * M[0];
	  M2MSum(Multipole[p],C,M);
	}
      }
    }

    void M2L() const {
      int iXc[3];
      int DM2LC = DM2L;
      getGlobIndex(iXc,MPIRANK,maxGlobLevel);
      for (int lev=1; lev<=maxLevel; lev++) {
	if (lev==maxLevel) DM2LC = DP2P;
	int levelOffset = ((1 << 3 * lev) - 1) / 7;
	int nunit = 1 << lev;
	int nunitGlob[3];
	for_3d nunitGlob[d] = nunit * numPartition[maxGlobLevel][d];
	int nxmin[3], nxmax[3];
	for_3d nxmin[d] = -iXc[d] * (nunit >> 1);
	for_3d nxmax[d] = (nunitGlob[d] >> 1) + nxmin[d] - 1;
	if (numImages != 0) {
	  for_3d nxmin[d] -= (nunitGlob[d] >> 1);
	  for_3d nxmax[d] += (nunitGlob[d] >> 1);
	}
	real_t diameter = 2 * R0 / (1 << lev);
#pragma omp parallel for
	for (int i=0; i<(1 << 3 * lev); i++) {
	  real_t L[LTERM];
	  for_l L[l] = 0;
	  int iX[3] = {0, 0, 0};
	  getIndex(iX,i);
	  int jxmin[3];
	  for_3d jxmin[d] = (std::max(nxmin[d],(iX[d] >> 1) - DM2L) << 1);
	  int jxmax[3];
	  for_3d jxmax[d] = (std::min(nxmax[d],(iX[d] >> 1) + DM2L) << 1) + 1;
	  int jx[3];
	  for (jx[2]=jxmin[2]; jx[2]<=jxmax[2]; jx[2]++) {
	    for (jx[1]=jxmin[1]; jx[1]<=jxmax[1]; jx[1]++) {
	      for (jx[0]=jxmin[0]; jx[0]<=jxmax[0]; jx[0]++) {
		if(jx[0] < iX[0]-DM2LC || iX[0]+DM2LC < jx[0] ||
		   jx[1] < iX[1]-DM2LC || iX[1]+DM2LC < jx[1] ||
		   jx[2] < iX[2]-DM2LC || iX[2]+DM2LC < jx[2]) {
		  int jxp[3];
		  for_3d jxp[d] = (jx[d] + nunit) % nunit;
		  int j = getKey(jxp,lev);
		  for_3d jxp[d] = (jx[d] + nunit) / nunit;
#if EXAFMM_SERIAL
		  int rankOffset = 13 * numCells;
#else
		  int rankOffset = (jxp[0] + 3 * jxp[1] + 9 * jxp[2]) * numCells;
#endif
		  j += rankOffset;
		  real_t M[MTERM];
		  for_m M[m] = Multipole[j][m];
		  real_t dX[3];
		  for_3d dX[d] = (iX[d] - jx[d]) * diameter;
		  real_t invR2 = 1. / (dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2]);
		  real_t invR  = sqrt(invR2);
		  real_t C[LTERM];
		  getCoef(C,dX,invR2,invR);
		  M2LSum(L,C,M);
		}
	      }
	    }
	  }
	  for_l Local[i+levelOffset][l] += L[l];
	}
      }
    }

    void L2L() const {
      for (int lev=1; lev<=maxLevel; lev++) {
	int childOffset = ((1 << 3 * lev) - 1) / 7;
	int parentOffset = ((1 << 3 * (lev - 1)) - 1) / 7;
	real_t radius = R0 / (1 << lev);
#pragma omp parallel for
	for (int i=0; i<(1 << 3 * lev); i++) {
	  int c = i + childOffset;
	  int p = (i >> 3) + parentOffset;
	  int iX[3];
	  iX[0] = (i & 1) * 2 - 1;
	  iX[1] = ((i / 2) & 1) * 2 - 1;
	  iX[2] = ((i / 4) & 1) * 2 - 1;
	  real_t dX[3];
	  for_3d dX[d] = iX[d] * radius;
	  real_t C[LTERM];
	  C[0] = 1;
	  powerL(C,dX);
	  for_l Local[c][l] += Local[p][l];
	  for (int l=1; l<LTERM; l++) Local[c][0] += C[l] * Local[p][l];
	  L2LSum(Local[c],C,Local[p]);
	}
      }
    }

    void L2P() const {
      int rankOffset = 13 * numLeafs;
      int levelOffset = ((1 << 3 * maxLevel) - 1) / 7;
#pragma omp parallel for
      for (int i=0; i<numLeafs; i++) {
	real_t center[3];
	getCenter(center,i,maxLevel);
	real_t L[LTERM];
	for_l L[l] = Local[i+levelOffset][l];
	for (int j=Leafs[i+rankOffset][0]; j<Leafs[i+rankOffset][1]; j++) {
	  real_t dX[3];
	  for_3d dX[d] = Jbodies[j][d] - center[d];
	  real_t C[LTERM];
	  C[0] = 1;
	  powerL(C,dX);
	  for_4d Ibodies[j][d] += L[d];
	  for (int l=1; l<LTERM; l++) Ibodies[j][0] += C[l] * L[l];
	  L2PSum(Ibodies[j],C,L);
	}
      }
    }

  public:
    UniformKernel(int _P) : P(_P), NTERM(P*(P+1)/2), MPISIZE(1), MPIRANK(0) {}
    ~UniformKernel() {}

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
