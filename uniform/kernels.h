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

#define for_3d for (int d=0; d<3; d++)
#define for_4d for (int d=0; d<4; d++)
#define for_m for (int m=0; m<NTERM; m++)
#define for_l for (int l=0; l<NTERM; l++)
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
    complex_t (*Multipole)[NTERM];
    complex_t (*Local)[NTERM];
    complex_t (*globMultipole)[NTERM];
    complex_t (*globLocal)[NTERM];
    vec4 *sendJbodies;
    vec4 *recvJbodies;
    fcomplex_t (*sendMultipole)[NTERM];
    fcomplex_t (*recvMultipole)[NTERM];

  private:
    inline int oddOrEven(int n) const {
      return (((n) & 1) == 1) ? -1 : 1;
    }

    void cart2sph(vec3 dX, real_t & r, real_t & theta, real_t & phi) const {
      r = sqrt(norm(dX));
      theta = r == 0 ? 0 : acos(dX[2] / r);
      phi = atan2(dX[1], dX[0]);
    }

    void sph2cart(real_t r, real_t theta, real_t phi, vec3 spherical, vec3 & cartesian) const {
      cartesian[0] = std::sin(theta) * std::cos(phi) * spherical[0]
        + std::cos(theta) * std::cos(phi) / r * spherical[1]
        - std::sin(phi) / r / std::sin(theta) * spherical[2];
      cartesian[1] = std::sin(theta) * std::sin(phi) * spherical[0]
        + std::cos(theta) * std::sin(phi) / r * spherical[1]
        + std::cos(phi) / r / std::sin(theta) * spherical[2];
      cartesian[2] = std::cos(theta) * spherical[0]
        - std::sin(theta) / r * spherical[1];
    }

    void evalMultipole(real_t rho, real_t alpha, real_t beta, complex_t * Ynm, complex_t * YnmTheta) const {
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

    void evalLocal(real_t rho, real_t alpha, real_t beta, complex_t * Ynm2) const {
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
    void P2M(vec3 dX, real_t SRC, complex_t *Mj) const {
      complex_t Ynm[P*P], YnmTheta[P*P];
      real_t rho, alpha, beta;
      cart2sph(dX, rho, alpha, beta);
      evalMultipole(rho, alpha, -beta, Ynm, YnmTheta);
      for (int n=0; n<P; n++) {
        for (int m=0; m<=n; m++) {
          int nm  = n * n + n + m;
          int nms = n * (n + 1) / 2 + m;
          Mj[nms] += SRC * Ynm[nm];
        }
      }
    }

    void M2M(vec3 dX, complex_t *Mc, complex_t *Mp) const {
      complex_t Ynm[P*P], YnmTheta[P*P];
      real_t rho, alpha, beta;
      cart2sph(dX, rho, alpha, beta);
      evalMultipole(rho, alpha, -beta, Ynm, YnmTheta);
      for (int j=0; j<P; j++) {
        for (int k=0; k<=j; k++) {
          int jk = j * j + j + k;
          int jks = j * (j + 1) / 2 + k;
          complex_t M = 0;
          for (int n=0; n<=j; n++) {
            for (int m=-n; m<=std::min(k-1,n); m++) {
              if (j-n >= k-m) {
                int jnkm  = (j - n) * (j - n) + j - n + k - m;
                int jnkms = (j - n) * (j - n + 1) / 2 + k - m;
                int nm    = n * n + n + m;
                M += Mc[jnkms] * std::pow(I,real_t(m-abs(m))) * Ynm[nm]
                  * real_t(oddOrEven(n) * Anm[nm] * Anm[jnkm] / Anm[jk]);
              }
            }
            for (int m=k; m<=n; m++) {
              if (j-n >= m-k) {
                int jnkm  = (j - n) * (j - n) + j - n + k - m;
                int jnkms = (j - n) * (j - n + 1) / 2 - k + m;
                int nm    = n * n + n + m;
                M += std::conj(Mc[jnkms]) * Ynm[nm]
                  * real_t(oddOrEven(k+n+m) * Anm[nm] * Anm[jnkm] / Anm[jk]);
              }
            } 
          }
          Mp[jks] += M * EPS;
        }
      }
    }

    void M2L(vec3 dX, complex_t *M, complex_t *L) const {
      complex_t Ynm2[4*P*P];
      real_t rho, alpha, beta;
      cart2sph(dX, rho, alpha, beta);
      evalLocal(rho, alpha, beta, Ynm2);
      for (int j=0; j<P; j++) {
        for (int k=0; k<=j; k++) {
          int jk = j * j + j + k;
          int jks = j * (j + 1) / 2 + k;
          complex_t Ljk = 0;
          for (int n=0; n<P; n++) {
            for (int m=-n; m<0; m++) {
              int nm   = n * n + n + m;
              int nms  = n * (n + 1) / 2 - m;
              int jknm = jk * P * P + nm;
              int jnkm = (j + n) * (j + n) + j + n + m - k;
              Ljk += std::conj(M[nms]) * Cnm[jknm] * Ynm2[jnkm];
            }
            for (int m=0; m<=n; m++) {
              int nm   = n * n + n + m;
              int nms  = n * (n + 1) / 2 + m;
              int jknm = jk * P * P + nm;
              int jnkm = (j + n) * (j + n) + j + n + m - k;
              Ljk += M[nms] * Cnm[jknm] * Ynm2[jnkm];
            }
          }
          L[jks] += Ljk;
        }
      }
    }

    void L2L(vec3 dX, complex_t *Lp, complex_t *Lc) const {
      complex_t Ynm[P*P], YnmTheta[P*P];
      real_t rho, alpha, beta;
      cart2sph(dX, rho, alpha, beta);
      evalMultipole(rho, alpha, beta, Ynm, YnmTheta);
      for (int j=0; j<P; j++) {
    	for (int k=0; k<=j; k++) {
          int jk = j * j + j + k;
          int jks = j * (j + 1) / 2 + k;
          complex_t L = 0;
          for (int n=j; n<P; n++) {
            for (int m=j+k-n; m<0; m++) {
              int jnkm = (n - j) * (n - j) + n - j + m - k;
              int nm   = n * n + n - m;
              int nms  = n * (n + 1) / 2 - m;
              L += std::conj(Lp[nms]) * Ynm[jnkm]
                * real_t(oddOrEven(k) * Anm[jnkm] * Anm[jk] / Anm[nm]);
            }
            for (int m=0; m<=n; m++) {
              if (n-j >= abs(m-k)) {
                int jnkm = (n - j) * (n - j) + n - j + m - k;
                int nm   = n * n + n + m;
                int nms  = n * (n + 1) / 2 + m;
                L += Lp[nms] * std::pow(I,real_t(m-k-abs(m-k)))
                  * Ynm[jnkm] * Anm[jnkm] * Anm[jk] / Anm[nm];
              }
            }
          }
          Lc[jks] += L * EPS;
        }
      }
    }

    void L2P(vec3 dX, complex_t *L, vec4 &TRG) const {
      complex_t Ynm[P*P], YnmTheta[P*P];
      vec3 spherical = 0;
      vec3 cartesian = 0;
      real_t r, theta, phi;
      cart2sph(dX, r, theta, phi);
      evalMultipole(r, theta, phi, Ynm, YnmTheta);
      for (int n=0; n<P; n++) {
        int nm  = n * n + n;
        int nms = n * (n + 1) / 2;
        TRG[0] += std::real(L[nms] * Ynm[nm]);
        spherical[0] += std::real(L[nms] * Ynm[nm]) / r * n;
        spherical[1] += std::real(L[nms] * YnmTheta[nm]);
        for (int m=1; m<=n; m++) {
          nm  = n * n + n + m;
          nms = n * (n + 1) / 2 + m;
          TRG[0] += 2 * std::real(L[nms] * Ynm[nm]);
          spherical[0] += 2 * std::real(L[nms] * Ynm[nm]) / r * n;
          spherical[1] += 2 * std::real(L[nms] * YnmTheta[nm]);
          spherical[2] += 2 * std::real(L[nms] * Ynm[nm] * I) * m;
        }
      }
      sph2cart(r, theta, phi, spherical, cartesian);
      TRG[1] += cartesian[0];
      TRG[2] += cartesian[1];
      TRG[3] += cartesian[2];
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

  public:
    Kernel() : MPISIZE(1), MPIRANK(0) {
      prefactor.resize(4*P*P);
      Anm.resize(4*P*P);
      Cnm.resize(P*P*P*P);
      for (int n=0; n<2*P; n++) {
        for (int m=-n; m<=n; m++) {
          int nm = n*n+n+m;
          int nabsm = abs(m);
          real_t fnmm = EPS;
          for (int i=1; i<=n-m; i++) fnmm *= i;
          real_t fnpm = EPS;
          for (int i=1; i<=n+m; i++) fnpm *= i;
          real_t fnma = 1.0;
          for (int i=1; i<=n-nabsm; i++) fnma *= i;
          real_t fnpa = 1.0;
          for (int i=1; i<=n+nabsm; i++) fnpa *= i;
          prefactor[nm] = std::sqrt(fnma/fnpa);
          Anm[nm] = oddOrEven(n)/std::sqrt(fnmm*fnpm);
        }
      }
      for (int j=0, jk=0, jknm=0; j<P; j++) {
        for (int k=-j; k<=j; k++, jk++) {
          for (int n=0, nm=0; n<P; n++) {
            for (int m=-n; m<=n; m++, nm++, jknm++) {
              const int jnkm = (j+n)*(j+n)+j+n+m-k;
              Cnm[jknm] = std::pow(I,real_t(abs(k-m)-abs(k)-abs(m)))
                * real_t(oddOrEven(j)*Anm[nm]*Anm[jk]/Anm[jnkm]) * EPS;
            }
          }
        }
      }
    }
    ~Kernel() {}

  };
}
