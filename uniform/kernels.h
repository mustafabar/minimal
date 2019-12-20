#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <omp.h>

#define for_3d for (int d=0; d<3; d++)
#define for_4d for (int d=0; d<4; d++)
#define for_m for (int m=0; m<NTERM; m++)

namespace exafmm {
  class Kernel {
  private:
    std::vector<real_t> prefactor;
    std::vector<real_t> Anm;
    std::vector<complex_t> Cnm;

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

    inline real_t weight(vec3 dX, real_t R) const {
      real_t D = 0.2 * R;
      vec3 X;
      for_3d X[d] = std::max(std::min(R - std::abs(dX[d]), D), -D) / D;
      real_t w = 1;
      for_3d w *= (2 + 3 * X[d] - X[d] * X[d] * X[d]) / 4;
      return w;
    }

  public:
    void P2M(vec3 dX, real_t R, real_t SRC, complex_t *Mj) const {
      complex_t Ynm[P*P], YnmTheta[P*P];
      real_t rho, alpha, beta;
      cart2sph(dX, rho, alpha, beta);
      evalMultipole(rho, alpha, -beta, Ynm, YnmTheta);
      real_t w = weight(dX, R);
      for (int n=0; n<P; n++) {
        for (int m=0; m<=n; m++) {
          int nm  = n * n + n + m;
          int nms = n * (n + 1) / 2 + m;
          Mj[nms] += w * SRC * Ynm[nm];
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

    void L2P(vec3 dX, real_t R, complex_t *L, vec4 &TRG) const {
      complex_t Ynm[P*P], YnmTheta[P*P];
      vec3 spherical = 0;
      vec3 cartesian = 0;
      real_t r, theta, phi;
      cart2sph(dX, r, theta, phi);
      evalMultipole(r, theta, phi, Ynm, YnmTheta);
      real_t w = weight(dX, R);
      for (int n=0; n<P; n++) {
        int nm  = n * n + n;
        int nms = n * (n + 1) / 2;
        TRG[0] += w * std::real(L[nms] * Ynm[nm]);
        spherical[0] += w * std::real(L[nms] * Ynm[nm]) / r * n;
        spherical[1] += w * std::real(L[nms] * YnmTheta[nm]);
        for (int m=1; m<=n; m++) {
          nm  = n * n + n + m;
          nms = n * (n + 1) / 2 + m;
          TRG[0] += 2 * w * std::real(L[nms] * Ynm[nm]);
          spherical[0] += 2 * w * std::real(L[nms] * Ynm[nm]) / r * n;
          spherical[1] += 2 * w * std::real(L[nms] * YnmTheta[nm]);
          spherical[2] += 2 * w * std::real(L[nms] * Ynm[nm] * I) * m;
        }
      }
      sph2cart(r, theta, phi, spherical, cartesian);
      TRG[1] += cartesian[0];
      TRG[2] += cartesian[1];
      TRG[3] += cartesian[2];
    }

    void P2P(std::vector<vec4> &Ibodies, int ibegin, int iend, vec3 Xi,
             std::vector<vec4> &Jbodies, int jbegin, int jend, vec3 Xj, real_t R, vec3 periodic) const {
      vec3 dX;
      for (int i=ibegin; i<iend; i++) {
        vec4 TRG = 0;
        for_3d dX[d] = Jbodies[i][d] - Xi[d];
        real_t wi = weight(dX, R);
	for (int j=jbegin; j<jend; j++) {
          for_3d dX[d] = Jbodies[j][d] + periodic[d] - Xj[d];
          real_t wj = weight(dX, R);
	  for_3d dX[d] = Jbodies[i][d] - Jbodies[j][d] - periodic[d];
	  real_t R2 = norm(dX);
	  real_t invR2 = 1.0 / R2;
	  if (R2 == 0) invR2 = 0;
	  real_t invR = Jbodies[j][3] * sqrt(invR2) * wj;
	  real_t invR3 = invR2 * invR;
	  TRG[0] += invR;
	  TRG[1] -= dX[0] * invR3;
	  TRG[2] -= dX[1] * invR3;
	  TRG[3] -= dX[2] * invR3;
	}
	Ibodies[i] += TRG * wi;
      }
    }

    void P2PX(std::vector<vec4> &Ibodies, int ibegin, int iend, vec3,
              std::vector<vec4> &Jbodies, int jbegin, int jend, vec3, real_t, vec3 periodic) const {
      for (int i=ibegin; i<iend; i++) {
        vec4 TRG = 0;
	for (int j=jbegin; j<jend; j++) {
	  vec3 dX;
	  for_3d dX[d] = Jbodies[i][d] - Jbodies[j][d] - periodic[d];
	  real_t R2 = norm(dX);
	  real_t invR2 = 1.0 / R2;
	  if (R2 == 0) invR2 = 0;
	  real_t invR = Jbodies[j][3] * sqrt(invR2);
	  real_t invR3 = invR2 * invR;
	  TRG[0] += invR;
	  TRG[1] -= dX[0] * invR3;
	  TRG[2] -= dX[1] * invR3;
	  TRG[3] -= dX[2] * invR3;
	}
	Ibodies[i] += TRG;
      }
    }

    void EwaldP2P(std::vector<vec4> &Ibodies, int ibegin, int iend,
                  std::vector<vec4> &Jbodies, int jbegin, int jend, vec3 periodic,
                  real_t alpha, real_t cutoff) const {
      for (int i=ibegin; i<iend; i++) {
        vec4 TRG = 0;
	for (int j=jbegin; j<jend; j++) {
	  vec3 dX;
	  for_3d dX[d] = Jbodies[i][d] - Jbodies[j][d] - periodic[d];
	  real_t R2 = norm(dX);
          if (0 < R2 && R2 < cutoff * cutoff) {
            real_t R2s = R2 * alpha * alpha;
            real_t Rs = std::sqrt(R2s);
            real_t invRs = 1 / Rs;
            real_t invR2s = invRs * invRs;
            real_t invR3s = invR2s * invRs;
            real_t dtmp = Jbodies[j][3] * (M_2_SQRTPI * std::exp(-R2s) * invR2s + erfc(Rs) * invR3s);
            dtmp *= alpha * alpha * alpha;
            TRG[0] += Jbodies[j][3] * erfc(Rs) * invRs * alpha;
            TRG[1] -= dX[0] * dtmp;
            TRG[2] -= dX[1] * dtmp;
            TRG[3] -= dX[2] * dtmp;
          }
	}
	Ibodies[i] += TRG;
      }
    }

    void VdWP2P(std::vector<vec4> &Ibodies, int ibegin, int iend,
                std::vector<vec4> &Jbodies, int jbegin, int jend, vec3 periodic,
                real_t cuton, real_t cutoff, int numTypes,
                real_t * rscale, real_t * gscale, real_t * fgscale) const {
      for (int i=ibegin; i<iend; i++) {
	int atypei = int(Jbodies[i][3]);
        vec4 TRG = 0;
	for (int j=jbegin; j<jend; j++) {
	  vec3 dX;
	  for_3d dX[d] = Jbodies[i][d] - Jbodies[j][d] - periodic[d];
	  real_t R2 = norm(dX);
	  if (R2 != 0) {
	    int atypej = int(Jbodies[j][3]);
	    real_t rs = rscale[atypei*numTypes+atypej];
	    real_t gs = gscale[atypei*numTypes+atypej];
	    real_t fgs = fgscale[atypei*numTypes+atypej];
	    real_t R2s = R2 * rs;
	    real_t invR2 = 1.0 / R2s;
	    real_t invR6 = invR2 * invR2 * invR2;
	    real_t cuton2 = cuton * cuton;
	    real_t cutoff2 = cutoff * cutoff;
	    if (R2 < cutoff2) {
	      real_t tmp = 0, dtmp = 0;
	      if (cuton2 < R2) {
		real_t tmp1 = (cutoff2 - R2) / ((cutoff2-cuton2)*(cutoff2-cuton2)*(cutoff2-cuton2));
		real_t tmp2 = tmp1 * (cutoff2 - R2) * (cutoff2 - 3 * cuton2 + 2 * R2);
		tmp = invR6 * (invR6 - 1) * tmp2;
		dtmp = invR6 * (invR6 - 1) * 12 * (cuton2 - R2) * tmp1
		  - 6 * invR6 * (invR6 + (invR6 - 1) * tmp2) * tmp2 / R2;
	      } else {
		tmp = invR6 * (invR6 - 1);
		dtmp = invR2 * invR6 * (2 * invR6 - 1);
	      }
	      dtmp *= fgs;
	      TRG[0] += gs * tmp;
	      TRG[1] -= dX[0] * dtmp;
	      TRG[2] -= dX[1] * dtmp;
	      TRG[3] -= dX[2] * dtmp;
	    }
	  }
	}
	Ibodies[i] += TRG;
      }
    }

    Kernel() {
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
