#ifndef laplace_h
#define laplace_h
#include "namespace.h"
#if EXAFMM_USE_SIMD
#include "simdvec.h"
#endif

namespace EXAFMM_NAMESPACE {
  class Kernel {
  private:
    std::vector<real_t> prefactor;
    std::vector<real_t> Anm;
    std::vector<complex_t> Cnm;

  public:
    const int P;
    const int NTERM;
    real_t eps2;
    vec3 Xperiodic;

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

  public:
    Kernel(int _P, real_t _eps2) : P(_P), NTERM(P*(P+1)/2), eps2(_eps2) {
      Xperiodic = 0;
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

    void P2P(C_iter Ci, C_iter Cj) {
      B_iter Bi = Ci->BODY;
      B_iter Bj = Cj->BODY;
      int ni = Ci->NBODY;
      int nj = Cj->NBODY;
      int i = 0;
#if EXAFMM_USE_SIMD
      for ( ; i<=ni-NSIMD; i+=NSIMD) {
        simdvec zero = 0.0;
        simdvec pot = zero;
        simdvec ax = zero;
        simdvec ay = zero;
        simdvec az = zero;

        simdvec xi = SIMD<simdvec,B_iter,0,NSIMD>::setBody(Bi,i);
        simdvec yi = SIMD<simdvec,B_iter,1,NSIMD>::setBody(Bi,i);
        simdvec zi = SIMD<simdvec,B_iter,2,NSIMD>::setBody(Bi,i);

        simdvec xj = Xperiodic[0];
        xi -= xj;
        simdvec yj = Xperiodic[1];
        yi -= yj;
        simdvec zj = Xperiodic[2];
        zi -= zj;

        for (int j=0; j<nj; j++) {
          simdvec dx = Bj[j].X[0];
          dx -= xi;
          simdvec dy = Bj[j].X[1];
          dy -= yi;
          simdvec dz = Bj[j].X[2];
          dz -= zi;
          simdvec mj = Bj[j].SRC;

          simdvec R2 = eps2;
          xj = dx;
          R2 += dx * dx;
          yj = dy;
          R2 += dy * dy;
          zj = dz;
          R2 += dz * dz;
          simdvec invR = rsqrt(R2);
          invR &= R2 > zero;

          mj *= invR;
          pot += mj;
          invR = invR * invR * mj;
          xj *= invR;
          ax += xj;
          yj *= invR;
          ay += yj;
          zj *= invR;
          az += zj;
        }
        for (int k=0; k<NSIMD; k++) {
          Bi[i+k].TRG[0] += transpose(pot, k);
          Bi[i+k].TRG[1] += transpose(ax, k);
          Bi[i+k].TRG[2] += transpose(ay, k);
          Bi[i+k].TRG[3] += transpose(az, k);
        }
      }
#endif
      for ( ; i<ni; i++) {
        real_t pot = 0;
        real_t ax = 0;
        real_t ay = 0;
        real_t az = 0;
        for (int j=0; j<nj; j++) {
          vec3 dX = Bi[i].X - Bj[j].X - Xperiodic;
          real_t R2 = norm(dX) + eps2;
          if (R2 != 0) {
            real_t invR2 = 1.0 / R2;
            real_t invR = Bj[j].SRC * sqrt(invR2);
            dX *= invR2 * invR;
            pot += invR;
            ax += dX[0];
            ay += dX[1];
            az += dX[2];
          }
        }
        Bi[i].TRG[0] += pot;
        Bi[i].TRG[1] -= ax;
        Bi[i].TRG[2] -= ay;
        Bi[i].TRG[3] -= az;
      }
    }

    void P2M(C_iter C) {
      complex_t Ynm[P*P], YnmTheta[P*P];
      for (B_iter B=C->BODY; B!=C->BODY+C->NBODY; B++) {
        vec3 dX = B->X - C->X;
        real_t rho, alpha, beta;
        cart2sph(dX, rho, alpha, beta);
        evalMultipole(rho, alpha, -beta, Ynm, YnmTheta);
        for (int n=0; n<P; n++) {
          for (int m=0; m<=n; m++) {
            int nm  = n * n + n + m;
            int nms = n * (n + 1) / 2 + m;
            C->M[nms] += B->SRC * Ynm[nm];
          }
        }
      }
    }

    void M2M(C_iter Ci, C_iter C0) {
      complex_t Ynm[P*P], YnmTheta[P*P];
      for (C_iter Cj=C0+Ci->ICHILD; Cj!=C0+Ci->ICHILD+Ci->NCHILD; Cj++) {
        vec3 dX = Ci->X - Cj->X;
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
                  M += Cj->M[jnkms] * std::pow(I,real_t(m-abs(m))) * Ynm[nm]
                    * real_t(oddOrEven(n) * Anm[nm] * Anm[jnkm] / Anm[jk]);
                }
              }
              for (int m=k; m<=n; m++) {
                if (j-n >= m-k) {
                  int jnkm  = (j - n) * (j - n) + j - n + k - m;
                  int jnkms = (j - n) * (j - n + 1) / 2 - k + m;
                  int nm    = n * n + n + m;
                  M += std::conj(Cj->M[jnkms]) * Ynm[nm]
                    * real_t(oddOrEven(k+n+m) * Anm[nm] * Anm[jnkm] / Anm[jk]);
                }
              }
            }
            Ci->M[jks] += M * EPS;
          }
        }
      }
    }

    void M2L(C_iter Ci, C_iter Cj) {
      complex_t Ynm2[4*P*P];
      vec3 dX = Ci->X - Cj->X - Xperiodic;
      real_t rho, alpha, beta;
      cart2sph(dX, rho, alpha, beta);
      evalLocal(rho, alpha, beta, Ynm2);
      for (int j=0; j<P; j++) {
        for (int k=0; k<=j; k++) {
          int jk = j * j + j + k;
          int jks = j * (j + 1) / 2 + k;
          complex_t L = 0;
          for (int n=0; n<P; n++) {
            for (int m=-n; m<0; m++) {
              int nm   = n * n + n + m;
              int nms  = n * (n + 1) / 2 - m;
              int jknm = jk * P * P + nm;
              int jnkm = (j + n) * (j + n) + j + n + m - k;
              L += std::conj(Cj->M[nms]) * Cnm[jknm] * Ynm2[jnkm];
            }
            for (int m=0; m<=n; m++) {
              int nm   = n * n + n + m;
              int nms  = n * (n + 1) / 2 + m;
              int jknm = jk * P * P + nm;
              int jnkm = (j + n) * (j + n) + j + n + m - k;
              L += Cj->M[nms] * Cnm[jknm] * Ynm2[jnkm];
            }
          }
          Ci->L[jks] += L;
        }
      }
    }

    void L2L(C_iter Ci, C_iter C0) {
      complex_t Ynm[P*P], YnmTheta[P*P];
      C_iter Cj = C0 + Ci->IPARENT;
      vec3 dX = Ci->X - Cj->X;
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
              L += std::conj(Cj->L[nms]) * Ynm[jnkm]
                * real_t(oddOrEven(k) * Anm[jnkm] * Anm[jk] / Anm[nm]);
            }
            for (int m=0; m<=n; m++) {
              if (n-j >= abs(m-k)) {
                int jnkm = (n - j) * (n - j) + n - j + m - k;
                int nm   = n * n + n + m;
                int nms  = n * (n + 1) / 2 + m;
                L += Cj->L[nms] * std::pow(I,real_t(m-k-abs(m-k)))
                  * Ynm[jnkm] * Anm[jnkm] * Anm[jk] / Anm[nm];
              }
            }
          }
          Ci->L[jks] += L * EPS;
        }
      }
    }

    void L2P(C_iter Ci) {
      complex_t Ynm[P*P], YnmTheta[P*P];
      for (B_iter B=Ci->BODY; B!=Ci->BODY+Ci->NBODY; B++) {
        vec3 dX = B->X - Ci->X + EPS;
        vec3 spherical = 0;
        vec3 cartesian = 0;
        real_t r, theta, phi;
        cart2sph(dX, r, theta, phi);
        evalMultipole(r, theta, phi, Ynm, YnmTheta);
        for (int n=0; n<P; n++) {
          int nm  = n * n + n;
          int nms = n * (n + 1) / 2;
          B->TRG[0] += std::real(Ci->L[nms] * Ynm[nm]);
          spherical[0] += std::real(Ci->L[nms] * Ynm[nm]) / r * n;
          spherical[1] += std::real(Ci->L[nms] * YnmTheta[nm]);
          for (int m=1; m<=n; m++) {
            nm  = n * n + n + m;
            nms = n * (n + 1) / 2 + m;
            B->TRG[0] += 2 * std::real(Ci->L[nms] * Ynm[nm]);
            spherical[0] += 2 * std::real(Ci->L[nms] * Ynm[nm]) / r * n;
            spherical[1] += 2 * std::real(Ci->L[nms] * YnmTheta[nm]);
            spherical[2] += 2 * std::real(Ci->L[nms] * Ynm[nm] * I) * m;
          }
        }
        sph2cart(r, theta, phi, spherical, cartesian);
        B->TRG[1] += cartesian[0];
        B->TRG[2] += cartesian[1];
        B->TRG[3] += cartesian[2];
      }
    }
  };
}
#endif
