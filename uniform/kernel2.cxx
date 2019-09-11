#include "types.h"
#include "kernels.h"

using namespace exafmm;

int main(int argc, char ** argv) {
  const int numBodies = 10;
  std::vector<vec4> Ibodies(2*numBodies);
  std::vector<vec4> Jbodies(2*numBodies);
  srand48(0);
  for (int i=0; i<numBodies; i++) {
    Ibodies[i] = 0;
    Jbodies[i][0] = drand48();
    Jbodies[i][1] = drand48();
    Jbodies[i][2] = drand48();
    Jbodies[i][3] = drand48();
    Ibodies[i+numBodies] = 0;
    Jbodies[i+numBodies][0] = drand48() + 5;
    Jbodies[i+numBodies][1] = drand48();
    Jbodies[i+numBodies][2] = drand48();
    Jbodies[i+numBodies][3] = drand48();
  }
  Kernel kernel;
  cvecP Mc[3][3][3], Mp[2][2][2], Lc[3][3][3], Lp[2][2][2];
  vec3 dX, Xi, Xj = 0.5;
  ivec3 iX, jX;
  for (iX[0]=0; iX[0]<3; iX[0]++) {
    for (iX[1]=0; iX[1]<3; iX[1]++) {
      for (iX[2]=0; iX[2]<3; iX[2]++) {
        Mc[iX[0]][iX[1]][iX[2]] = complex_t(0);
        Lc[iX[0]][iX[1]][iX[2]] = complex_t(0);
        for (int i=0; i<numBodies; i++) {
          for_3d dX[d] = Jbodies[i][d] - iX[d] - Xj[d] + 1;
          kernel.P2M(dX, 0.5, Jbodies[i][3], Mc[iX[0]][iX[1]][iX[2]]);
        }
        std::cout << iX << " " << Mc[iX[0]][iX[1]][iX[2]][0] << std::endl;
      }
    }
  }
  for (iX[0]=0; iX[0]<2; iX[0]++) {
    for (iX[1]=0; iX[1]<2; iX[1]++) {
      for (iX[2]=0; iX[2]<2; iX[2]++) {
        Mp[iX[0]][iX[1]][iX[2]] = complex_t(0);
        Lp[iX[0]][iX[1]][iX[2]] = complex_t(0);
      }
    }
  }
  for (jX[0]=0; jX[0]<3; jX[0]++) {
    for (jX[1]=0; jX[1]<3; jX[1]++) {
      for (jX[2]=0; jX[2]<3; jX[2]++) {
        iX = (jX + 1) / 2;
        for_3d dX[d] = 2 * iX[d] - jX[d] - Xj[d];
        kernel.M2M(dX, Mc[jX[0]][jX[1]][jX[2]], Mp[iX[0]][iX[1]][iX[2]]);
      }
    }
  }
  Xi = 0;
  Xi[0] = 6;
  for (iX[0]=0; iX[0]<2; iX[0]++) {
    for (iX[1]=0; iX[1]<2; iX[1]++) {
      for (iX[2]=0; iX[2]<2; iX[2]++) {
        for (jX[0]=0; jX[0]<2; jX[0]++) {
          for (jX[1]=0; jX[1]<2; jX[1]++) {
            for (jX[2]=0; jX[2]<2; jX[2]++) {
              for_3d dX[d] = 2 * iX[d] - 2 * jX[d] + Xi[d];
              kernel.M2L(dX, Mp[jX[0]][jX[1]][jX[2]], Lp[iX[0]][iX[1]][iX[2]]);
            }
          }
        }
      }
    }
  }
  Xi = 0.5;
  Xi[0] = -0.5;
  for (iX[0]=0; iX[0]<3; iX[0]++) {
    for (iX[1]=0; iX[1]<3; iX[1]++) {
      for (iX[2]=0; iX[2]<3; iX[2]++) {
        jX[0] = iX[0] / 2;
        jX[1] = (iX[1] + 1) / 2;
        jX[2] = (iX[2] + 1) / 2;
        for_3d dX[d] = iX[d] - 2 * jX[d] + Xi[d];
        kernel.L2L(dX, Lp[jX[0]][jX[1]][jX[2]], Lc[iX[0]][iX[1]][iX[2]]);
      }
    }
  }
  Xi = 0.5;
  Xi[0] = 5.5;
  for (iX[0]=0; iX[0]<3; iX[0]++) {
    for (iX[1]=0; iX[1]<3; iX[1]++) {
      for (iX[2]=0; iX[2]<3; iX[2]++) {
        for (int i=0; i<numBodies; i++) {
          for_3d dX[d] = Jbodies[i+numBodies][d] - Xi[d] - iX[d] + 1;
          kernel.L2P(dX, 0.5, Lc[iX[0]][iX[1]][iX[2]], Ibodies[i]);
        }
      }
    }
  }
  dX = 0;
  kernel.P2P(Ibodies, numBodies, 2*numBodies, Xi, Jbodies, 0, numBodies, Xj, 0.5, dX);
  double potDif = 0, potNrm = 0, accDif = 0, accNrm = 0;
  for (int i=0; i<numBodies; i++) {
    potDif += (Ibodies[i+numBodies][0] - Ibodies[i][0]) * (Ibodies[i+numBodies][0] - Ibodies[i][0]);
    potNrm += Ibodies[i+numBodies][0] * Ibodies[i+numBodies][0];
    accDif += (Ibodies[i+numBodies][1] - Ibodies[i][1]) * (Ibodies[i+numBodies][1] - Ibodies[i][1]);
    accDif += (Ibodies[i+numBodies][2] - Ibodies[i][2]) * (Ibodies[i+numBodies][2] - Ibodies[i][2]);
    accDif += (Ibodies[i+numBodies][3] - Ibodies[i][3]) * (Ibodies[i+numBodies][3] - Ibodies[i][3]);
    accNrm += Ibodies[i+numBodies][1] * Ibodies[i+numBodies][1];
    accNrm += Ibodies[i+numBodies][2] * Ibodies[i+numBodies][2];
    accNrm += Ibodies[i+numBodies][3] * Ibodies[i+numBodies][3];
  }  
  printf("Rel. L2 Error (pot) : %e\n",std::sqrt(potDif/potNrm));
  printf("Rel. L2 Error (acc) : %e\n",std::sqrt(accDif/accNrm));
}
