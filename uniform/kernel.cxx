#include "types.h"
#include "kernels.h"

using namespace exafmm;

int main(int argc, char ** argv) {
  const int numBodies = 10;
  std::vector<vec4> Ibodies(2*numBodies);
  std::vector<vec4> Jbodies(2*numBodies);
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
  cvecP Mc[27], Mp[8], Lc[27], Lp[8];
  vec3 dX, Xj, Xi;
  Xj = 0.5;
  for (int i=0; i<numBodies; i++) {
    for_3d dX[d] = Jbodies[i][d] - Xj[d];
    kernel.P2M(dX, 0.5, Jbodies[i][3], Mc[13]);
  }
  dX = 0.5;
  kernel.M2M(dX, Mc[13], Mp[0]);
  dX = 0;
  dX[0] = 4;
  Lp[0] = complex_t(0);
  kernel.M2L(dX, Mp[0], Lp[0]);
  dX = -0.5;
  dX[0] = 0.5;
  Lc[13] = complex_t(0);
  kernel.L2L(dX, Lp[0], Lc[13]);
  Xi = 0.5;
  Xi[0] = 5.5;
  for (int i=0; i<numBodies; i++) {
    for_3d dX[d] = Jbodies[i+numBodies][d] - Xi[d];
    kernel.L2P(dX, 0.5, Lc[13], Ibodies[i]);
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
