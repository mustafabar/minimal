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
  cvecP Mc, Mp, Lc = complex_t(0), Lp = complex_t(0);
  vec3 dX;
  for (int i=0; i<numBodies; i++) {
    for_3d dX[d] = Jbodies[i][d] - 0.5;
    kernel.P2M(dX, Jbodies[i][3], Mc);
  }
  dX = 0.5;
  kernel.M2M(dX, Mc, Mp);
  dX = 0;
  dX[0] = 4;
  kernel.M2L(dX, Mp, Lp);
  dX = -0.5;
  dX[0] = 0.5;
  kernel.L2L(dX, Lp, Lc);
  for (int i=0; i<numBodies; i++) {
    for_3d dX[d] = Jbodies[i+numBodies][d] - 0.5;
    dX[0] = Jbodies[i+numBodies][0] - 5.5;
    kernel.L2P(dX, Lc, Ibodies[i]);
  }
  dX = 0;
  kernel.P2P(Ibodies, numBodies, 2*numBodies, Jbodies, 0, numBodies, dX);
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
