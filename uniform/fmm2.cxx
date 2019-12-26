#include "args.h"
#include "ewald.h"
#include "verify.h"
#include "serialfmm.h"
using namespace exafmm;

int main(int argc, char ** argv) {
  Args args(argc, argv);
  Verify verify;

  const int numBodies = 16;
  const int ncrit = 9;
  const int maxLevel = 3;
  const int numImages = 0;

  SerialFMM FMM(numBodies, maxLevel, numImages);
  VERBOSE = 1;
  args.verbose = VERBOSE;
  print("FMM Parameters");
  args.print(stringLength);

  print("FMM Profiling");
  start("Total FMM");
  FMM.R0 = 4;
  for_3d FMM.X0[d] = 0;
  srand48(0);
  for (int i=0; i<8; i++) {
    FMM.Jbodies[i][0] = drand48();
    FMM.Jbodies[i][1] = drand48();
    FMM.Jbodies[i][2] = drand48();
    FMM.Jbodies[i][3] = drand48();
    FMM.Jbodies[i+8][0] = drand48() - 3;
    FMM.Jbodies[i+8][1] = drand48();
    FMM.Jbodies[i+8][2] = drand48();
    FMM.Jbodies[i+8][3] = 0;
  }

  start("Grow tree");
  FMM.sortBodies();
  FMM.buildTree();
  stop("Grow tree");
  FMM.upwardPass();
  FMM.periodicM2L();
  FMM.downwardPass();
  stop("Total FMM");

  start("Direct");
  std::vector<vec4> Ibodies(FMM.numBodies);
  for (int b=0; b<FMM.numBodies; b++) {
    Ibodies[b] = FMM.Ibodies[b];
    FMM.Ibodies[b] = 0;
  }
  vec3 periodic = 0;
  FMM.P2PX(FMM.Ibodies,0,FMM.numBodies,FMM.X0,
           FMM.Jbodies,0,FMM.numBodies,FMM.X0,FMM.R0,periodic);

  stop("Direct");
  double potSum = verify.getSumScalar(FMM.Ibodies);
  double potSum2 = verify.getSumScalar(Ibodies);
  double accDif = verify.getDifVector(FMM.Ibodies, Ibodies);
  double accNrm = verify.getNrmVector(FMM.Ibodies);
  print("FMM vs. direct");
  double potDif = (potSum - potSum2) * (potSum - potSum2);
  double potNrm = potSum * potSum;
  verify.print("Rel. L2 Error (pot)",std::sqrt(potDif/potNrm));
  verify.print("Rel. L2 Error (acc)",std::sqrt(accDif/accNrm));
}
