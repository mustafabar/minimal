#include "args.h"
#include "ewald.h"
#include "verify.h"
#include "serialfmm.h"
using namespace exafmm;

int main(int argc, char ** argv) {
  const int ksize = 14;
  const real_t cycle = 10 * M_PI;
  const real_t alpha = 10 / cycle;
  const real_t sigma = .25 / M_PI;
  const real_t cutoff = 10;
  Args args(argc, argv);
  Ewald ewald(ksize, alpha, sigma, cutoff, cycle);
  Verify verify;

  const int numBodies = args.numBodies;
  const int ncrit = args.ncrit;
  const int maxLevel = numBodies >= ncrit ? 1 + int(log(numBodies / ncrit)/M_LN2/3) : 0;
  const int gatherLevel = 1;
  const int numImages = args.images;

  SerialFMM FMM(numBodies, maxLevel, numImages);
  VERBOSE = 1;
  args.verbose = VERBOSE;
  print("FMM Parameters");
  args.print(stringLength);

  print("FMM Profiling");
  start("Total FMM");
  start("Partition");
  FMM.partitioner(gatherLevel);
  stop("Partition");

  int iX[3] = {0, 0, 0};
  FMM.R0 = 0.5 * cycle;
  for_3d FMM.RGlob[d] = FMM.R0;
  FMM.getGlobIndex(iX,0,FMM.maxGlobLevel);
  for_3d FMM.X0[d] = 2 * FMM.R0 * (iX[d] + .5);
  srand48(0);
  real_t average = 0;
  for (int i=0; i<FMM.numBodies; i++) {
    FMM.Jbodies[i][0] = 2 * FMM.R0 * (drand48() + iX[0]);
    FMM.Jbodies[i][1] = 2 * FMM.R0 * (drand48() + iX[1]);
    FMM.Jbodies[i][2] = 2 * FMM.R0 * (drand48() + iX[2]);
    FMM.Jbodies[i][3] = drand48();
    average += FMM.Jbodies[i][3];
  }
  average /= FMM.numBodies;
  for (int i=0; i<FMM.numBodies; i++) {
    FMM.Jbodies[i][3] -= average;
  }

  start("Grow tree");
  FMM.sortBodies();
  FMM.buildTree();
  stop("Grow tree");
  FMM.upwardPass();
  FMM.periodicM2L();
  FMM.downwardPass();
  stop("Total FMM");

  vec3 dipole = FMM.getDipole();
  FMM.dipoleCorrection(dipole, FMM.numBodies);

  start("Total Ewald");
  std::vector<vec4> Ibodies(FMM.numBodies);
  for (int b=0; b<FMM.numBodies; b++) {
    Ibodies[b] = FMM.Ibodies[b];
    Ibodies[b][0] *= FMM.Jbodies[b][3];
    FMM.Ibodies[b] = 0;
  }
  start("Ewald real part");
  FMM.ewaldRealPart(alpha,cutoff);
  stop("Ewald real part");
  FMM.Ibodies.resize(FMM.numBodies);
  FMM.Jbodies.resize(FMM.numBodies);
  start("Ewald wave part");
  Waves waves = ewald.initWaves();
  ewald.dft(waves,FMM.Jbodies);
  ewald.wavePart(waves);
  ewald.idft(waves,FMM.Ibodies,FMM.Jbodies);
  stop("Ewald wave part");
  ewald.selfTerm(FMM.Ibodies, FMM.Jbodies);
  for (int b=0; b<FMM.numBodies; b++) {
    FMM.Ibodies[b][0] *= FMM.Jbodies[b][3];
  }
  stop("Total Ewald");
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
