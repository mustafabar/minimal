#include "base_mpi.h"
#include "args.h"
#include "ewald.h"
#include "verify.h"
#if EXAFMM_SERIAL
#include "serialfmm.h"
#else
#include "parallelfmm.h"
#endif
using namespace exafmm;

int main(int argc, char ** argv) {
  const int ksize = 14;
  const vec3 cycle = 10 * M_PI;
  const real_t alpha = 10 / max(cycle);
  const real_t sigma = .25 / M_PI;
  const real_t cutoff = 10;
  Args args(argc, argv);
  BaseMPI baseMPI;
  Ewald ewald(ksize, alpha, sigma, cutoff, cycle);
  Verify verify;

  args.numBodies /= baseMPI.mpisize;
  const int numBodies = args.numBodies;
  const int ncrit = args.ncrit;
  const int maxLevel = numBodies >= ncrit ? 1 + int(log(numBodies / ncrit)/M_LN2/3) : 0;
  const int gatherLevel = 1;
  const int numImages = args.images;
  if (numImages > 0 && int(log2(baseMPI.mpisize)) % 3 != 0) {
    if (baseMPI.mpirank==0) printf("Warning: MPISIZE must be a power of 8 for periodic domain to be square\n");
  }

#if EXAFMM_SERIAL
  SerialFMM FMM(numBodies, maxLevel, numImages);
#else
  ParallelFMM FMM(numBodies, maxLevel, numImages);
#endif
  VERBOSE = FMM.MPIRANK == 0;
  args.verbose = VERBOSE;
  print("FMM Parameters");
  args.print(stringLength);

  print("FMM Profiling");
  start("Total FMM");
  start("Partition");
  FMM.partitioner(gatherLevel);
  stop("Partition");

  for (int it=0; it<1; it++) {
    int iX[3] = {0, 0, 0};
    FMM.R0 = 0.5 * max(cycle) / FMM.numPartition[FMM.maxGlobLevel][0];
    for_3d FMM.RGlob[d] = FMM.R0 * FMM.numPartition[FMM.maxGlobLevel][d];
    FMM.getGlobIndex(iX,FMM.MPIRANK,FMM.maxGlobLevel);
    for_3d FMM.X0[d] = 2 * FMM.R0 * (iX[d] + .5);
    srand48(FMM.MPIRANK);
    real_t average = 0;
    for (int i=0; i<FMM.numBodies; i++) {
      FMM.Jbodies[i][0] = 2 * FMM.R0 * (drand48() + iX[0]);
      FMM.Jbodies[i][1] = 2 * FMM.R0 * (drand48() + iX[1]);
      FMM.Jbodies[i][2] = 2 * FMM.R0 * (drand48() + iX[2]);
      FMM.Jbodies[i][3] = (drand48() - .5) / FMM.numBodies;
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

#if EXAFMM_SERIAL
#else
    start("Comm LET bodies");
    FMM.P2PSend();
    FMM.P2PRecv();
    stop("Comm LET bodies");
#endif

    FMM.upwardPass();

#if EXAFMM_SERIAL
#else
    start("Comm LET cells");
    for (int lev=FMM.maxLevel; lev>0; lev--) {
      MPI_Barrier(MPI_COMM_WORLD);
      FMM.M2LSend(lev);
      FMM.M2LRecv(lev);
    }
    FMM.rootGather();
    stop("Comm LET cells");
    FMM.globM2M();
    FMM.globM2L();
#endif

    FMM.periodicM2L();

#if EXAFMM_SERIAL
#else
    FMM.globL2L();
#endif

    FMM.downwardPass();
    stop("Total FMM");

    vec3 localDipole = FMM.getDipole();
    vec3 globalDipole = baseMPI.allreduceVec3(localDipole);
    int globalNumBodies = baseMPI.allreduceInt(FMM.numBodies);
    FMM.dipoleCorrection(globalDipole, globalNumBodies);

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
    waves = baseMPI.allreduceWaves(waves);
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
#if EXAFMM_SERIAL
    double potDif = (potSum - potSum2) * (potSum - potSum2);
    double potNrm = potSum * potSum;
    verify.print("Rel. L2 Error (pot)",std::sqrt(potDif/potNrm));
    verify.print("Rel. L2 Error (acc)",std::sqrt(accDif/accNrm));
#else
    double potSumGlob, potSumGlob2, accDifGlob, accNrmGlob;
    MPI_Reduce(&potSum,  &potSumGlob,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&potSum2, &potSumGlob2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&accDif,  &accDifGlob,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&accNrm,  &accNrmGlob,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double potDifGlob = (potSumGlob - potSumGlob2) * (potSumGlob - potSumGlob2);
    double potNrmGlob = potSumGlob * potSumGlob;
    double potRel = std::sqrt(potDifGlob/potNrmGlob);
    double accRel = std::sqrt(accDifGlob/accNrmGlob);
    verify.print("Rel. L2 Error (pot)",potRel);
    verify.print("Rel. L2 Error (acc)",accRel);
#endif
  }
}
