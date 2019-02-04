#include "base_mpi.h"
#include "args.h"
#include "bound_box.h"
#include "build_tree.h"
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
  const vec3 cycle = 20 * M_PI;
  const real_t alpha = 10 / max(cycle);
  const real_t sigma = .25 / M_PI;
  const real_t cutoff = 20;
  Args args(argc, argv);
  BaseMPI baseMPI;
  BoundBox boundBox;
  BuildTree buildTree(args.ncrit);
  Ewald ewald(ksize, alpha, sigma, cutoff, cycle);
  Verify verify;

  args.numBodies /= baseMPI.mpisize;
  int numBodies = args.numBodies;
  const int ncrit = 100;
  const int maxLevel = numBodies >= ncrit ? 1 + int(log(numBodies / ncrit)/M_LN2/3) : 0;
  const int gatherLevel = 1;
  const int numImages = args.images;
  if (numImages > 0 && int(log2(baseMPI.mpisize)) % 3 != 0) {
    if (baseMPI.mpirank==0) printf("Warning: MPISIZE must be a power of 8 for periodic domain to be square\n");
  }

#if EXAFMM_SERIAL
  SerialFMM FMM;
#else
  ParallelFMM FMM;
#endif
  FMM.allocate(numBodies, maxLevel, numImages);
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
    start("Downward pass");
    FMM.globL2L();
    stop("Downward pass");
#endif

    FMM.downwardPass();
    stop("Total FMM");

    vec3 localDipole = FMM.getDipole();
    vec3 globalDipole = baseMPI.allreduceVec3(localDipole);
    numBodies = baseMPI.allreduceInt(FMM.numBodies);
    FMM.dipoleCorrection(globalDipole, numBodies);

    Bodies bodies(FMM.numBodies);
    B_iter B = bodies.begin();
    for (int b=0; b<FMM.numBodies; b++, B++) {
      for_3d B->X[d] = FMM.Jbodies[b][d];
      B->SRC = FMM.Jbodies[b][3];
      B->TRG = FMM.Ibodies[b];
    }

    start("Total Ewald");
    Bounds bounds = boundBox.getBounds(bodies);
    Bodies buffer = bodies;
    Cells cells = buildTree.buildTree(bodies, buffer, bounds);
    std::vector<vec4> ibodies2(FMM.numBodies);
    B = bodies.begin();
    for (int b=0; b<FMM.numBodies; b++, B++) {
      ibodies2[b] = B->TRG;
      ibodies2[b][0] *= B->SRC;
    }
    Bodies jbodies = bodies;
    ewald.initTarget(bodies);
    for (int i=0; i<FMM.MPISIZE; i++) {
      if (VERBOSE) std::cout << "Ewald loop           : " << i+1 << "/" << FMM.MPISIZE << std::endl;
      if (FMM.MPISIZE > 1) baseMPI.shiftBodies(jbodies);
      bounds = boundBox.getBounds(jbodies);
      buffer = jbodies;
      Cells jcells = buildTree.buildTree(jbodies, buffer, bounds);
      start("Ewald real part");
      ewald.realPart(cells, jcells);
      stop("Ewald real part");
    }
    start("Ewald wave part");
    Waves waves = ewald.initWaves();
    ewald.dft(waves,jbodies);
    waves = baseMPI.allreduceWaves(waves);
    ewald.wavePart(waves);
    ewald.idft(waves,bodies);
    stop("Ewald wave part");
    ewald.selfTerm(bodies);
    std::vector<vec4> ibodies(FMM.numBodies);
    B = bodies.begin();
    for (int b=0; b<FMM.numBodies; b++, B++) {
      ibodies[b] = B->TRG;
      ibodies[b][0] *= B->SRC;
    }
    stop("Total Ewald");
    double potSum = verify.getSumScalar(ibodies);
    double potSum2 = verify.getSumScalar(ibodies2);
    double accDif = verify.getDifVector(ibodies, ibodies2);
    double accNrm = verify.getNrmVector(ibodies);
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
