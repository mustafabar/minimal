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
  const int ksize = 11;
  const vec3 cycle = 20 * M_PI;
  const real_t alpha = 10 / max(cycle);
  const real_t sigma = .25 / M_PI;
  const real_t cutoff = 20;
  const real_t eps2 = 0.0;
  const complex_t wavek = complex_t(10.,1.) / real_t(2 * M_PI);
  Args args(argc, argv);
  BaseMPI baseMPI;
  BoundBox boundBox;
  BuildTree buildTree(args.ncrit);
  Ewald ewald(ksize, alpha, sigma, cutoff, cycle);
#if EXAFMM_SERIAL
  SerialFMM FMM;
#else
  ParallelFMM FMM;
#endif
  Verify verify(args.path);
  verify.verbose = args.verbose;

  args.numBodies /= FMM.MPISIZE;
  int numBodies = args.numBodies;
  const int ncrit = 100;
  const int maxLevel = numBodies >= ncrit ? 1 + int(log(numBodies / ncrit)/M_LN2/3) : 0;
  const int gatherLevel = 1;
  const int numImages = args.images;
  if (numImages > 0 && int(log2(FMM.MPISIZE)) % 3 != 0) {
    if (FMM.MPIRANK==0) printf("Warning: MPISIZE must be a power of 8 for periodic domain to be square\n");
  }

  FMM.allocate(numBodies, maxLevel, numImages);
  args.verbose &= FMM.MPIRANK == 0;
  logger::verbose = args.verbose;
  logger::printTitle("FMM Parameters");
  args.print(logger::stringLength);

  logger::printTitle("FMM Profiling");
  logger::startTimer("Total FMM");
  logger::startTimer("Partition");
  FMM.partitioner(gatherLevel);
  logger::stopTimer("Partition");

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

    logger::startTimer("Grow tree");
    FMM.sortBodies();
    FMM.buildTree();
    logger::stopTimer("Grow tree");

#if EXAFMM_SERIAL
#else
    logger::startTimer("Comm LET bodies");
    FMM.P2PSend();
    FMM.P2PRecv();
    logger::stopTimer("Comm LET bodies");
#endif

    FMM.upwardPass();

#if EXAFMM_SERIAL
#else
    logger::startTimer("Comm LET cells");
    for (int lev=FMM.maxLevel; lev>0; lev--) {
      MPI_Barrier(MPI_COMM_WORLD);
      FMM.M2LSend(lev);
      FMM.M2LRecv(lev);
    }
    FMM.rootGather();
    logger::stopTimer("Comm LET cells", 0);
    FMM.globM2M();
    FMM.globM2L();
#endif

    FMM.periodicM2L();

#if EXAFMM_SERIAL
#else
    logger::startTimer("Downward pass");
    FMM.globL2L();
    logger::stopTimer("Downward pass", 0);
#endif

    FMM.downwardPass();
    logger::stopTimer("Total FMM", 0);

    Bodies bodies(FMM.numBodies);
    B_iter B = bodies.begin();
    for (int b=0; b<FMM.numBodies; b++, B++) {
      for_3d B->X[d] = FMM.Jbodies[b][d];
      B->SRC = FMM.Jbodies[b][3];
      for_4d B->TRG[d] = FMM.Ibodies[b][d];
    }
    Bodies jbodies = bodies;
    vec3 localDipole = ewald.getDipole(bodies, FMM.RGlob[0]);
    vec3 globalDipole = baseMPI.allreduceVec3(localDipole);
    numBodies = baseMPI.allreduceInt(bodies.size());

    ewald.dipoleCorrection(bodies, globalDipole, numBodies, cycle);
    logger::startTimer("Total Ewald");
    Bounds bounds = boundBox.getBounds(bodies);
    Bodies buffer = bodies;
    Cells cells = buildTree.buildTree(bodies, buffer, bounds);
    Bodies bodies2 = bodies;
    ewald.initTarget(bodies);
    for (int i=0; i<FMM.MPISIZE; i++) {
      if (args.verbose) std::cout << "Ewald loop           : " << i+1 << "/" << FMM.MPISIZE << std::endl;
      if (FMM.MPISIZE > 1) baseMPI.shiftBodies(jbodies);
      bounds = boundBox.getBounds(jbodies);
      buffer = jbodies;
      Cells jcells = buildTree.buildTree(jbodies, buffer, bounds);
      ewald.wavePart(bodies, jbodies);
      ewald.realPart(cells, jcells);
    }
    ewald.selfTerm(bodies);
    double potSum = verify.getSumScalar(bodies);
    double potSum2 = verify.getSumScalar(bodies2);
    double accDif = verify.getDifVector(bodies, bodies2);
    double accNrm = verify.getNrmVector(bodies);
    logger::printTitle("FMM vs. direct");
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
  FMM.deallocate();

}
