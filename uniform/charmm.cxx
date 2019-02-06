#include "base_mpi.h"
#include "args.h"
#include "ewald.h"
#include "verify.h"
#include "parallelfmm.h"
using namespace exafmm;

void splitRange(int & begin, int & end, int iSplit, int numSplit) {
  assert(end > begin);
  int size = end - begin;
  int increment = size / numSplit;
  int remainder = size % numSplit;
  begin += iSplit * increment + std::min(iSplit,remainder);
  end = begin + increment;
  if (remainder > iSplit) end++;
}

int main(int argc, char ** argv) {
  const int ksize = 14;
  const int nat = 16;
  const vec3 cycle = 10 * M_PI;
  const real_t alpha = 10 / max(cycle);
  const real_t sigma = .25 / M_PI;
  const real_t cuton = 9.5;
  const real_t cutoff = 10;
  Args args(argc, argv);
  BaseMPI baseMPI;
  Ewald ewald(ksize, alpha, sigma, cutoff, cycle);
  Verify verify;

  const int nglobal = args.numBodies;
  args.numBodies /= baseMPI.mpisize;
  const int numBodies = args.numBodies;
  const int ncrit = args.ncrit;
  const int maxLevel = numBodies >= ncrit ? 1 + int(log(numBodies / ncrit)/M_LN2/3) : 0;
  const int gatherLevel = 1;
  const int numImages = args.images;
  if (numImages > 0 && int(log2(baseMPI.mpisize)) % 3 != 0) {
    if (baseMPI.mpirank==0) printf("Warning: MPISIZE must be a power of 8 for periodic domain to be square\n");
  }

  ParallelFMM FMM(numBodies, maxLevel, numImages);
  VERBOSE = FMM.MPIRANK == 0;
  args.verbose = VERBOSE;
  print("FMM Parameters");
  args.print(stringLength);

  print("FMM Profiling");
  start("Total FMM");

  std::vector<double> x(3*nglobal);
  std::vector<double> q(nglobal);
  std::vector<double> p(nglobal, 0);
  std::vector<double> f(3*nglobal, 0);
  std::vector<int> icpumap(nglobal,0);
  std::vector<int> atype(nglobal);
  std::vector<int> numex(nglobal);
  std::vector<int> natex(nglobal);
  std::vector<double> rscale(nat*nat);
  std::vector<double> gscale(nat*nat);
  std::vector<double> fgscale(nat*nat);

  double average = 0;
  for (int i=0; i<nglobal; i++) {
    for_3d x[3*i+d] = drand48() * cycle[d];
    q[i] = drand48();
    average += q[i];
  }
  average /= nglobal;
  for (int i=0; i<nglobal; i++)	{
    q[i] -= average;
  }
  for (int i=0; i<nglobal; i++)	{
    numex[i] = 1;
    if (i % 2 == 0) {
      natex[i] = i+1;
    } else {
      natex[i] = i-1;
    }
    atype[i] = 1;
  }
  for (int i=0; i<nat*nat; i++) {
    rscale[i] = 1;
    gscale[i] = 0.0001;
    fgscale[i] = gscale[i];
  }
  int ista = 0;
  int iend = nglobal;
  splitRange(ista, iend, baseMPI.mpirank, baseMPI.mpisize);
  for (int i=ista; i<iend; i++) {
    icpumap[i] = 1;
  }

  // Init
  start("Partition");
  FMM.partitioner(gatherLevel);
  stop("Partition");
  int iX[3] = {0, 0, 0};
  FMM.R0 = 0.5 * max(cycle) / FMM.numPartition[FMM.maxGlobLevel][0];
  for_3d FMM.RGlob[d] = FMM.R0 * FMM.numPartition[FMM.maxGlobLevel][d];
  FMM.getGlobIndex(iX,FMM.MPIRANK,FMM.maxGlobLevel);
  for_3d FMM.X0[d] = 2 * FMM.R0 * (iX[d] + .5);

  // Partition
  int nlocal = 0;
  for (int i=0; i<nglobal; i++) {
    if (icpumap[i] == 1) nlocal++;
  }
  FMM.numBodies = nlocal;
  FMM.Jbodies.resize(nlocal);
  int b = 0;
  for (int i=0; i<nglobal; i++) {
    if (icpumap[i] == 1) {
      FMM.Jbodies[b][0] = x[3*i+0];
      FMM.Jbodies[b][1] = x[3*i+1];
      FMM.Jbodies[b][2] = x[3*i+2];
      FMM.Jbodies[b][3] = q[i];
      b++;
    }
  }
  FMM.partitionComm();

  start("Grow tree");
  FMM.sortBodies();
  FMM.buildTree();
  stop("Grow tree");
  start("Comm LET bodies");
  FMM.P2PSend();
  FMM.P2PRecv();
  stop("Comm LET bodies");
  FMM.upwardPass();
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
  FMM.periodicM2L();
  FMM.globL2L();
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
}
