#include "base_mpi.h"
#include "args.h"
#include "ewald.h"
#include "verify.h"
#include "parallelfmm.h"
using namespace exafmm;

static const real_t Celec = 332.0716;

void splitRange(int & begin, int & end, int iSplit, int numSplit) {
  assert(end > begin);
  int size = end - begin;
  int increment = size / numSplit;
  int remainder = size % numSplit;
  begin += iSplit * increment + std::min(iSplit,remainder);
  end = begin + increment;
  if (remainder > iSplit) end++;
}

void directVanDerWaals(int nglobal, int * icpumap, int * atype,
                       double * x, double * p, double * f,
                       double cuton, double cutoff, double cycle,
                       int numTypes, double * rscale, double * gscale, double * fgscale) {
  for (int i=0; i<nglobal; i++) {
    if (icpumap[i] == 1) {
      int atypei = atype[i]-1;
      real_t pp = 0, fx = 0, fy = 0, fz = 0;
      for (int j=0; j<nglobal; j++) {
        vec3 dX;
        for (int d=0; d<3; d++) dX[d] = x[3*i+d] - x[3*j+d];
        wrap(dX, cycle);
        real_t R2 = norm(dX);
        if (R2 != 0) {
          int atypej = atype[j]-1;
          real_t rs = rscale[atypei*numTypes+atypej];
          real_t gs = gscale[atypei*numTypes+atypej];
          real_t fgs = fgscale[atypei*numTypes+atypej];
          real_t R2s = R2 * rs;
          real_t invR2 = 1.0 / R2s;
          real_t invR6 = invR2 * invR2 * invR2;
          real_t cuton2 = cuton * cuton;
          real_t cutoff2 = cutoff * cutoff;
          if (R2 < cutoff2) {
            real_t tmp = 0, dtmp = 0;
            if (cuton2 < R2) {
              real_t tmp1 = (cutoff2 - R2) / ((cutoff2-cuton2)*(cutoff2-cuton2)*(cutoff2-cuton2));
              real_t tmp2 = tmp1 * (cutoff2 - R2) * (cutoff2 - 3 * cuton2 + 2 * R2);
              tmp = invR6 * (invR6 - 1) * tmp2;
              dtmp = invR6 * (invR6 - 1) * 12 * (cuton2 - R2) * tmp1
                - 6 * invR6 * (invR6 + (invR6 - 1) * tmp2) * tmp2 / R2;
            } else {
              tmp = invR6 * (invR6 - 1);
              dtmp = invR2 * invR6 * (2 * invR6 - 1);
            }
            dtmp *= fgs;
            pp += gs * tmp;
            fx += dX[0] * dtmp;
            fy += dX[1] * dtmp;
            fz += dX[2] * dtmp;
          }
        }
      }
      p[i] += pp;
      f[3*i+0] -= fx;
      f[3*i+1] -= fy;
      f[3*i+2] -= fz;
    }
  }
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

  print("Coulomb");
  start("Total FMM");

  std::vector<double> x(3*nglobal);
  std::vector<double> q(nglobal);
  std::vector<double> p(nglobal, 0);
  std::vector<double> f(3*nglobal, 0);
  std::vector<double> p2(nglobal, 0);
  std::vector<double> f2(3*nglobal, 0);
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

  // fmm_init
  start("Partition");
  FMM.partitioner(gatherLevel);
  stop("Partition");
  int iX[3] = {0, 0, 0};
  FMM.R0 = 0.5 * max(cycle) / FMM.numPartition[FMM.maxGlobLevel][0];
  for_3d FMM.RGlob[d] = FMM.R0 * FMM.numPartition[FMM.maxGlobLevel][d];
  FMM.getGlobIndex(iX,FMM.MPIRANK,FMM.maxGlobLevel);
  for_3d FMM.X0[d] = 2 * FMM.R0 * (iX[d] + .5);

  // fmm_partition
  int nlocal = 0;
  for (int i=0; i<nglobal; i++) {
    if (icpumap[i] == 1) nlocal++;
  }
  FMM.numBodies = nlocal;
  FMM.Jbodies.resize(nlocal);
  for (int i=0,b=0; i<nglobal; i++) {
    if (icpumap[i] == 1) {
      FMM.Jbodies[b][0] = x[3*i+0];
      FMM.Jbodies[b][1] = x[3*i+1];
      FMM.Jbodies[b][2] = x[3*i+2];
      FMM.Jbodies[b][3] = q[i];
      FMM.Index[b] = i;
      b++;
    }
  }
  FMM.partitionComm();
  for (int i=0; i<nglobal; i++) {
    icpumap[i] = 0;
  }
  for (int b=0; b<FMM.numBodies; b++) {
    int i = FMM.Index[b];
    x[3*i+0] = FMM.Jbodies[b][0];
    x[3*i+1] = FMM.Jbodies[b][1];
    x[3*i+2] = FMM.Jbodies[b][2];
    q[i] = FMM.Jbodies[b][3];
    icpumap[i] = 1;
  }
  
  // fmm_coulomb
  nlocal = 0;
  for (int i=0; i<nglobal; i++) {
    if (icpumap[i] == 1) nlocal++;
    else icpumap[i] = 0;
  }
  FMM.numBodies = nlocal;
  FMM.Jbodies.resize(nlocal);
  for (int i=0,b=0; i<nglobal; i++) {
    if (icpumap[i] == 1) {
      FMM.Jbodies[b][0] = x[3*i+0];
      FMM.Jbodies[b][1] = x[3*i+1];
      FMM.Jbodies[b][2] = x[3*i+2];
      FMM.Jbodies[b][3] = q[i];
      FMM.Index[b] = i;
      FMM.Ibodies[b] = 0;
      b++;
    }
  }
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
  for (int b=0; b<FMM.numBodies; b++) {
    int i = FMM.Index[b];
    p[i]     += FMM.Ibodies[b][0] * FMM.Jbodies[b][3] * Celec;
    f[3*i+0] += FMM.Ibodies[b][1] * FMM.Jbodies[b][3] * Celec;
    f[3*i+1] += FMM.Ibodies[b][2] * FMM.Jbodies[b][3] * Celec;
    f[3*i+2] += FMM.Ibodies[b][3] * FMM.Jbodies[b][3] * Celec;
  }

  // ewald_coulomb
  start("Total Ewald");
  nlocal = 0;
  for (int i=0; i<nglobal; i++) {
    if (icpumap[i] == 1) nlocal++;
    else icpumap[i] = 0;
  }
  FMM.numBodies = nlocal;
  FMM.Jbodies.resize(nlocal);
  for (int i=0,b=0; i<nglobal; i++) {
    if (icpumap[i] == 1) {
      FMM.Jbodies[b][0] = x[3*i+0];
      FMM.Jbodies[b][1] = x[3*i+1];
      FMM.Jbodies[b][2] = x[3*i+2];
      FMM.Jbodies[b][3] = q[i];
      FMM.Index[b] = i;
      FMM.Ibodies[b] = 0;
      b++;
    }
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
    int i = FMM.Index[b];
    p2[i]     += FMM.Ibodies[b][0] * FMM.Jbodies[b][3] * Celec;
    f2[3*i+0] += FMM.Ibodies[b][1] * FMM.Jbodies[b][3] * Celec;
    f2[3*i+1] += FMM.Ibodies[b][2] * FMM.Jbodies[b][3] * Celec;
    f2[3*i+2] += FMM.Ibodies[b][3] * FMM.Jbodies[b][3] * Celec;
  }
  stop("Total Ewald");

  // verify
  double potSum=0, potSum2=0, accDif=0, accNrm=0;
  for (int i=0; i<nglobal; i++) {
    if (icpumap[i] == 1) {
      potSum += p[i];
      potSum2 += p2[i];
      accDif += (f[3*i+0] - f2[3*i+0]) * (f[3*i+0] - f2[3*i+0])
        + (f[3*i+1] - f2[3*i+1]) * (f[3*i+1] - f2[3*i+1])
        + (f[3*i+2] - f2[3*i+2]) * (f[3*i+2] - f2[3*i+2]);
      accNrm += f2[3*i+0] * f2[3*i+0]
        + f2[3*i+1] * f2[3*i+1]
        + f2[3*i+2] * f2[3*i+2];
    }
  }
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
  Verify verify;
  verify.print("Rel. L2 Error (pot)",potRel);
  verify.print("Rel. L2 Error (acc)",accRel);

  std::fill(p.begin(),p.end(),0);
  std::fill(f.begin(),f.end(),0);
  std::fill(p2.begin(),p2.end(),0);
  std::fill(f2.begin(),f2.end(),0);
  for (int b=0; b<FMM.numBodies; b++) {
    FMM.Ibodies[b] = 0;
  }
  
  // fmm_vanderwaals
  print("Van der Waals");
  start("FMM Van der Waals");
  nlocal = 0;
  for (int i=0; i<nglobal; i++) {
    if (icpumap[i] == 1) nlocal++;
    else icpumap[i] = 0;
  }
  FMM.numBodies = nlocal;
  FMM.Jbodies.resize(nlocal);
  for (int i=0,b=0; i<nglobal; i++) {
    if (icpumap[i] == 1) {
      FMM.Jbodies[b][0] = x[3*i+0];
      FMM.Jbodies[b][1] = x[3*i+1];
      FMM.Jbodies[b][2] = x[3*i+2];
      FMM.Jbodies[b][3] = q[i];
      FMM.Index[b] = i;
      FMM.Ibodies[b] = 0;
      b++;
    }
  }
  FMM.vanDerWaals(cuton, cutoff, nat, rscale, gscale, fgscale);
  for (int b=0; b<FMM.numBodies; b++) {
    int i = FMM.Index[b];
    p[i]     += FMM.Ibodies[b][0];
    f[3*i+0] += FMM.Ibodies[b][1];
    f[3*i+1] += FMM.Ibodies[b][2];
    f[3*i+2] += FMM.Ibodies[b][3];
  }
  stop("FMM Van der Waals");

  // Direct Van der Waals
  start("Direct Van der Waals");
  directVanDerWaals(nglobal, &icpumap[0], &atype[0], &x[0], &p2[0], &f2[0],
                    cuton, cutoff, cycle[0], nat, &rscale[0], &gscale[0], &fgscale[0]);
  stop("Direct Van der Waals");

    // verify
  potSum=0, potSum2=0, accDif=0, accNrm=0;
  for (int i=0; i<nglobal; i++) {
    if (icpumap[i] == 1) {
      potSum += p[i];
      potSum2 += p2[i];
      accDif += (f[3*i+0] - f2[3*i+0]) * (f[3*i+0] - f2[3*i+0])
        + (f[3*i+1] - f2[3*i+1]) * (f[3*i+1] - f2[3*i+1])
        + (f[3*i+2] - f2[3*i+2]) * (f[3*i+2] - f2[3*i+2]);
      accNrm += f2[3*i+0] * f2[3*i+0]
        + f2[3*i+1] * f2[3*i+1]
        + f2[3*i+2] * f2[3*i+2];
    }
  }
  print("FMM vs. direct");
  potSumGlob, potSumGlob2, accDifGlob, accNrmGlob;
  MPI_Reduce(&potSum,  &potSumGlob,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&potSum2, &potSumGlob2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&accDif,  &accDifGlob,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&accNrm,  &accNrmGlob,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  potDifGlob = (potSumGlob - potSumGlob2) * (potSumGlob - potSumGlob2);
  potNrmGlob = potSumGlob * potSumGlob;
  potRel = std::sqrt(potDifGlob/potNrmGlob);
  accRel = std::sqrt(accDifGlob/accNrmGlob);
  verify.print("Rel. L2 Error (pot)",potRel);
  verify.print("Rel. L2 Error (acc)",accRel);
}
