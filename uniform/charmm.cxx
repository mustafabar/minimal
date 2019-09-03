#include "base_mpi.h"
#include "ewald.h"
#include "parallelfmm.h"
#include <fstream>
using namespace exafmm;

static const real_t Celec = 332.0716;
static const int shift = 29;
static const int mask = ~(0x7U << shift);

BaseMPI * baseMPI;
ParallelFMM * FMM;
Ewald * ewald;

int wrap2(vec4 & X, const real_t & cycle) {
  int iwrap = 0;
  for (int d=0; d<3; d++) {
    if(X[d] < 0) {
      X[d] += cycle;
      iwrap |= 1 << d;
    }
    if(X[d] > cycle) {
      X[d] -= cycle;
      iwrap |= 1 << d;
    }
  }
  return iwrap;
}

void unwrap2(vec4 & X, const real_t & cycle, const int & iwrap) {
  for (int d=0; d<3; d++) {
    if((iwrap >> d) & 1) X[d] += (X[d] > (cycle/2) ? -cycle : cycle);
  }
}

void splitRange(int & begin, int & end, int iSplit, int numSplit) {
  assert(end > begin);
  int size = end - begin;
  int increment = size / numSplit;
  int remainder = size % numSplit;
  begin += iSplit * increment + std::min(iSplit,remainder);
  end = begin + increment;
  if (remainder > iSplit) end++;
}

extern "C" void fmm_init_(int & nglobal, int & images, int & verbose) {
  baseMPI = new BaseMPI;
  const int numBodies = nglobal / baseMPI->mpisize;
  const int ncrit = 32;
  const int maxLevel = numBodies >= ncrit ? 1 + int(log(numBodies / ncrit)/M_LN2/3) : 0;
  const int numImages = images;
  FMM = new ParallelFMM(numBodies, maxLevel, numImages);
  VERBOSE = verbose & (FMM->MPIRANK == 0);
  if (numImages > 0 && int(log2(FMM->MPISIZE)) % 3 != 0) {
    if (FMM->MPIRANK==0) printf("Warning: MPISIZE must be a power of 8 for periodic domain to be square\n");
  }
}

extern "C" void fmm_finalize_() {
  delete baseMPI;
  delete FMM;
}

extern "C" void fmm_partition_(int & nglobal, int * icpumap, double * x, double * q,
                               double * xold, double & cycle) {
  start("Partition");
  const int gatherLevel = 1;
  FMM->partitioner(gatherLevel);
  int iX[3] = {0, 0, 0};
  FMM->R0 = 0.5 * cycle / FMM->numPartition[FMM->maxGlobLevel][0];
  for_3d FMM->RGlob[d] = FMM->R0 * FMM->numPartition[FMM->maxGlobLevel][d];
  FMM->getGlobIndex(iX,FMM->MPIRANK,FMM->maxGlobLevel);
  for_3d FMM->X0[d] = 2 * FMM->R0 * (iX[d] + .5);

  int ista = 0;
  int iend = nglobal;
  splitRange(ista, iend, baseMPI->mpirank, baseMPI->mpisize);
  for (int i=ista; i<iend; i++) {
    icpumap[i] = 1;
  }
  int nlocal = 0;
  for (int i=0; i<nglobal; i++) {
    if (icpumap[i] == 1) nlocal++;
  }
  FMM->numBodies = nlocal;
  FMM->Jbodies.resize(nlocal);
  for (int i=0,b=0; i<nglobal; i++) {
    if (icpumap[i] == 1) {
      FMM->Jbodies[b][0] = x[3*i+0];
      FMM->Jbodies[b][1] = x[3*i+1];
      FMM->Jbodies[b][2] = x[3*i+2];
      FMM->Jbodies[b][3] = q[i];
      int iwrap = wrap2(FMM->Jbodies[b], cycle);
      FMM->Index[b] = i | (iwrap << shift);
      FMM->Ibodies[b][0] = xold[3*i+0];
      FMM->Ibodies[b][1] = xold[3*i+1];
      FMM->Ibodies[b][2] = xold[3*i+2];
      b++;
    }
  }
  FMM->partitionComm();
  stop("Partition");
  start("Grow tree");
  FMM->sortBodies();
  FMM->buildTree();
  stop("Grow tree");
  for (int i=0; i<nglobal; i++) {
    icpumap[i] = 0;
  }
  for (int b=0; b<FMM->numBodies; b++) {
    int i = FMM->Index[b] & mask;
    int iwrap = unsigned(FMM->Index[b]) >> shift;
    unwrap2(FMM->Jbodies[b], cycle, iwrap);
    x[3*i+0] = FMM->Jbodies[b][0];
    x[3*i+1] = FMM->Jbodies[b][1];
    x[3*i+2] = FMM->Jbodies[b][2];
    q[i] = FMM->Jbodies[b][3];
    xold[3*i+0] = FMM->Ibodies[b][0];
    xold[3*i+1] = FMM->Ibodies[b][1];
    xold[3*i+2] = FMM->Ibodies[b][2];
    icpumap[i] = 1;
  }
}

extern "C" void fmm_coulomb_(int & nglobal, int * icpumap,
                             double * x, double * q, double * p, double * f,
                             double & cycle) {
  int nlocal = 0;
  for (int i=0; i<nglobal; i++) {
    if (icpumap[i] == 1) nlocal++;
    else icpumap[i] = 0;
  }
  FMM->numBodies = nlocal;
  FMM->Jbodies.resize(nlocal);
  for (int i=0,b=0; i<nglobal; i++) {
    if (icpumap[i] == 1) {
      FMM->Jbodies[b][0] = x[3*i+0];
      FMM->Jbodies[b][1] = x[3*i+1];
      FMM->Jbodies[b][2] = x[3*i+2];
      FMM->Jbodies[b][3] = q[i] == 0 ? EPS : q[i];
      FMM->Ibodies[b] = 0;
      int iwrap = wrap2(FMM->Jbodies[b], cycle);
      FMM->Index[b] = i | (iwrap << shift);
      b++;
    }
  }
  FMM->sortBodies();
  start("Comm LET bodies");
  FMM->P2PSend();
  FMM->P2PRecv();
  stop("Comm LET bodies");
  FMM->upwardPass();
  start("Comm LET cells");
  for (int lev=FMM->maxLevel; lev>0; lev--) {
    MPI_Barrier(MPI_COMM_WORLD);
    FMM->M2LSend(lev);
    FMM->M2LRecv(lev);
  }
  FMM->rootGather();
  stop("Comm LET cells");
  FMM->globM2M();
  FMM->globM2L();
  FMM->periodicM2L();
  FMM->globL2L();
  FMM->downwardPass();
  vec3 localDipole = FMM->getDipole();
  vec3 globalDipole = baseMPI->allreduceVec3(localDipole);
  int globalNumBodies = baseMPI->allreduceInt(FMM->numBodies);
  FMM->dipoleCorrection(globalDipole, globalNumBodies);
  for (int b=0; b<FMM->numBodies; b++) { 
    int i = FMM->Index[b] & mask;
    p[i]     += FMM->Ibodies[b][0] * FMM->Jbodies[b][3] * Celec;
    f[3*i+0] += FMM->Ibodies[b][1] * FMM->Jbodies[b][3] * Celec;
    f[3*i+1] += FMM->Ibodies[b][2] * FMM->Jbodies[b][3] * Celec;
    f[3*i+2] += FMM->Ibodies[b][3] * FMM->Jbodies[b][3] * Celec;
  }
}

extern "C" void ewald_coulomb_(int & nglobal, int * icpumap, double * x, double * q, double * p, double * f,
                               int & ksize, double & alpha, double & sigma, double & cutoff, double & cycle) {
  ewald = new Ewald(ksize, alpha, sigma, cutoff, cycle);
  int nlocal = 0;
  for (int i=0; i<nglobal; i++) {
    if (icpumap[i] == 1) nlocal++;
    else icpumap[i] = 0;
  }
  FMM->numBodies = nlocal;
  FMM->Jbodies.resize(nlocal);
  for (int i=0,b=0; i<nglobal; i++) {
    if (icpumap[i] == 1) {
      FMM->Jbodies[b][0] = x[3*i+0];
      FMM->Jbodies[b][1] = x[3*i+1];
      FMM->Jbodies[b][2] = x[3*i+2];
      FMM->Jbodies[b][3] = q[i];
      int iwrap = wrap2(FMM->Jbodies[b], cycle);
      FMM->Index[b] = i | (iwrap << shift);
      FMM->Ibodies[b] = 0;
      b++;
    }
  }
  FMM->sortBodies();
  start("Ewald real part");
  FMM->ewaldRealPart(alpha,cutoff);
  stop("Ewald real part");
  FMM->Ibodies.resize(FMM->numBodies);
  FMM->Jbodies.resize(FMM->numBodies);
  start("Ewald wave part");
  Waves waves = ewald->initWaves();
  ewald->dft(waves,FMM->Jbodies);
  waves = baseMPI->allreduceWaves(waves);
  ewald->wavePart(waves);
  ewald->idft(waves,FMM->Ibodies,FMM->Jbodies);
  stop("Ewald wave part");
  ewald->selfTerm(FMM->Ibodies, FMM->Jbodies);
  for (int b=0; b<FMM->numBodies; b++) {
    int i = FMM->Index[b] & mask;
    p[i]     += FMM->Ibodies[b][0] * FMM->Jbodies[b][3] * Celec;
    f[3*i+0] += FMM->Ibodies[b][1] * FMM->Jbodies[b][3] * Celec;
    f[3*i+1] += FMM->Ibodies[b][2] * FMM->Jbodies[b][3] * Celec;
    f[3*i+2] += FMM->Ibodies[b][3] * FMM->Jbodies[b][3] * Celec;
  }
  delete ewald;
}

extern "C" void coulomb_exclusion_(int & nglobal, int * icpumap,
                                   double * x, double * q, double * p, double * f,
                                   double & cycle, int * numex, int * natex) {
  for (int i=0, ic=0; i<nglobal; i++) {
    if (icpumap[i] == 1) {
      real_t pp = 0, fx = 0, fy = 0, fz = 0;
      for (int jc=0; jc<numex[i]; jc++, ic++) {
        int j = natex[ic]-1;
        vec3 dX;
        for (int d=0; d<3; d++) dX[d] = x[3*i+d] - x[3*j+d];
        wrap(dX, cycle);
        real_t R2 = norm(dX);
        real_t invR = 1 / std::sqrt(R2);
        if (R2 == 0) invR = 0;
        real_t invR3 = q[j] * invR * invR * invR;
        pp += q[j] * invR;
        fx += dX[0] * invR3;
        fy += dX[1] * invR3;
        fz += dX[2] * invR3;
      }
      p[i] -= pp * q[i] * Celec;
      f[3*i+0] += fx * q[i] * Celec;
      f[3*i+1] += fy * q[i] * Celec;
      f[3*i+2] += fz * q[i] * Celec;
    } else {
      ic += numex[i];
    }
  }
}

extern "C" void fmm_vanderwaals_(int & nglobal, int * icpumap, int * atype,
                                 double * x, double * p, double * f,
                                 double & cuton, double & cutoff, double & cycle,
                                 int & nat, double * rscale, double * gscale, double * fgscale) {
  int nlocal = 0;
  for (int i=0; i<nglobal; i++) {
    if (icpumap[i] == 1) nlocal++;
    else icpumap[i] = 0;
  }
  FMM->numBodies = nlocal;
  FMM->Jbodies.resize(nlocal);
  for (int i=0,b=0; i<nglobal; i++) {
    if (icpumap[i] == 1) {
      FMM->Jbodies[b][0] = x[3*i+0];
      FMM->Jbodies[b][1] = x[3*i+1];
      FMM->Jbodies[b][2] = x[3*i+2];
      FMM->Jbodies[b][3] = atype[i] - .5;
      int iwrap = wrap2(FMM->Jbodies[b], cycle);
      FMM->Index[b] = i | (iwrap << shift);
      FMM->Ibodies[b] = 0;
      b++;
    }
  }
  FMM->sortBodies();
  start("Comm LET bodies");
  FMM->P2PSend();
  FMM->P2PRecv();
  stop("Comm LET bodies");
  FMM->vanDerWaals(cuton, cutoff, nat, rscale, gscale, fgscale);
  for (int b=0; b<FMM->numBodies; b++) {
    int i = FMM->Index[b] & mask;
    p[i]     += FMM->Ibodies[b][0];
    f[3*i+0] += FMM->Ibodies[b][1];
    f[3*i+1] += FMM->Ibodies[b][2];
    f[3*i+2] += FMM->Ibodies[b][3];
  }
}

extern "C" void direct_vanderwaals_(int & nglobal, int * icpumap, int * atype,
                                    double * x, double * p, double * f,
                                    double & cuton, double & cutoff, double & cycle,
                                    int & nat, double * rscale, double * gscale, double * fgscale) {
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
          real_t rs = rscale[atypei*nat+atypej];
          real_t gs = gscale[atypei*nat+atypej];
          real_t fgs = fgscale[atypei*nat+atypej];
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

extern "C" void vanderwaals_exclusion_(int & nglobal, int * icpumap, int * atype,
                                       double * x, double * p, double * f,
                                       double & cuton, double & cutoff, double & cycle,
                                       int & numTypes, double * rscale, double * gscale,
                                       double * fgscale, int * numex, int * natex) {
  for (int i=0, ic=0; i<nglobal; i++) {
    if (icpumap[i] == 1) {
      int atypei = atype[i]-1;
      for (int jc=0; jc<numex[i]; jc++, ic++) {
        int j = natex[ic]-1;
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
            p[i] -= gs * tmp;
            f[3*i+0] += dX[0] * dtmp;
            f[3*i+1] += dX[1] * dtmp;
            f[3*i+2] += dX[2] * dtmp;
          }
        }
      }
    } else {
      ic += numex[i];
    }
  }
}

#ifndef LIBRARY
int main(int argc, char ** argv) {
  int nglobal = 2991;
  int images = 6;
  int ksize = 14;
  int nat = 2;
  int verbose = 1;
  real_t cycle = 10 * M_PI;
  real_t alpha = 10 / cycle;
  real_t sigma = .25 / M_PI;
  real_t cuton = 9.5;
  real_t cutoff = 10;

  std::vector<double> x(3*nglobal);
  std::vector<double> q(nglobal);
  std::vector<double> xold(3*nglobal);
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
    for_3d x[3*i+d] = drand48() * cycle;
    for_3d xold[3*i+d] = drand48() * cycle;
    q[i] = drand48();
    average += q[i];
  }
  average /= nglobal;
  for (int i=0; i<nglobal; i++)	{
    q[i] -= average;
  }
  for (int i=0; i<nglobal; i++)	{
    numex[i] = 2;
    if (i % 2 == 0) {
      natex[i] = i+1;
    } else {
      natex[i] = i-1;
    }
    atype[i] = 2 - ((i % 3) == 0);
  }
  for (int i=0; i<nat*nat; i++) {
    rscale[i] = 1;
    gscale[i] = 0.0001;
    fgscale[i] = gscale[i];
  }

  ksize = 13;
  cycle = 31.1149;
  alpha = 0.35126591448392;
  cuton = 10.818015058191;
  cutoff = 11.3873844146729;
  rscale[0] = 0.1007443196;
  rscale[1] = 0.3172922689;
  rscale[2] = 0.3172922689;
  rscale[3] = 6.2495773825;
  gscale[0] = 0.6084000000;
  gscale[1] = 0.3345827252;
  gscale[2] = 0.3345827252;
  gscale[3] = 0.1840000000;
  fgscale[0] = 0.3677570643;
  fgscale[1] = 0.6369630721;
  fgscale[2] = 0.6369630721;
  fgscale[3] = 6.8995334303;
  std::ifstream file("initial.dat");
  std::string line;
  for (int i=0; i<nglobal; i++) {
    std::getline(file, line);
    std::istringstream iss(line);
    iss >> x[3*i+0] >> x[3*i+1] >> x[3*i+2] >> q[i];   
  }
  file.close();

  fmm_init_(nglobal, images, verbose);  
  print("Coulomb");
  start("Total FMM");
  fmm_partition_(nglobal, &icpumap[0], &x[0], &q[0], &xold[0], cycle);
  fmm_coulomb_(nglobal, &icpumap[0], &x[0], &q[0], &p[0], &f[0], cycle);
  //coulomb_exclusion_(nglobal, &icpumap[0], &x[0], &q[0], &p[0], &f[0],
  //                   cycle, &numex[0], &natex[0]);
  stop("Total FMM");
  start("Total Ewald");
  ewald_coulomb_(nglobal, &icpumap[0], &x[0], &q[0], &p2[0], &f2[0],
                 ksize, alpha, sigma, cutoff, cycle);
  //coulomb_exclusion_(nglobal, &icpumap[0], &x[0], &q[0], &p2[0], &f2[0],
  //                   cycle, &numex[0], &natex[0]);
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
  print("Rel. L2 Error (pot)",potRel);
  print("Rel. L2 Error (acc)",accRel);

  std::fill(p.begin(),p.end(),0);
  std::fill(f.begin(),f.end(),0);
  std::fill(p2.begin(),p2.end(),0);
  std::fill(f2.begin(),f2.end(),0);
  
  print("Van der Waals");
  start("FMM Van der Waals");
  fmm_vanderwaals_(nglobal, &icpumap[0], &atype[0], &x[0], &p[0], &f[0],
                   cuton, cutoff, cycle, nat, &rscale[0], &gscale[0], &fgscale[0]);
  stop("FMM Van der Waals");
  start("Direct Van der Waals");
  direct_vanderwaals_(nglobal, &icpumap[0], &atype[0], &x[0], &p2[0], &f2[0],
                      cuton, cutoff, cycle, nat, &rscale[0], &gscale[0], &fgscale[0]);
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
  print("Rel. L2 Error (pot)",potRel);
  print("Rel. L2 Error (acc)",accRel);
  fmm_finalize_();
}
#endif
