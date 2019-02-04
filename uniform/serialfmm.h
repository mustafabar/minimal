#include <mpi.h>
#include "kernels.h"

namespace exafmm {
  class SerialFMM : public Kernel {
  protected:
    int bodiesDispl[26];
    int bodiesCount[26];
    int sendBodiesDispl[1024];
    int sendBodiesCount[1024];
    int recvBodiesDispl[1024];
    int recvBodiesCount[1024];
    int multipoleDispl[10][26];
    int multipoleCount[10][26];
    int leafsDispl[26];
    int leafsCount[26];
    ivec3 IX[10];
    int gatherLevel;
    MPI_Comm MPI_COMM_LOCAL, MPI_COMM_GLOBAL;

  public:
    vec3 X0;
    real_t R0;
    vec3 RGlob;
    std::vector<int> Index;
    std::vector<int> Rank;
    std::vector<int> sendIndex;
    std::vector<int> recvIndex;
    std::vector<Range> Leafs;
    std::vector<Range> sendLeafs;
    std::vector<Range> recvLeafs;
    std::vector<vec4> Ibodies;
    std::vector<vec4> Jbodies;
    std::vector<cvecP> Multipole;
    std::vector<cvecP> Local;
    std::vector<cvecP> globMultipole;
    std::vector<cvecP> globLocal;
    std::vector<vec4> sendJbodies;
    std::vector<vec4> recvJbodies;
    std::vector<fcvecP> sendMultipole;
    std::vector<fcvecP> recvMultipole;

    
  private:
    void checkPartition(ivec3 &maxPartition) {
      int partitionSize = 1;
      for_3d partitionSize *= maxPartition[d];
      assert( MPISIZE == partitionSize );
      int mpisize = MPISIZE;
      while (mpisize > 0) {
	assert( mpisize % 8 == 0 || mpisize == 1 );
	mpisize /= 8;
      }
      ivec3 checkLevel, partition;
      partition = maxPartition;
      for( int d=0; d<3; d++ ) {
	int lev = 1;
	while( partition[d] != 1 ) {
	  int ndiv = 2;
	  if( (partition[d] % 3) == 0 ) ndiv = 3;
	  partition[d] /= ndiv;
	  lev++;
	}
	checkLevel[d] = lev - 1;
      }
      maxGlobLevel = std::max(std::max(checkLevel[0],checkLevel[1]),checkLevel[2]);
      numPartition[0] = 1;
      partition = maxPartition;
      for( int lev=1; lev<=maxGlobLevel; lev++ ) {
	for( int d=0; d<3; d++ ) {
	  int ndiv = 2;
	  if( (partition[d] % 3) == 0 ) ndiv = 3;
	  if( checkLevel[d] < maxGlobLevel && lev == 1 ) ndiv = 1;
	  numPartition[lev][d] = ndiv * numPartition[lev-1][d];
	  partition[d] /= ndiv;
	}
      }
    }

    void setSendCounts() {
      ivec3 leafsType, bodiesType;
      for_3d leafsType[d] = 1 << (d * maxLevel);
      bodiesType = leafsType * float(numBodies) / numLeafs * 4;
      int i = 0;
      ivec3 iX;
      bodiesDispl[0] = leafsDispl[0] = 0;
      for( iX[2]=-1; iX[2]<=1; iX[2]++ ) {
	for( iX[1]=-1; iX[1]<=1; iX[1]++ ) {
	  for( iX[0]=-1; iX[0]<=1; iX[0]++ ) {
	    if( iX[0] != 0 || iX[1] != 0 || iX[2] != 0 ) {
	      int zeros = 0;
	      for_3d zeros += iX[d] == 0;
	      bodiesCount[i] = bodiesType[zeros];
	      leafsCount[i] = leafsType[zeros];
	      if( i > 0 ) {
		bodiesDispl[i] = bodiesDispl[i-1] + bodiesCount[i-1];
		leafsDispl[i] = leafsDispl[i-1] + leafsCount[i-1];
	      }
	      i++;
	    }
	  }
	}
      }
      assert( numSendBodies >= bodiesDispl[25] + bodiesCount[25] );
      assert( bodiesDispl[25] + bodiesCount[25] > 0 );
      assert( numSendLeafs == leafsDispl[25] + leafsCount[25] );
      int sumSendCells = 0;
      for( int lev=1; lev<=maxLevel; lev++ ) {
	int multipoleType[3] = {8, 4*(1<<lev), 2*(1<<(2*lev))};
	multipoleDispl[lev][0] = 0;
	i = 0;
	for( iX[2]=-1; iX[2]<=1; iX[2]++ ) {
	  for( iX[1]=-1; iX[1]<=1; iX[1]++ ) {
	    for( iX[0]=-1; iX[0]<=1; iX[0]++ ) {
	      if( iX[0] != 0 || iX[1] != 0 || iX[2] != 0 ) {
		int zeros = 0;
		for_3d zeros += iX[d] == 0;
		multipoleCount[lev][i] = multipoleType[zeros];
		sumSendCells += multipoleCount[lev][i];
		if( i > 0 ) {
		  multipoleDispl[lev][i] = multipoleDispl[lev][i-1] + multipoleCount[lev][i-1];
		}
		i++;
	      }
	    }
	  }
	}
      }
      assert( numSendCells == sumSendCells );
    }

  protected:
    inline void getIndex(int i, ivec3 &iX, real_t diameter) const {
#if NOWRAP
      i = (i / 3) * 3;
#endif
      for_3d iX[d] = int((Jbodies[i][d] + R0 - X0[d]) / diameter);
    }

    inline void getIndex(ivec3 &iX, int index) const {
      iX = 0;
      int d = 0, level = 0;
      while (index != 0) {
        iX[d] += (index % 2) * (1 << level);
        index >>= 1;
        d = (d+1) % 3;
        if (d == 0) level++;
      }
    }

    inline void setGlobIndex(int i, ivec3 &iX) const {
#if NOWRAP
      i = (i / 3) * 3;
#endif
      for_3d iX[d] = int(Jbodies[i][d] / (2 * R0));
      iX %= numPartition[maxGlobLevel];
    }

    inline int getKey(ivec3 &iX, int level, bool levelOffset=true) const {
      int id = 0;
      if (levelOffset) id = ((1 << 3 * level) - 1) / 7;
      for(int lev=0; lev<level; ++lev ) {
        for_3d id += iX[d] % 2 << (3 * lev + d);
        iX >>= 1;
      }
      return id;
    }

    inline int getGlobKey(ivec3 &iX, int level) const {
      return iX[0] + (iX[1] + iX[2] * numPartition[level][1]) * numPartition[level][0];
    }

    void getCenter(vec3 &dX, int index, int level) const {
      real_t R = R0 / (1 << level);
      ivec3 iX = 0;
      getIndex(iX, index);
      for_3d dX[d] = X0[d] - R0 + (2 * iX[d] + 1) * R;
    }

    void sort(std::vector<vec4> &bodies, std::vector<vec4> &buffer, std::vector<int> &index,
              std::vector<int> &ibuffer, std::vector<int> &key) const {
      int Imax = key[0];
      int Imin = key[0];
      for( int i=0; i<numBodies; i++ ) {
	Imax = std::max(Imax,key[i]);
	Imin = std::min(Imin,key[i]);
      }
      int numBucket = Imax - Imin + 1;
      std::vector<int> bucket(numBucket);
      for( int i=0; i<numBucket; i++ ) bucket[i] = 0;
      for( int i=0; i<numBodies; i++ ) bucket[key[i]-Imin]++;
      for( int i=1; i<numBucket; i++ ) bucket[i] += bucket[i-1];
      for( int i=numBodies-1; i>=0; --i ) {
	bucket[key[i]-Imin]--;
	int inew = bucket[key[i]-Imin];
	ibuffer[inew] = index[i];
	buffer[inew] = bodies[i];
      }
    }

  public:
    void allocate(int N, int L, int Im) {
      maxLevel = L;
      numBodies = N;
      numImages = Im;
      numCells = ((1 << 3 * (L + 1)) - 1) / 7;
      numLeafs = 1 << 3 * L;
      numSendCells = 64 * L + 48 * ((1 << (L + 1)) - 2) + 12 * (((1 << (2 * L + 2)) - 1) / 3 - 1);
      numSendLeafs = 8 + 12 * (1 << L) + 6 * (1 << (2 * L));
      numSendBodies = numSendLeafs * float(numBodies) / numLeafs * 4;
      Index.resize(2*numBodies);
      Rank.resize(2*numBodies);
      sendIndex.resize(2*numBodies);
      recvIndex.resize(2*numBodies);
      Leafs.resize(27*numLeafs);
      sendLeafs.resize(numSendLeafs);
      recvLeafs.resize(numSendLeafs);
      Ibodies.resize(2*numBodies);
      Jbodies.resize(2*numBodies+numSendBodies);
      Multipole.resize(27*numCells);
      Local.resize(numCells);
      globMultipole.resize(2*MPISIZE);
      globLocal.resize(10);
      sendJbodies.resize(2*numBodies+numSendBodies);
      recvJbodies.resize(2*numBodies+numSendBodies);
      sendMultipole.resize(numSendCells);
      recvMultipole.resize(numSendCells);
    }

    void deallocate() {
    }

    inline void getGlobIndex(int *iX, int index, int level) const {
      iX[0] = index % numPartition[level][0];
      iX[1] = index / numPartition[level][0] % numPartition[level][1];
      iX[2] = index / numPartition[level][0] / numPartition[level][1];
    }

    void partitioner(int level) {
      int mpisize = MPISIZE;
      ivec3 maxPartition = 1;
      int dim = 0;
      while( mpisize != 1 ) {
	int ndiv = 2;
	if( (mpisize % 3) == 0 ) ndiv = 3;
	maxPartition[dim] *= ndiv;
	mpisize /= ndiv;
	dim = (dim + 1) % 3;
      }
      checkPartition(maxPartition);
      numGlobCells = 0;
      for( int lev=0; lev<=maxGlobLevel; lev++ ) {
	globLevelOffset[lev] = numGlobCells;
	numGlobCells += numPartition[lev][0] * numPartition[lev][1] * numPartition[lev][2];
      }
      getGlobIndex(IX[maxGlobLevel],MPIRANK,maxGlobLevel);
      for( int lev=maxGlobLevel; lev>0; lev-- ) {
	IX[lev-1] = IX[lev] * numPartition[lev-1] / numPartition[lev];
      }
      setSendCounts();
      gatherLevel = level;
      if(gatherLevel > maxGlobLevel) gatherLevel = maxGlobLevel;
#if EXAFMM_SERIAL
#else
      ivec3 numChild = numPartition[maxGlobLevel] / numPartition[gatherLevel];
      ivec3 iX = IX[maxGlobLevel] % numChild;
      int key = iX[0] + (iX[1] + iX[2] * numChild[1]) * numChild[0];
      int color = getGlobKey(IX[gatherLevel],gatherLevel);
      MPI_Comm_split(MPI_COMM_WORLD, color, key, &MPI_COMM_LOCAL);
      MPI_Comm_split(MPI_COMM_WORLD, key, color, &MPI_COMM_GLOBAL);
#endif
    }

    void sortBodies() {
      std::vector<int> key(numBodies);
      real_t diameter = 2 * R0 / (1 << maxLevel);
      ivec3 iX = 0;
      for( int i=0; i<numBodies; i++ ) {
	getIndex(i,iX,diameter);
	key[i] = getKey(iX,maxLevel);
      }
      sort(Jbodies,sendJbodies,Index,sendIndex,key);
      for( int i=0; i<numBodies; i++ ) {
	Index[i] = sendIndex[i];
	Jbodies[i] = sendJbodies[i];
      }
    }

    void buildTree() {
      int rankOffset = 13 * numLeafs;
      for( int i=rankOffset; i<numLeafs+rankOffset; i++ ) {
	Leafs[i].begin = Leafs[i].end = 0;
      }
      real_t diameter = 2 * R0 / (1 << maxLevel);
      ivec3 iX = 0;
      getIndex(0,iX,diameter);
      int ileaf = getKey(iX,maxLevel,false) + rankOffset;
      Leafs[ileaf].begin = 0;
      for( int i=0; i<numBodies; i++ ) {
	getIndex(i,iX,diameter);
	int inew = getKey(iX,maxLevel,false) + rankOffset;
	if( ileaf != inew ) {
	  Leafs[ileaf].end = Leafs[inew].begin = i;
	  ileaf = inew;
	}
      }
      Leafs[ileaf].end = numBodies;
    }

    void upwardPass() {
      int rankOffset = 13 * numCells;
      for( int i=0; i<numCells; i++ ) {
	Multipole[i+rankOffset] = 0;
	Local[i] = 0;
      }

      start("P2M");
      rankOffset = 13 * numLeafs;
      int levelOffset = ((1 << 3 * maxLevel) - 1) / 7 + 13 * numCells;
#pragma omp parallel for
      for (int i=0; i<numLeafs; i++) {
	vec3 center;
	getCenter(center,i,maxLevel);
	for (int j=Leafs[i+rankOffset].begin; j<Leafs[i+rankOffset].end; j++) {
	  vec3 dX;
          for_3d dX[d] = Jbodies[j][d] - center[d];
          P2M(dX,Jbodies[j][3],Multipole[i+levelOffset]);
	}
      }
      stop("P2M");

      start("M2M");
      rankOffset = 13 * numCells;
      for (int lev=maxLevel; lev>0; lev--) {
	int childOffset = ((1 << 3 * lev) - 1) / 7 + rankOffset;
	int parentOffset = ((1 << 3 * (lev - 1)) - 1) / 7 + rankOffset;
	real_t radius = R0 / (1 << lev);
#pragma omp parallel for schedule(static, 8)
	for (int i=0; i<(1 << 3 * lev); i++) {
	  int c = i + childOffset;
	  int p = (i >> 3) + parentOffset;
	  ivec3 iX;
	  iX[0] = 1 - (i & 1) * 2;
	  iX[1] = 1 - ((i / 2) & 1) * 2;
	  iX[2] = 1 - ((i / 4) & 1) * 2;
	  vec3 dX;
	  for_3d dX[d] = iX[d] * radius;
          M2M(dX,Multipole[c],Multipole[p]);
	}
      }
      stop("M2M");
    }

    void downwardPass() {
      start("M2L");
      ivec3 iXc;
      int DM2LC = DM2L;
      getGlobIndex(iXc,MPIRANK,maxGlobLevel);
      for (int lev=1; lev<=maxLevel; lev++) {
	if (lev==maxLevel) DM2LC = DP2P;
	int levelOffset = ((1 << 3 * lev) - 1) / 7;
	int nunit = 1 << lev;
	ivec3 nunitGlob = numPartition[maxGlobLevel] * nunit;
	ivec3 nxmin = -iXc * (nunit >> 1);
	ivec3 nxmax = (nunitGlob >> 1) + nxmin - 1;
	if (numImages != 0) {
	  nxmin -= (nunitGlob >> 1);
	  nxmax += (nunitGlob >> 1);
	}
	real_t diameter = 2 * R0 / (1 << lev);
#pragma omp parallel for
	for (int i=0; i<(1 << 3 * lev); i++) {
	  cvecP L = complex_t(0);
	  ivec3 iX = 0;
	  getIndex(iX,i);
	  ivec3 jXmin = (max(nxmin,(iX >> 1) - DM2L) << 1);
	  ivec3 jXmax = (min(nxmax,(iX >> 1) + DM2L) << 1) + 1;
	  ivec3 jX;
	  for (jX[2]=jXmin[2]; jX[2]<=jXmax[2]; jX[2]++) {
	    for (jX[1]=jXmin[1]; jX[1]<=jXmax[1]; jX[1]++) {
	      for (jX[0]=jXmin[0]; jX[0]<=jXmax[0]; jX[0]++) {
		if(jX[0] < iX[0]-DM2LC || iX[0]+DM2LC < jX[0] ||
		   jX[1] < iX[1]-DM2LC || iX[1]+DM2LC < jX[1] ||
		   jX[2] < iX[2]-DM2LC || iX[2]+DM2LC < jX[2]) {
		  ivec3 jXp = (jX + nunit) % nunit;
		  int j = getKey(jXp,lev);
		  jXp = (jX + nunit) / nunit;
#if EXAFMM_SERIAL
		  int rankOffset = 13 * numCells;
#else
		  int rankOffset = (jXp[0] + 3 * jXp[1] + 9 * jXp[2]) * numCells;
#endif
		  j += rankOffset;
		  vec3 dX;
                  for_3d dX[d]= (iX[d] - jX[d]) * diameter;
                  M2L(dX,Multipole[j],L);
		}
	      }
	    }
	  }
	  Local[i+levelOffset] += L;
	}
      }
      stop("M2L");

      start("L2L");
      for (int lev=1; lev<=maxLevel; lev++) {
	int childOffset = ((1 << 3 * lev) - 1) / 7;
	int parentOffset = ((1 << 3 * (lev - 1)) - 1) / 7;
	real_t radius = R0 / (1 << lev);
#pragma omp parallel for
	for (int i=0; i<(1 << 3 * lev); i++) {
	  int c = i + childOffset;
	  int p = (i >> 3) + parentOffset;
	  ivec3 iX;
	  iX[0] = (i & 1) * 2 - 1;
	  iX[1] = ((i / 2) & 1) * 2 - 1;
	  iX[2] = ((i / 4) & 1) * 2 - 1;
	  vec3 dX;
	  for_3d dX[d] = iX[d] * radius;
          L2L(dX,Local[p],Local[c]);
	}
      }
      stop("L2L");

      start("L2P");
      int rankOffset = 13 * numLeafs;
      int levelOffset = ((1 << 3 * maxLevel) - 1) / 7;
#pragma omp parallel for
      for (int i=0; i<numLeafs; i++) {
	vec3 center;
	getCenter(center,i,maxLevel);
	cvecP L = Local[i+levelOffset];
	for (int j=Leafs[i+rankOffset].begin; j<Leafs[i+rankOffset].end; j++) {
	  vec3 dX;
	  for_3d dX[d] = Jbodies[j][d] - center[d];
          L2P(dX,L,Ibodies[j]);
	}
      }
      stop("L2P");

      start("P2P");
      getGlobIndex(iXc,MPIRANK,maxGlobLevel);
      int nunit = 1 << maxLevel;
      ivec3 nunitGlob = numPartition[maxGlobLevel] * nunit;
      ivec3 nxmin = -iXc * nunit;
      ivec3 nxmax = nunitGlob + nxmin - 1;
      if (numImages != 0) {
	nxmin -= nunitGlob;
	nxmax += nunitGlob;
      }
#pragma omp parallel for
      for (int i=0; i<numLeafs; i++) {
	ivec3 iX = 0;
	getIndex(iX,i);
	ivec3 jXmin = max(nxmin,iX - DP2P);
	ivec3 jXmax = min(nxmax,iX + DP2P);
	ivec3 jX;
	for (jX[2]=jXmin[2]; jX[2]<=jXmax[2]; jX[2]++) {
	  for (jX[1]=jXmin[1]; jX[1]<=jXmax[1]; jX[1]++) {
	    for (jX[0]=jXmin[0]; jX[0]<=jXmax[0]; jX[0]++) {
	      ivec3 jXp = (jX + nunit) % nunit;
	      int j = getKey(jXp,maxLevel,false);
	      jXp = (jX + nunit) / nunit;
#if EXAFMM_SERIAL
	      int rankOffset = 13 * numLeafs;
#else
	      int rankOffset = (jXp[0] + 3 * jXp[1] + 9 * jXp[2]) * numLeafs;
#endif
	      j += rankOffset;
	      rankOffset = 13 * numLeafs;
	      jXp = (jX + iXc * nunit + nunitGlob) / nunitGlob;
	      vec3 periodic;
	      for_3d periodic[d] = (jXp[d] - 1) * 2 * RGlob[d];
	      P2P(Ibodies,Leafs[i+rankOffset].begin,Leafs[i+rankOffset].end,
                  Jbodies,Leafs[j].begin,Leafs[j].end,periodic);
	    }
	  }
	}
      }
      stop("P2P");
    }

    void periodicM2L() {
      cvecP M;
#if EXAFMM_SERIAL
      M = Multipole[13*numCells];
#else
      M = globMultipole[0];
#endif
      cvecP L = complex_t(0);
      for( int lev=1; lev<numImages; lev++ ) {
	vec3 diameter = RGlob * 2 * std::pow(3.,lev-1);
	ivec3 jX;
	for( jX[2]=-4; jX[2]<=4; jX[2]++ ) {
	  for( jX[1]=-4; jX[1]<=4; jX[1]++ ) {
	    for( jX[0]=-4; jX[0]<=4; jX[0]++ ) {
	      if(jX[0] < -1 || 1 < jX[0] ||
		 jX[1] < -1 || 1 < jX[1] ||
		 jX[2] < -1 || 1 < jX[2]) {
		vec3 dX;
		for_3d dX[d] = jX[d] * diameter[d];
                M2L(dX,M,L);
	      }
	    }
	  }
	}
	cvecP M3 = complex_t(0);
	ivec3 iX;
	for( iX[2]=-1; iX[2]<=1; iX[2]++ ) {
	  for( iX[1]=-1; iX[1]<=1; iX[1]++ ) {
	    for( iX[0]=-1; iX[0]<=1; iX[0]++ ) {
	      vec3 dX;
	      for_3d dX[d] = iX[d] * diameter[d];
              M2M(dX,M,M3);
	    }
	  }
	}
	M = M3;
      }
#if EXAFMM_SERIAL
      Local[0] += L;
#else
      globLocal[0] += L;
#endif
    }

    vec3 getDipole() {
      vec3 dipole = 0;
      for (int i=0; i<numBodies; i++) {
        for_3d dipole[d] += (Jbodies[i][d] - RGlob[d]) * Jbodies[i][3];
      }
      return dipole;
    }

    void dipoleCorrection(vec3 dipole, int numBodiesGlob) {
      vec3 cycle = RGlob * 2;
      real_t coef = 4 * M_PI / (3 * cycle[0] * cycle[1] * cycle[2]);
      for (int i=0; i<numBodies; i++) {
        Ibodies[i][0] -= coef * norm(dipole) / numBodiesGlob / Jbodies[i][3];
        for (int d=0; d<3; d++) {
          Ibodies[i][d+1] -= coef * dipole[d];
        }
      }
    }
  };
}
