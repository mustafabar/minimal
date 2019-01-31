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

  private:
    void checkPartition(int *maxPartition) {
      int partitionSize = 1;
      for_3d partitionSize *= maxPartition[d];
      assert( MPISIZE == partitionSize );
      int mpisize = MPISIZE;
      while (mpisize > 0) {
	assert( mpisize % 8 == 0 || mpisize == 1 );
	mpisize /= 8;
      }
      ivec3 checkLevel, partition;
      for_3d partition[d] = maxPartition[d];
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
      for_3d numPartition[0][d] = 1;
      for_3d partition[d] = maxPartition[d];
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
      for_3d bodiesType[d] = leafsType[d] * float(numBodies) / numLeafs * 4;
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
    
    inline void setGlobIndex(int i, int *iX) const {
#if NOWRAP
      i = (i / 3) * 3;
#endif
      for_3d iX[d] = int(Jbodies[i][d] / (2 * R0));
      for_3d iX[d] = iX[d] % numPartition[maxGlobLevel][d];
    }

    inline int getKey(int *iX, int level, bool levelOffset=true) const {
      int id = 0;
      if (levelOffset) id = ((1 << 3 * level) - 1) / 7;
      for(int lev=0; lev<level; ++lev ) {
        for_3d id += iX[d] % 2 << (3 * lev + d);
        for_3d iX[d] >>= 1;
      }
      return id;
    }
    
    inline int getGlobKey(int *iX, int level) const {
      return iX[0] + (iX[1] + iX[2] * numPartition[level][1]) * numPartition[level][0];
    }

    void getCenter(vec3 &dX, int index, int level) const {
      real_t R = R0 / (1 << level);
      ivec3 iX = 0;
      getIndex(iX, index);
      for_3d dX[d] = X0[d] - R0 + (2 * iX[d] + 1) * R;
    }
    
    void sort(vec4 *bodies, vec4 *buffer, int *index, int *ibuffer, int *key) const {
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
	for_4d buffer[inew][d] = bodies[i][d];
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
      Index = new int [2*numBodies];
      Rank = new int [2*numBodies];
      sendIndex = new int [2*numBodies];
      recvIndex = new int [2*numBodies];
      Leafs = new int [27*numLeafs][2]();
      sendLeafs = new int [numSendLeafs][2]();
      recvLeafs = new int [numSendLeafs][2]();
      Ibodies = new vec4 [2*numBodies]();
      Jbodies = new vec4 [2*numBodies+numSendBodies];
      Multipole = new complex_t [27*numCells][NTERM];
      Local = new complex_t [numCells][NTERM]();
      globMultipole = new complex_t [2*MPISIZE][NTERM]();
      globLocal = new complex_t [10][NTERM]();
      sendJbodies = new vec4 [2*numBodies+numSendBodies];
      recvJbodies = new vec4 [2*numBodies+numSendBodies];
      sendMultipole = new fcomplex_t [numSendCells][NTERM]();
      recvMultipole = new fcomplex_t [numSendCells][NTERM]();
    }

    void deallocate() {
      delete[] Index;
      delete[] Rank;
      delete[] sendIndex;
      delete[] recvIndex;
      delete[] Leafs;
      delete[] sendLeafs;
      delete[] recvLeafs;
      delete[] Ibodies;
      delete[] Jbodies;
      delete[] Multipole;
      delete[] Local;
      delete[] globMultipole;
      delete[] globLocal;
      delete[] sendJbodies;
      delete[] recvJbodies;
      delete[] sendMultipole;
      delete[] recvMultipole;
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
	for_3d IX[lev-1][d] = IX[lev][d] * numPartition[lev-1][d] / numPartition[lev][d];
      }
      setSendCounts();
      gatherLevel = level;
      if(gatherLevel > maxGlobLevel) gatherLevel = maxGlobLevel;
#if EXAFMM_SERIAL
#else
      ivec3 iX, numChild;
      for_3d numChild[d] = numPartition[maxGlobLevel][d] / numPartition[gatherLevel][d];
      for_3d iX[d] = IX[maxGlobLevel][d] % numChild[d];
      int key = iX[0] + (iX[1] + iX[2] * numChild[1]) * numChild[0];
      int color = getGlobKey(IX[gatherLevel],gatherLevel);
      MPI_Comm_split(MPI_COMM_WORLD, color, key, &MPI_COMM_LOCAL);
      MPI_Comm_split(MPI_COMM_WORLD, key, color, &MPI_COMM_GLOBAL);
#endif
    }

    void sortBodies() const {
      int *key = new int [numBodies];
      real_t diameter = 2 * R0 / (1 << maxLevel);
      ivec3 iX = 0;
      for( int i=0; i<numBodies; i++ ) {
	getIndex(i,iX,diameter);
	key[i] = getKey(iX,maxLevel);
      }
      sort(Jbodies,sendJbodies,Index,sendIndex,key);
      for( int i=0; i<numBodies; i++ ) {
	Index[i] = sendIndex[i];
	for_4d Jbodies[i][d] = sendJbodies[i][d];
      }
      delete[] key;
    }

    void buildTree() const {
      int rankOffset = 13 * numLeafs;
      for( int i=rankOffset; i<numLeafs+rankOffset; i++ ) {
	Leafs[i][0] = Leafs[i][1] = 0;
      }
      real_t diameter = 2 * R0 / (1 << maxLevel);
      ivec3 iX = 0;
      getIndex(0,iX,diameter);
      int ileaf = getKey(iX,maxLevel,false) + rankOffset;
      Leafs[ileaf][0] = 0;
      for( int i=0; i<numBodies; i++ ) {
	getIndex(i,iX,diameter);
	int inew = getKey(iX,maxLevel,false) + rankOffset;
	if( ileaf != inew ) {
	  Leafs[ileaf][1] = Leafs[inew][0] = i;
	  ileaf = inew;
	}
      }
      Leafs[ileaf][1] = numBodies;
      for( int i=rankOffset; i<numLeafs+rankOffset; i++ ) {
	//assert( Leafs[i][1] != Leafs[i][0] );
      }
    }

    void upwardPass() const {
      int rankOffset = 13 * numCells;
      for( int i=0; i<numCells; i++ ) {
	for_m Multipole[i+rankOffset][m] = 0;
	for_l Local[i][l] = 0;
      }

      logger::startTimer("P2M");
      rankOffset = 13 * numLeafs;
      int levelOffset = ((1 << 3 * maxLevel) - 1) / 7 + 13 * numCells;
#pragma omp parallel for
      for (int i=0; i<numLeafs; i++) {
	vec3 center;
	getCenter(center,i,maxLevel);
	for (int j=Leafs[i+rankOffset][0]; j<Leafs[i+rankOffset][1]; j++) {
	  vec3 dX;
          for_3d dX[d] = Jbodies[j][d] - center[d];
          P2M(dX,Jbodies[j][3],Multipole[i+levelOffset]);
	}
      }
      logger::stopTimer("P2M");

      logger::startTimer("M2M");
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
      logger::stopTimer("M2M");
    }

    void downwardPass() {
      logger::startTimer("M2L"); 
      ivec3 iXc;
      int DM2LC = DM2L;
      getGlobIndex(iXc,MPIRANK,maxGlobLevel);
      for (int lev=1; lev<=maxLevel; lev++) {
	if (lev==maxLevel) DM2LC = DP2P;
	int levelOffset = ((1 << 3 * lev) - 1) / 7;
	int nunit = 1 << lev;
	ivec3 nunitGlob;
	for_3d nunitGlob[d] = nunit * numPartition[maxGlobLevel][d];
	ivec3 nxmin, nxmax;
	for_3d nxmin[d] = -iXc[d] * (nunit >> 1);
	for_3d nxmax[d] = (nunitGlob[d] >> 1) + nxmin[d] - 1;
	if (numImages != 0) {
	  for_3d nxmin[d] -= (nunitGlob[d] >> 1);
	  for_3d nxmax[d] += (nunitGlob[d] >> 1);
	}
	real_t diameter = 2 * R0 / (1 << lev);
#pragma omp parallel for
	for (int i=0; i<(1 << 3 * lev); i++) {
	  complex_t L[NTERM];
	  for_l L[l] = 0;
	  ivec3 iX = 0;
	  getIndex(iX,i);
	  ivec3 jXmin;
	  for_3d jXmin[d] = (std::max(nxmin[d],(iX[d] >> 1) - DM2L) << 1);
	  ivec3 jXmax;
	  for_3d jXmax[d] = (std::min(nxmax[d],(iX[d] >> 1) + DM2L) << 1) + 1;
	  ivec3 jX;
	  for (jX[2]=jXmin[2]; jX[2]<=jXmax[2]; jX[2]++) {
	    for (jX[1]=jXmin[1]; jX[1]<=jXmax[1]; jX[1]++) {
	      for (jX[0]=jXmin[0]; jX[0]<=jXmax[0]; jX[0]++) {
		if(jX[0] < iX[0]-DM2LC || iX[0]+DM2LC < jX[0] ||
		   jX[1] < iX[1]-DM2LC || iX[1]+DM2LC < jX[1] ||
		   jX[2] < iX[2]-DM2LC || iX[2]+DM2LC < jX[2]) {
		  ivec3 jXp;
		  for_3d jXp[d] = (jX[d] + nunit) % nunit;
		  int j = getKey(jXp,lev);
		  for_3d jXp[d] = (jX[d] + nunit) / nunit;
#if EXAFMM_SERIAL
		  int rankOffset = 13 * numCells;
#else
		  int rankOffset = (jXp[0] + 3 * jXp[1] + 9 * jXp[2]) * numCells;
#endif
		  j += rankOffset;
		  vec3 dX;
		  for_3d dX[d] = (iX[d] - jX[d]) * diameter;
                  M2L(dX,Multipole[j],L);
		}
	      }
	    }
	  }
	  for_l Local[i+levelOffset][l] += L[l];
	}
      }
      logger::stopTimer("M2L");

      logger::startTimer("L2L");
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
      logger::stopTimer("L2L");

      logger::startTimer("L2P");
      int rankOffset = 13 * numLeafs;
      int levelOffset = ((1 << 3 * maxLevel) - 1) / 7;
#pragma omp parallel for
      for (int i=0; i<numLeafs; i++) {
	vec3 center;
	getCenter(center,i,maxLevel);
	complex_t L[NTERM];
	for_l L[l] = Local[i+levelOffset][l];
	for (int j=Leafs[i+rankOffset][0]; j<Leafs[i+rankOffset][1]; j++) {
	  vec3 dX;
	  for_3d dX[d] = Jbodies[j][d] - center[d];
          L2P(dX,L,Ibodies[j]);
	}
      }
      logger::stopTimer("L2P");

      logger::startTimer("P2P");
      getGlobIndex(iXc,MPIRANK,maxGlobLevel);
      int nunit = 1 << maxLevel;
      ivec3 nunitGlob;
      for_3d nunitGlob[d] = nunit * numPartition[maxGlobLevel][d];
      ivec3 nxmin, nxmax;
      for_3d nxmin[d] = -iXc[d] * nunit;
      for_3d nxmax[d] = nunitGlob[d] + nxmin[d] - 1;
      if (numImages != 0) {
	for_3d nxmin[d] -= nunitGlob[d];
	for_3d nxmax[d] += nunitGlob[d];
      }
#pragma omp parallel for
      for (int i=0; i<numLeafs; i++) {
	ivec3 iX = 0;
	getIndex(iX,i);
	ivec3 jXmin, jXmax;
	for_3d jXmin[d] = std::max(nxmin[d],iX[d] - DP2P);
	for_3d jXmax[d] = std::min(nxmax[d],iX[d] + DP2P);
	ivec3 jX;
	for (jX[2]=jXmin[2]; jX[2]<=jXmax[2]; jX[2]++) {
	  for (jX[1]=jXmin[1]; jX[1]<=jXmax[1]; jX[1]++) {
	    for (jX[0]=jXmin[0]; jX[0]<=jXmax[0]; jX[0]++) {
	      ivec3 jXp;
	      for_3d jXp[d] = (jX[d] + nunit) % nunit;
	      int j = getKey(jXp,maxLevel,false);
	      for_3d jXp[d] = (jX[d] + nunit) / nunit;
#if EXAFMM_SERIAL
	      int rankOffset = 13 * numLeafs;
#else
	      int rankOffset = (jXp[0] + 3 * jXp[1] + 9 * jXp[2]) * numLeafs;
#endif
	      j += rankOffset;
	      rankOffset = 13 * numLeafs;
	      vec3 periodic = 0;
	      for_3d jXp[d] = (jX[d] + iXc[d] * nunit + nunitGlob[d]) / nunitGlob[d];
	      for_3d periodic[d] = (jXp[d] - 1) * 2 * RGlob[d];
	      P2P(Leafs[i+rankOffset][0],Leafs[i+rankOffset][1],Leafs[j][0],Leafs[j][1],periodic);
	    }
	  }
	}
      }
      logger::stopTimer("P2P");
    }

    void periodicM2L() {
      complex_t M[NTERM];
#if EXAFMM_SERIAL
      for_m M[m] = Multipole[13*numCells][m];
#else
      for_m M[m] = globMultipole[0][m];
#endif
      complex_t L[NTERM];
      for_l L[l] = 0;
      for( int lev=1; lev<numImages; lev++ ) {
	vec3 diameter;
	for_3d diameter[d] = 2 * RGlob[d] * std::pow(3.,lev-1);
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
	complex_t M3[NTERM];
	for_m M3[m] = 0;
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
	for_m M[m] = M3[m];
      }
#if EXAFMM_SERIAL
      for_l Local[0][l] += L[l];
#else
      for_l globLocal[0][l] += L[l];
#endif
    }
  };
}
