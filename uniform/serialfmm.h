#include <mpi.h>
#include "kernels.h"

namespace exafmm {
  class SerialFMM : public Kernel {
  protected:
    const int DP2P = 2;
    const int DREG = 1;

  public:
    static vec3 Xperiodic;
    int maxLevel;
    int numBodies;
    int numImages;
    int numCells;
    int numLeafs;
    vec3 X0;
    real_t R0;
    std::vector<int> Index;
    std::vector<Range> Leafs;
    std::vector<vec4> Ibodies;
    std::vector<vec4> Jbodies;
    std::vector<cvecP> Multipole;
    std::vector<cvecP> Local;
    
  protected:
    inline void getIndex(int i, ivec3 &iX, real_t diameter) const {
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

    inline int getKey(ivec3 &iX, int level, bool levelOffset=true) const {
      int id = 0;
      if (levelOffset) id = ((1 << 3 * level) - 1) / 7;
      for(int lev=0; lev<level; ++lev ) {
        for_3d id += iX[d] % 2 << (3 * lev + d);
        iX >>= 1;
      }
      return id;
    }

    void getCenter(vec3 &dX, ivec3 iX, int level) const {
      real_t R = R0 / (1 << level);
      for_3d dX[d] = X0[d] - R0 + (2 * iX[d] + 1) * R;
    }
    
    void getCenter(vec3 &dX, int index, int level) const {
      ivec3 iX = 0;
      getIndex(iX, index);
      getCenter(dX, iX, level);
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
	for_4d buffer[inew][d] = bodies[i][d];
      }
    }

  public:
    SerialFMM(int N, int L, int Im) {
      maxLevel = L;
      numBodies = N;
      numImages = Im;
      numCells = ((1 << 3 * (L + 1)) - 1) / 7;
      numLeafs = 1 << 3 * L;
      Index.resize(numBodies);
      Leafs.resize(numLeafs);
      Ibodies.resize(numBodies);
      Jbodies.resize(numBodies);
      Multipole.resize(numCells);
      Local.resize(numCells);
    }

    void sortBodies() {
      std::vector<int> key(numBodies);
      real_t diameter = 2 * R0 / (1 << maxLevel);
      ivec3 iX = 0;
      for( int i=0; i<numBodies; i++ ) {
	getIndex(i,iX,diameter);
	key[i] = getKey(iX,maxLevel);
      }
      std::vector<int> Index2(numBodies);
      std::vector<vec4> Jbodies2(numBodies);
      sort(Jbodies,Jbodies2,Index,Index2,key);
      for( int i=0; i<numBodies; i++ ) {
	Index[i] = Index2[i];
	for_4d Jbodies[i][d] = Jbodies2[i][d];
      }
    }

    void buildTree() {
      for( int i=0; i<numLeafs; i++ ) {
	Leafs[i].begin = Leafs[i].end = 0;
      }
      real_t diameter = 2 * R0 / (1 << maxLevel);
      ivec3 iX = 0;
      getIndex(0,iX,diameter);
      int ileaf = getKey(iX,maxLevel,false);
      Leafs[ileaf].begin = 0;
      for( int i=0; i<numBodies; i++ ) {
	getIndex(i,iX,diameter);
	int inew = getKey(iX,maxLevel,false);
	if( ileaf != inew ) {
	  Leafs[ileaf].end = Leafs[inew].begin = i;
	  ileaf = inew;
	}
      }
      Leafs[ileaf].end = numBodies;
    }

    void upwardPass() {
      for( int i=0; i<numCells; i++ ) {
	Multipole[i] = 0;
	Local[i] = 0;
      }

      start("P2M");
      ivec3 nunit = 1 << maxLevel;
      int levelOffset = ((1 << 3 * maxLevel) - 1) / 7 + 0 * numCells;
      real_t R = R0 / (1 << maxLevel);
      //#pragma omp parallel for
      for (int j=0; j<numLeafs; j++) {
	ivec3 jX = 0;
	getIndex(jX,j);
	ivec3 iXmin = jX - DREG;
	ivec3 iXmax = jX + DREG;
	ivec3 iX;
	for (iX[2]=iXmin[2]; iX[2]<=iXmax[2]; iX[2]++) {
	  for (iX[1]=iXmin[1]; iX[1]<=iXmax[1]; iX[1]++) {
	    for (iX[0]=iXmin[0]; iX[0]<=iXmax[0]; iX[0]++) {
	      ivec3 iXwrap = (iX + nunit) % nunit;
	      int i = getKey(iXwrap,maxLevel,false);
              vec3 X;
              getCenter(X,iX,maxLevel);
              for (int b=Leafs[j].begin; b<Leafs[j].end; b++) {
                vec3 dX;
                for_3d dX[d] = Jbodies[b][d] - X[d];
                P2M(dX,R,Jbodies[b][3],Multipole[i+levelOffset]);
              }
            }
          }
	}
      }
      stop("P2M");

      start("M2M");
      for (int lev=maxLevel; lev>0; lev--) {
	int childOffset = ((1 << 3 * lev) - 1) / 7;
	int parentOffset = ((1 << 3 * (lev - 1)) - 1) / 7;
	real_t radius = R0 / (1 << lev);
        //#pragma omp parallel for schedule(static, 8)
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
#if 0
      {
      int lev = maxLevel;
      int levelOffset = ((1 << 3 * lev) - 1) / 7;
      ivec3 nunit = 1 << lev;
      ivec3 nxmin = 0;
      ivec3 nxmax = (nunit >> 1) + nxmin - 1;
      if (numImages != 0) {
        nxmin -= nunit;
        nxmax += nunit;
      }
      real_t diameter = 2 * R0 / (1 << lev);
      for (int i=0; i<(1 << 3 * lev); i++) {
        cvecP L = complex_t(0);
        ivec3 iX = 0;
        getIndex(iX,i);
        for (int j=0; j<(1 << 3 * lev); j++) {
          ivec3 jX = 0;
          getIndex(jX,j);
          if(jX[0] < iX[0]-DP2P || iX[0]+DP2P < jX[0] ||
             jX[1] < iX[1]-DP2P || iX[1]+DP2P < jX[1] ||
             jX[2] < iX[2]-DP2P || iX[2]+DP2P < jX[2]) {
            ivec3 jXwrap = (jX + nunit) % nunit;
            int j = getKey(jXwrap,lev);
            vec3 dX;
            for_3d dX[d]= (iX[d] - jX[d]) * diameter;
            M2L(dX,Multipole[j],L);
          }
        }
        Local[i+levelOffset] += L;
      }
      }
#else
      int DM2LC = 1;
      for (int lev=1; lev<=maxLevel; lev++) {
	if (lev==maxLevel) DM2LC = DP2P;
	int levelOffset = ((1 << 3 * lev) - 1) / 7;
	ivec3 nunit = 1 << lev;
	ivec3 nxmin = 0;
	ivec3 nxmax = (nunit >> 1) + nxmin - 1;
        if (numImages != 0) {
          nxmin -= nunit;
          nxmax += nunit;
        }
	real_t diameter = 2 * R0 / (1 << lev);
#pragma omp parallel for
	for (int i=0; i<(1 << 3 * lev); i++) {
	  cvecP L = complex_t(0);
	  ivec3 iX = 0;
	  getIndex(iX,i);
	  ivec3 jXmin = (max(nxmin,(iX >> 1) - 1) << 1);
	  ivec3 jXmax = (min(nxmax,(iX >> 1) + 1) << 1) + 1;
	  ivec3 jX;
	  for (jX[2]=jXmin[2]; jX[2]<=jXmax[2]; jX[2]++) {
	    for (jX[1]=jXmin[1]; jX[1]<=jXmax[1]; jX[1]++) {
	      for (jX[0]=jXmin[0]; jX[0]<=jXmax[0]; jX[0]++) {
		if(jX[0] < iX[0]-DM2LC || iX[0]+DM2LC < jX[0] ||
		   jX[1] < iX[1]-DM2LC || iX[1]+DM2LC < jX[1] ||
		   jX[2] < iX[2]-DM2LC || iX[2]+DM2LC < jX[2]) {
		  ivec3 jXwrap = (jX + nunit) % nunit;
		  int j = getKey(jXwrap,lev);
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
#endif
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
          if (i==165&&lev==3) {
            std::cout << std::fixed << std::setprecision(9) << Local[c][1] << std::endl;
          }
          L2L(dX,Local[p],Local[c]);
	}
      }
      stop("L2L");

      start("L2P");
      ivec3 nunit = 1 << maxLevel;
      int levelOffset = ((1 << 3 * maxLevel) - 1) / 7;
      real_t R = R0 / (1 << maxLevel);
#pragma omp parallel for
      for (int i=0; i<numLeafs; i++) {
	ivec3 iX = 0;
	getIndex(iX,i);
	ivec3 jXmin = iX - DREG;
	ivec3 jXmax = iX + DREG;
	ivec3 jX;
	for (jX[2]=jXmin[2]; jX[2]<=jXmax[2]; jX[2]++) {
	  for (jX[1]=jXmin[1]; jX[1]<=jXmax[1]; jX[1]++) {
	    for (jX[0]=jXmin[0]; jX[0]<=jXmax[0]; jX[0]++) {
	      ivec3 jXwrap = (jX + nunit) % nunit;
	      int j = getKey(jXwrap,maxLevel,false);
              vec3 X;
              getCenter(X,jX,maxLevel);
              cvecP L = Local[j+levelOffset];
              for (int b=Leafs[i].begin; b<Leafs[i].end; b++) {
                vec3 dX;
                for_3d dX[d] = Jbodies[b][d] - X[d];
                L2P(dX,R,L,Ibodies[b]);
              }
              if (i==385&&jX[0]==1&&jX[1]==4&&jX[2]==3) std::cout << std::fixed << std::setprecision(9) << Ibodies[4] << std::endl;
            }
          }
	}
      }
      for (int i=0; i<8; i++) {
        //std::cout << i+8 << " " << std::fixed << std::setprecision(6) << Ibodies[i] << std::endl;
      }
      stop("L2P");

      start("P2P");
      ivec3 nxmin = 0;
      ivec3 nxmax = nunit + nxmin - 1;
      if (numImages != 0) {
        nxmin -= nunit * 2;
        nxmax += nunit * 2;
      }
#pragma omp parallel for
      for (int i=0; i<numLeafs; i++) {
        ivec3 iX = 0;
        getIndex(iX,i); 
	ivec3 irXmin = iX - DREG;
	ivec3 irXmax = iX + DREG;
	ivec3 irX;
	for (irX[2]=irXmin[2]; irX[2]<=irXmax[2]; irX[2]++) {
	  for (irX[1]=irXmin[1]; irX[1]<=irXmax[1]; irX[1]++) {
	    for (irX[0]=irXmin[0]; irX[0]<=irXmax[0]; irX[0]++) {
              vec3 Xi;
              getCenter(Xi,irX,maxLevel);
              ivec3 jXmin = max(nxmin,iX - DP2P);
              ivec3 jXmax = min(nxmax,iX + DP2P);
              ivec3 jX;
              for (jX[2]=jXmin[2]; jX[2]<=jXmax[2]; jX[2]++) {
                for (jX[1]=jXmin[1]; jX[1]<=jXmax[1]; jX[1]++) {
                  for (jX[0]=jXmin[0]; jX[0]<=jXmax[0]; jX[0]++) {
                    ivec3 jXwrap = (jX + nunit) % nunit;
                    int j = getKey(jXwrap,maxLevel,false);
                    ivec3 pX = (jX + nunit) / nunit;
                    vec3 periodic;
                    for_3d periodic[d] = (pX[d] - 1) * 2 * R0;
                    ivec3 jrXmin = jX - DREG;
                    ivec3 jrXmax = jX + DREG;
                    ivec3 jrX;
                    for (jrX[2]=jrXmin[2]; jrX[2]<=jrXmax[2]; jrX[2]++) {
                      for (jrX[1]=jrXmin[1]; jrX[1]<=jrXmax[1]; jrX[1]++) {
                        for (jrX[0]=jrXmin[0]; jrX[0]<=jrXmax[0]; jrX[0]++) {
                          if(irX[0]-DP2P <= jrX[0] && jrX[0] <= irX[0]+DP2P &&
                             irX[1]-DP2P <= jrX[1] && jrX[1] <= irX[1]+DP2P &&
                             irX[2]-DP2P <= jrX[2] && jrX[2] <= irX[2]+DP2P) {
                            vec3 Xj;
                            getCenter(Xj,jrX,maxLevel);
                            P2P(Ibodies,Leafs[i].begin,Leafs[i].end,Xi,
                                Jbodies,Leafs[j].begin,Leafs[j].end,Xj,R,periodic);
                          }
                        }
                      }
                    }
                    if (i==385 && j==448) {
                      //std::cout << irX << std::endl;
                      for (int i=0; i<8; i++) {
                        //std::cout << i+8 << " " << std::fixed << std::setprecision(6) << Ibodies[i] << std::endl;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      stop("P2P");
    }

    void periodicM2L() {
      cvecP M;
      M = Multipole[0*numCells];
      cvecP L = complex_t(0);
      for( int lev=1; lev<numImages; lev++ ) {
	vec3 diameter = R0 * 2 * std::pow(3.,lev-1);
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
      Local[0] += L;
    }

    void ewaldRealPart(real_t alpha, real_t cutoff) {
      ivec3 nunit = 1 << maxLevel;
      ivec3 nxmin = 0;
      ivec3 nxmax = nunit + nxmin - 1;
      if (numImages != 0) {
	nxmin -= nunit;
	nxmax += nunit;
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
	      ivec3 jXwrap = (jX + nunit) % nunit;
	      int j = getKey(jXwrap,maxLevel,false);
	      ivec3 pX = (jX + nunit) / nunit;
	      vec3 periodic;
	      for_3d periodic[d] = (pX[d] - 1) * 2 * R0;
	      EwaldP2P(Ibodies,Leafs[i].begin,Leafs[i].end,
                       Jbodies,Leafs[j].begin,Leafs[j].end,periodic,
                       alpha,cutoff);
	    }
	  }
	}
      }
    }

    void vanDerWaals(real_t cuton, real_t cutoff, int numTypes,
                     real_t * rscale, real_t * gscale, real_t * fgscale) {
      ivec3 nunit = 1 << maxLevel;
      ivec3 nxmin = 0;
      ivec3 nxmax = nunit + nxmin - 1;
      if (numImages != 0) {
	nxmin -= nunit;
	nxmax += nunit;
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
	      ivec3 jXwrap = (jX + nunit) % nunit;
	      int j = getKey(jXwrap,maxLevel,false);
	      ivec3 pX = (jX + nunit) / nunit;
	      vec3 periodic;
	      for_3d periodic[d] = (pX[d] - 1) * 2 * R0;
	      VdWP2P(Ibodies,Leafs[i].begin,Leafs[i].end,
                     Jbodies,Leafs[j].begin,Leafs[j].end,periodic,
                     cuton,cutoff,numTypes,rscale,gscale,fgscale);
	    }
	  }
	}
      }
    }

    vec3 getDipole() {
      vec3 dipole = 0;
      for (int i=0; i<numBodies; i++) {
        for_3d dipole[d] += (Jbodies[i][d] - R0) * Jbodies[i][3];
      }
      return dipole;
    }

    void dipoleCorrection(vec3 dipole) {
      vec3 cycle = R0 * 2;
      real_t coef = 4 * M_PI / (3 * cycle[0] * cycle[1] * cycle[2]);
      for (int i=0; i<numBodies; i++) {
        Ibodies[i][0] -= coef * norm(dipole) / numBodies / Jbodies[i][3];
        for (int d=0; d<3; d++) {
          Ibodies[i][d+1] -= coef * dipole[d];
        }
      }
    }
  };
}
