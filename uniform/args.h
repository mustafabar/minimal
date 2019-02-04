#ifndef args_h
#define args_h
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <iomanip>
#include <stdint.h>
#include "types.h"

namespace exafmm {
  static struct option long_options[] = {
    {"ncrit",        required_argument, 0, 'c'},
    {"help",         no_argument,       0, 'h'},
    {"images",       required_argument, 0, 'i'},
    {"numBodies",    required_argument, 0, 'n'},
    {"verbose",      no_argument,       0, 'v'},
    {0, 0, 0, 0}
  };

  class Args {
  public:
    int ncrit;
    int images;
    int numBodies;
    int verbose;

  private:
    void usage(char * name) {
      fprintf(stderr,
	      "Usage: %s [options]\n"
	      "Long option (short option)       : Description (Default value)\n"
	      " --ncrit (-c)                    : Number of bodies per leaf cell (%d)\n"
	      " --help (-h)                     : Show this help document\n"
	      " --images (-i)                   : Number of periodic image levels (%d)\n"
	      " --numBodies (-n)                : Number of bodies (%d)\n"
	      " --verbose (-v)                  : Print information to screen (%d)\n",
	      name,
	      ncrit,
	      images,
	      numBodies,
	      verbose);
    }

  public:
    Args(int argc=0, char ** argv=NULL) :
      ncrit(32),
      images(6),
      numBodies(1000),
      verbose(0) {
      while (1) {
	int option_index;
	int c = getopt_long(argc, argv, "c:hi:n:v", long_options, &option_index);
	if (c == -1) break;
	switch (c) {
	case 'c':
	  ncrit = atoi(optarg);
	  break;
	case 'h':
	  usage(argv[0]);
	  abort();
	case 'i':
	  images = atoi(optarg);
	  break;
	case 'n':
	  numBodies = atoi(optarg);
	  break;
	case 'v':
	  verbose = 1;
	  break;
	default:
	  usage(argv[0]);
	  abort();
	}
      }
    }

    void print(int stringLength) {
      if (verbose) {
	std::cout << std::setw(stringLength) << std::fixed << std::left
		  << "ncrit" << " : " << ncrit << std::endl
		  << std::setw(stringLength)
		  << "images" << " : " << images << std::endl
		  << std::setw(stringLength)
		  << "numBodies" << " : " << numBodies << std::endl
		  << std::setw(stringLength)
		  << "verbose" << " : " << verbose << std::endl;
      }
    }
  };
}
#endif
