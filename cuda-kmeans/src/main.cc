#include <iostream>
#include <string>
#include <unistd.h>
#include "kmeans.h"
#include "utils.h"

using namespace std;

int main(int argc, char** argv) {  
  char* filename = nullptr;
  int c = -1;
  int x = -1;
  int t = -1; 
  int iters = 1;
  int o;
  while ((o = getopt(argc, argv, "c:f:x:t:s:i:")) != -1) {
    switch (o) {  
      case 'c':
        c = atoi(optarg);
        break;
      case 'f':
        filename = optarg;
        break;
      case 'x':
        x = atoi(optarg);
        break;
      case 't':
        t = atoi(optarg);
        break;
      case 'i':
        iters = atoi(optarg);
        break;
      case '?':
        if (optopt == 'c' || optopt == 'f' || optopt == 'x')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
        return 1;
    }
  }

  // validate input
  if(filename == nullptr || c == -1 || x == -1 || t == -1) {
    cout << "Invalid program arguments" << endl;
    return 1;
  }

  for(int i = 0; i < iters; ++i) {
    data_t* data = extract_data(filename);
    if(x > 0) {
    	cuda_kmeans(x, c, data, t);
    }
  }
}
