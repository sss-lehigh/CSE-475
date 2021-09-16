#ifndef UTILS_H
#define UTILS_H

#include <string>

struct data_t {
  int numPoints;
  int numAttrs;
  float** data;

  data_t();
  data_t(const int, const int);
  ~data_t();
  
  void setData(float**, int, int);
};

data_t* extract_data(std::string);
void printClusters(float*, int, int);
void setRandomIndices(int*, int);

#endif
