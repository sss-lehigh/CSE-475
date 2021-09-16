#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>
#include "utils.h"

#define MAX_LINE_LENGTH 1000000
#define IS_BINARY 0

data_t::data_t() : numPoints(0), numAttrs(0), data(nullptr) {
}

data_t::data_t(const int points, const int attrs) :
                numPoints(points), numAttrs(attrs) {
        data = new float*[numPoints];
        for (int i = 0; i < numPoints; ++i) {
                data[i] = new float[numAttrs];
        }
}

data_t::~data_t() {
        for (int i = 0; i < numPoints; ++i) {
                delete[] data[i];
        }
        delete[] data;
}

void data_t::setData(float** d, int p, int a) {
        numPoints = p;
        numAttrs = a;

        // reinit data array
        if (data != nullptr) {
                for (int i = 0; i < numPoints; ++i) {
                        delete[] data[i];
                }
                delete[] data;
        }

        data = new float*[numPoints];
        for (int i = 0; i < numPoints; ++i) {
                data[i] = new float[numAttrs];
        }

        for (int i = 0; i < p; ++i) {
                for (int j = 0; j < a; ++j) {
                        data[i][j] = d[i][j];
                }
        }
}

data_t* extract_data(std::string filename) {
        // get the args, and input file or whatever
        // setup variables
        auto line = new char[MAX_LINE_LENGTH];
        float** data;
        int numObjects = 0, numAttributes = 0;
        char* attr;
        char* end;


	if (IS_BINARY) {
		auto infile = std::ifstream(filename, std::ios::binary | std::ios::binary);
        	if (!infile.is_open()) {
        	        fprintf(stderr, "Error: no such file (%s)\n", filename.c_str());
        	        exit(1);
        	}
		//infile >> numObjects;
		//infile >> numAttributes;
		infile.read(reinterpret_cast<char*>(&numObjects), sizeof(numObjects));
		infile.read(reinterpret_cast<char*>(&numAttributes), sizeof(numAttributes));

                // initialize data array
       	        data = new float*[numObjects];
       	        for (int i = 0; i < numObjects; ++i) {
       	                data[i] = new float[numAttributes];
       	        }
		for (int i = 0; i < numObjects; ++i) {
			for (int j = 0; j < numAttributes; ++j) {
				infile.read(reinterpret_cast<char*>(&data[i][j]), sizeof(float));
			}
		}
	}	
	else {
        	auto infile = std::ifstream(filename, std::ios::in);
        	if (!infile.is_open()) {
        	        fprintf(stderr, "Error: no such file (%s)\n", filename.c_str());
        	        exit(1);
        	}
        	while (infile.getline(line, MAX_LINE_LENGTH)) {
        	        if (std::strtok(line, " \t\n") != 0) {
        	                numObjects++;
        	        }
        	}

        	// clear error state from reading EOF (failbit and eofbit are currently set)
        	// seek will not work if eofbit or failbit are set
        	infile.clear();
        	infile.seekg(0); // then seek to the beginning

        	while (infile.getline(line, MAX_LINE_LENGTH)) {
        	        if (strtok(line, " \t\n") != nullptr) {
        	                /* Ignore the id (first attribute): numAttributes = 1; */
        	                while (strtok(nullptr, " ,\t\n") != nullptr) {
        	                        numAttributes++;
        	                }
        	                break;
        	        }
        	}

        	infile.clear();
        	infile.seekg(0);

        	int i = 0;


        	// initialize data array
       		data = new float*[numObjects];
       		for (int i = 0; i < numObjects; ++i) {
       		        data[i] = new float[numAttributes];
       		}
        	while (infile.getline(line, MAX_LINE_LENGTH)) {
        	        if (strtok(line, " \t\n") != nullptr) {
        	                int j = 0;
        	                while ((attr = strtok(nullptr, " ,\t\n")) != nullptr) {
        	                        data[i][j] = std::strtof(attr, &end);
        	                        ++j;
        	                }
        	        }
        	        ++i;
        	}
	}

        data_t* d = new data_t();
        d->setData(data, numObjects, numAttributes);
        for (int i = 0; i < numObjects; ++i) {
                delete[] data[i];
        }
        delete[] data;
        delete[] line;
        return d;
}

void printClusters(float* clusters, int nclusters, int nfeatures) {
        std::cout << "[" << std::endl;
        for (int i = 0; i < nclusters; ++i) {
                std::cout << "cluster " << i << ": [";
                for (int j = 0; j < nfeatures; ++j) {
                        std::cout << clusters[i * nfeatures + j];
                        if (j < nfeatures - 1) {
                                std::cout << ", ";
                        }
                }
                std::cout << "]\n";
        }
        std::cout << "]\n";
}

void setRandomIndices(int* ptr, int len) {
  int seed = std::rand() * 100;
  std::vector<int> rand_idx(len);
  for (int i = 0; i < len; ++i) {
	  rand_idx[i] = i;
  }
  std::shuffle(rand_idx.begin(), rand_idx.end(), std::default_random_engine(seed));
  for (int i = 0; i < len; ++i) {
  	ptr[i] = rand_idx[i];
  }
}
