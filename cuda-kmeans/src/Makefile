TARGET = cuda_kmeans

BUILD_TYPE=

CUDA = /usr/local/cuda/bin/nvcc
CXX = g++

CXXSRC = utils.cc main.cc 
CUDASRC = kmeans.cu kernels.cu

IDIR = ../include
IFLAGS = -I$(IDIR)

CXXFLAGS = 
ARCH = sm_75
CUFLAGS = -arch=$(ARCH) -Xptxas -v
SHAREDFLAGS = -std=c++11 -v

ODIR = obj
CXXOBJ = $(patsubst %.cc,$(ODIR)/%.cc.o,$(CXXSRC))
CUDAOBJ = $(patsubst %.cu,$(ODIR)/%.cu.o,$(CUDASRC))
OBJ = $(CXXOBJ) $(CUDAOBJ)

all: CXXFLAGS += -O3
all: CUFLAGS += -O3 -Xptxas -dlcm=ca
all: BUILD_TYPE = Release
all: $(TARGET)
	@echo "Done"

debug: CXXFLAGS += -O0 -Wall 
debug: CUFLAGS += -G
debug: SHAREDFLAGS += -g
debug: BUILD_TYPE = Debug
debug: $(TARGET)
	@echo "Done"

$(ODIR)/%.cc.o: %.cc $(IFILES)
	@echo $< " --> " $@ "["$(BUILD_TYPE)"]"
	@$(CXX) -c $(CXXFLAGS) $(SHAREDFLAGS) $(IFLAGS) $(LFLAGS) $(LIBS) -o $@ $< 

$(ODIR)/%.cu.o: %.cu $(IFILES)
	@echo $< " --> " $@ "["$(BUILD_TYPE)"]"
	@$(CUDA) -c -o $@ $< $(CUFLAGS) $(SHAREDFLAGS) $(IFLAGS)

$(TARGET): $(OBJ)
	@echo "Linking object files..." $(LFLAGS) 
	@$(CUDA) -L$(LDIR) $(LIBS) $(OBJ) -o $@ $(CUFLAGS) 

.PHONY: clean

clean:
	rm -rf $(ODIR)/*.o *~ $(IDIR)/*~
	rm $(TARGET)
