# CUDA KMEANS

This is example CUDA code that performs the Kmeans algorithm.
To run it give it the following parameters.

cuda-kmeans -f \<dataFilename> -c \<number of clusters> -t \<number of threads> -x \<algorithm> -i \<iterations>

- f for filename of the data to run kmeans on
- c for the number of clusters we want to create (the k in the algorithm)
- t for the number of threads we want to run on the GPU
- x for the algorithm (this is an integer which is explained below)

The algorithms are as follows:

| Algorithm Name                      |Integer |
|-------------------------------------|--------|
| Global Memory Centroid Threading    |    0   |
| Global Memory Dimension Threading   |    1   |
| Shared Memory Centroid Threading    |    2   |
| Shared Memory Dimension Threading   |    3   |

Centroid threading parallelizes over centroids. Dimension threading parallelizes over dimensions of centroids.

The memory type denotes where intermediate data is stored when operating.

## Data Set Format

The data files need to be formated as follows:

\<point number> \<dim 1 value> \<dim 2 value> ... \<dim n value>

For example:

```{text}
1 1.0 2.0
2 0.0 -1.0
```

## Headers and Sources

- kernels.h and kernels.cu contain the CUDA kernels utilized for various algorithms
- kmeans.h and kmeans.cu contains the actual kmeans algorithm which is run by calling cuda\_kmeans
- utils.h and utils.c contain utilities for setting up the data and reading in files

## Strange Conventions
 
To handle errors in GPU code we have to check if the code returned an error. We wrap this functionality
in the gpuAssert function and gpuErrchk macro in kernels.h. Whenever an error may be thrown by the CUDA
runtime we call gpuErrchk to make sure it is caught and the program shuts down.

## Acknowledgements

Jacob Nelson for the code this was modified from.
