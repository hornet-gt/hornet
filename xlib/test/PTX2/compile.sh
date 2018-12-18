nvcc -arch=sm_35 -DARCH=350 -DSM=12 --resource-usage --std=c++11 -I../../include -keep -keep-dir TMP  PTXtest.cu
