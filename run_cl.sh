rm -f *.o alexnet_cl alexnet_cpu
g++  -I.//inc -I.   -c alexnet_host.cpp -o alexnet_host.o 
g++  -I.//inc -I. -o alexnet_cl alexnet_host.o -L.//lib -lOpenCL
