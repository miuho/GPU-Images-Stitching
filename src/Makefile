PROJ = interest_point
CC = g++ -O3
NVCC = nvcc -O3 --ptxas-options=-v

CFLAGS = -c -g -Wall -I/opt/local/include -I$(HOME)/cppunit/include -I$(CUDA_HOME)/include
LDFLAGS = -L/opt/local/lib -L$(HOME)/cppunit/lib -L$(CUDA_HOME)/lib
LIBS = -lcppunit -ldl -lcufft
OBJS = interest_point.o tests.o

all: $(PROJ)

$(PROJ): $(OBJS) $(TEST_CASES)
	$(NVCC) $(LDFLAGS) $^ -o $@ $(LIBS)

%.o : %.cu %.h
	$(NVCC) -c $< -o $@

%.o : %.cpp
	$(CC) $(CFLAGS) $< -o $@ 

clean:
	rm -f $(PROJ) $(OBJS) $(TEST_CASES) *.xml
