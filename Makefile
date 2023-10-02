CXX = g++
CXXFLAGS = -std=c++17
CLSPVFLAGS = -cl-std=CL2.0 -inline-entry-points

.PHONY: clean

all: build easyvk blit

build:
	mkdir -p build

clean:
	rm -r build

easyvk: easyvk/src/easyvk.cpp easyvk/src/easyvk.h
	$(CXX) $(CXXFLAGS) -Ieasyvk/src -c easyvk/src/easyvk.cpp -o build/easyvk.o

blit: vect-add.spv vect-add.cinit vect-add.cpp
	$(CXX) $(CXXFLAGS) -Ieasyvk/src build/easyvk.o vect-add.cpp -lvulkan -o build/vect-add.run

%.spv: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points $< -o build/$@

%.cinit: %.cl
	clspv -cl-std=CL2.0 -inline-entry-points -output-format=c $< -o build/$@
