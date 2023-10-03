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

blit: blit.cinit blit.cpp
	$(CXX) $(CXXFLAGS) -Ieasyvk/src build/easyvk.o blit.cpp -lvulkan -o build/blit.run

%.spv: %.cl
	clspv -w -cl-std=CL2.0 -inline-entry-points $< -o $@

%.cinit: %.cl
	clspv -w -cl-std=CL2.0 -inline-entry-points -output-format=c $< -o build/$@
