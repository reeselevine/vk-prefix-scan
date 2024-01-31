CXX = g++
CXXFLAGS = -std=c++17
CLSPVFLAGS = -cl-std=CL2.0 -inline-entry-points

.PHONY: clean

all: build easyvk blit prefix-scan amd-issue amd-mesa-issue intel-issue

build:
	mkdir -p build

clean:
	rm -r build

easyvk: easyvk/src/easyvk.cpp easyvk/src/easyvk.h
	$(CXX) $(CXXFLAGS) -Ieasyvk/src -c easyvk/src/easyvk.cpp -o build/easyvk.o

blit: blit.cinit blit.cpp
	$(CXX) $(CXXFLAGS) -Ieasyvk/src build/easyvk.o blit.cpp -lvulkan -o build/blit.run

amd-issue: amd-issue.cinit amd-issue.cpp
	$(CXX) $(CXXFLAGS) -Ieasyvk/src build/easyvk.o amd-issue.cpp -lvulkan -o build/amd-issue.run

amd-mesa-issue: amd-mesa-issue.cinit amd-mesa-issue.cpp
	$(CXX) $(CXXFLAGS) -Ieasyvk/src build/easyvk.o amd-mesa-issue.cpp -lvulkan -o build/amd-mesa-issue.run

intel-issue: intel-issue.cinit intel-issue.cpp
	$(CXX) $(CXXFLAGS) -Ieasyvk/src build/easyvk.o intel-issue.cpp -lvulkan -o build/intel-issue.run

prefix-scan: prefix-scan.cinit prefix-scan.cpp
	$(CXX) $(CXXFLAGS) -Ieasyvk/src build/easyvk.o prefix-scan.cpp -lvulkan -o build/prefix-scan.run

%.spv: %.cl
	clspv -w -cl-std=CL2.0 -inline-entry-points $< -o build/$@

%.cinit: %.cl
	clspv -w -cl-std=CL2.0 -inline-entry-points -output-format=c $< -o build/$@
