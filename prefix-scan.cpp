#include <vector>
#include <iostream>
#include <easyvk.h>
#include <cassert>
#include <vector>
#include <unistd.h>
#include <numeric>

#define BATCH_SIZE 8

void computeReferencePrefixSum(uint32_t* ref, int size, int overflow) {
  ref[0] = 1;
  if (overflow){
	for (int i = 1; i < size; i++) {
    	ref[i] = ref[i - 1] + i;
  	}
  }else{
	for (int i = 1; i < size; i++) {
    	ref[i] = i + 1;
  	}
  }
}

int main(int argc, char* argv[]) {
  int workgroupSize = 128;
  int numWorkgroups = 1;
  int deviceID = 1;
  bool enableValidationLayers = false;
  bool checkResults = false;
  int c;
  char alg = 'a';

    while ((c = getopt (argc, argv, "vct:w:d:b:")) != -1)
    switch (c)
      {
      case 't':
        workgroupSize = atoi(optarg);
        break;
      case 'w':
        numWorkgroups = atoi(optarg);
        break;
      case 'v':
	enableValidationLayers = true;
	break;
		case 'b':
		alg = optarg[0];
		break;
      case 'c':
	checkResults = true;
	break;
      case 'd':
	deviceID = atoi(optarg);
	break;
      case '?':
        if (optopt == 't' || optopt == 'w')
          std::cerr << "Option -" << optopt << "requires an argument\n";
        else 
          std::cerr << "Unknown option" << optopt << "\n";
        return 1;
      default:
        abort ();
      }
    auto size = numWorkgroups * workgroupSize * BATCH_SIZE;
	auto sizeBytes = numWorkgroups * workgroupSize * BATCH_SIZE * sizeof(uint);
	// Initialize instance.
	auto instance = easyvk::Instance(enableValidationLayers);
	// Get list of available physical devices.
	auto physicalDevices = instance.physicalDevices();
	// Create device from first physical device.
	auto device = easyvk::Device(instance, physicalDevices.at(deviceID));
	std::cout << "Using device: " << device.properties.deviceName << "\n";
    std::cout << "Device subgroup size: " << device.subgroupSize() << "\n";
	
	
	// Define the host local buffers. 
	std::vector<uint> hostOut(size, 0);
	std::vector<uint> hostDebug(2, 0);
	std::vector<uint> ref(size, 0);

	// Define in buffer, fill with 1s.
	auto in = easyvk::Buffer(device, sizeBytes, true);
	in.fill(1);

	// Define device local buffers
	auto out = easyvk::Buffer(device, sizeBytes, true);
	auto debug = easyvk::Buffer(device, sizeof(uint), true);

	// Use debug buffer to also change reduction algorithm.
	debug.fill(alg); 

	// Initilize program
	std::vector<easyvk::Buffer> bufs = {in, out, debug};
	std::vector<uint32_t> spvCode = 
	#include "build/prefix-scan.cinit"
	;
	auto program = easyvk::Program(device, spvCode, bufs);

	program.setWorkgroups(numWorkgroups);
	program.setWorkgroupSize(workgroupSize);
	program.setWorkgroupMemoryLength(workgroupSize*sizeof(uint), 0);

	// Run the kernel.
	program.initialize("prefix_scan");

	float time = program.runWithDispatchTiming();
	
	// Retrieve data from GPU buffers
	out.load(hostOut.data(), sizeBytes);
	debug.load(hostDebug.data(), sizeof(uint));


	if (checkResults) {
		computeReferencePrefixSum(ref.data(), size, false);
		for (int i = 0; i < size; i++) {
			std::cout << "out[" << i << "]: " << hostOut[i] << ", ref:" << ref[i] << "\n";
			assert(hostOut[i] == ref[i]);
		}
	}

	std::cout << "debug: " << hostDebug[0] << "\n";
	// time is returned in ns, so don't need to divide by bytes to get GBPS
    std::cout << "GPU Time: " << time / 1000000 << " ms\n";
	std::cout << "Throughput: " << (((long) size) * 4 * 2)/(time) << " GBPS\n";

	// Cleanup.
	program.teardown();
	in.teardown();
	out.teardown();
	debug.teardown();
	device.teardown();
	instance.teardown();
	return 0;
}
