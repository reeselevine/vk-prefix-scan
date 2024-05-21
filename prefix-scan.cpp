#include <vector>
#include <iostream>
#include <easyvk.h>
#include <cassert>
#include <vector>
#include <unistd.h>

#define BATCH_SIZE 8

void computeReferencePrefixSum(uint32_t* ref, int size) {
  ref[0] = 0;
  for (int i = 1; i < size; i++) {
    ref[i] = ref[i - 1] + i;
  }
}

int main(int argc, char* argv[]) {
  int workgroupSize = 64;
  int numWorkgroups = 32;
  int deviceID = 0;
  bool enableValidationLayers = false;
  bool checkResults = false;
  int c;

    while ((c = getopt (argc, argv, "vct:w:d:")) != -1)
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
	// Initialize instance.
	auto instance = easyvk::Instance(enableValidationLayers);
	// Get list of available physical devices.
	auto physicalDevices = instance.physicalDevices();
	// Create device from first physical device.
	auto device = easyvk::Device(instance, physicalDevices.at(deviceID));
	std::cout << "Using device: " << device.properties.deviceName << "\n";
  std::cout << "Device subgroup size: " << device.subgroupSize() << "\n";
	// Define the buffers to use in the kernel. 
	auto in = easyvk::Buffer(device, size, sizeof(uint32_t));
	auto out = easyvk::Buffer(device, size, sizeof(uint32_t));
	auto prefixStates = easyvk::Buffer(device, numWorkgroups, 3*sizeof(uint32_t));
	auto partitionCtr = easyvk::Buffer(device, 1, sizeof(uint32_t));
	auto debug = easyvk::Buffer(device, 64, sizeof(uint32_t));

	// Write initial values to the buffers.
	for (int i = 0; i < size; i++) {
		// The buffer provides an untyped view of the memory, so you must specify
		// the type when using the load/store method. 
		in.store<uint32_t>(i, i);
	}
	
	out.clear();
	prefixStates.clear();
	partitionCtr.clear();
	std::vector<easyvk::Buffer> bufs = { partitionCtr, debug};

	std::vector<uint32_t> spvCode = 
	#include "build/prefix-scan.cinit"
	;
	auto program = easyvk::Program(device, spvCode, bufs);

	program.setWorkgroups(numWorkgroups);
	program.setWorkgroupSize(workgroupSize);
	program.setWorkgroupMemoryLength(workgroupSize*sizeof(uint32_t), 0);

	// Run the kernel.
	program.initialize("prefix_scan");

	float time = program.runWithDispatchTiming();


	std::cout << "first subgroup observes exclusive_prefix " << debug.load<uint>(0) << "\n";
	std::cout << "second subgroup observes exclusive_prefix " << debug.load<uint>(1) << "\n";

	// for (int i = 1; i < 65; i++) {
	// 	std::cout << "debug: " << debug.load<uint>(i) << "\n";
	// }
	

	// Check the output.
	if (checkResults) {
		uint32_t ref = 0;
		for (int i = 0; i < size; i++) {
			ref += i;
			std::cout << "out[" << i << "]: " << out.load<uint>(i) << ", ref:" << ref << "\n";
			assert(out.load<uint>(i) == ref);
		}
	}

	// time is returned in ns, so don't need to divide by bytes to get GBPS
        std::cout << "GPU Time: " << time / 1000000 << " ms\n";
	std::cout << "Throughput: " << (((long) size) * 4 * 2)/(time) << " GBPS\n";

	// Cleanup.
	program.teardown();
	in.teardown();
	out.teardown();
	prefixStates.teardown();
	partitionCtr.teardown();
	device.teardown();
	instance.teardown();
	return 0;
}
