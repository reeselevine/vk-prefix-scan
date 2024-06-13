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
  int workgroupSize = 32;
  int numWorkgroups = 1;
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
    auto size = (numWorkgroups * workgroupSize * BATCH_SIZE);

	// Initialize instance.
	auto instance = easyvk::Instance(enableValidationLayers);
	// Get list of available physical devices.
	auto physicalDevices = instance.physicalDevices();
	// Create device from first physical device.
	auto device = easyvk::Device(instance, physicalDevices.at(deviceID));
	std::cout << "Using device: " << device.properties.deviceName << "\n";
    std::cout << "Device subgroup size: " << device.subgroupSize() << "\n";
	// Define the buffers to use in the kernel. 
	
	std::vector<uint> InVector(size, 42);
	std::vector<uint> OutVector(size, 0);
	//std::vector<uint> partitionCtrVector(size, 0);
	
	std::cout << "peanut ";
	auto in = easyvk::Buffer(device, InVector.data(), size, sizeof(uint));
	std::cout << "peanut ";
	auto out = easyvk::Buffer(device, OutVector.data(), size, sizeof(uint));
	std::cout << "peanut ";
	//auto partitionCtr = easyvk::Buffer(device, partitionCtrVector.data(), numWorkgroups, sizeof(uint));

	
	//std::vector<easyvk::Buffer> bufs = {in, out, partitionCtr};
	std::vector<easyvk::Buffer> bufs = {in, out};

	std::vector<uint32_t> spvCode = 
	#include "build/memcpy.cinit"
	;
	auto program = easyvk::Program(device, spvCode, bufs);

	program.setWorkgroups(numWorkgroups);
	program.setWorkgroupSize(workgroupSize);
	//program.setWorkgroupMemoryLength(workgroupSize*sizeof(uint32_t), 0);

	std::cout << "yo\n";
	// Run the kernel.
	program.initialize("memcpy");

	float time = program.runWithDispatchTiming();

	std::vector<uint> Outfr(size, 0);

	//out.returnAccess(device, size);

	out.load(device, Outfr, size);

	if (checkResults) {
		for (int i = 0; i < size; i++) {
			//std::cout << i << std::endl;
			std::cout << "out[" << i << "]: " << Outfr[i] << ", ref:" << i << "\n";
		}
	}

	std::cout << "GPU Time: " << time / 1000000 << " ms\n";
	std::cout << "Throughput: " << (((long) size) * 4 * 2)/(time) << " GBPS\n";

	// Cleanup.
	program.teardown();
	in.teardown();
	//std::cout << "out does not teardown cuz I already did it." << std::endl;
	out.teardown();
	device.teardown();
	instance.teardown();
	return 0;
}
