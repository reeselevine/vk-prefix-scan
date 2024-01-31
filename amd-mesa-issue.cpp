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
  int numWorkgroups = 1;
  int deviceID = 0;
  bool enableValidationLayers = false;
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
	// Initialize instance.
	auto instance = easyvk::Instance(enableValidationLayers);
	// Get list of available physical devices.
	auto physicalDevices = instance.physicalDevices();
	// Create device from first physical device.
	auto device = easyvk::Device(instance, physicalDevices.at(deviceID));
	std::cout << "Using device: " << device.properties.deviceName << "\n";
  std::cout << "Device subgroup size: " << device.subgroupSize() << "\n";
	// Define the buffers to use in the kernel. 
	auto out = easyvk::Buffer(device, 1, sizeof(uint32_t));
	auto partitionCtr = easyvk::Buffer(device, 1, sizeof(uint32_t));

	out.clear();
	partitionCtr.store<uint>(0, 1);
	std::vector<easyvk::Buffer> bufs = {out, partitionCtr};

	std::vector<uint32_t> spvCode = 
	#include "build/amd-mesa-issue.cinit"
	;
	auto program = easyvk::Program(device, spvCode, bufs);

	program.setWorkgroups(numWorkgroups);
	program.setWorkgroupSize(workgroupSize);

	// Run the kernel.
	program.initialize("test");

	float time = program.runWithDispatchTiming();

	std::cout << "debug: " << out.load<uint>(0) << "\n";

	// Cleanup.
	program.teardown();
	out.teardown();
	partitionCtr.teardown();
	device.teardown();
	instance.teardown();
	return 0;
}
