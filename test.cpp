#include <vector>
#include <iostream>
#include <easyvk.h>
#include <cassert>
#include <vector>
#include <unistd.h>

int main(int argc, char* argv[]) {
  int workgroupSize = 128;
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
  auto size = numWorkgroups * workgroupSize;
	// Initialize instance.
	auto instance = easyvk::Instance(enableValidationLayers);
	// Get list of available physical devices.
	auto physicalDevices = instance.physicalDevices();
	// Create device from first physical device.
	auto device = easyvk::Device(instance, physicalDevices.at(deviceID));
	std::cout << "Using device: " << device.properties.deviceName << "\n";
  std::cout << "Device subgroup size: " << device.subgroupSize() << "\n";
	// Define the buffers to use in the kernel. 
	auto out = easyvk::Buffer(device, size, sizeof(uint32_t));
	auto partition = easyvk::Buffer(device, 1, sizeof(uint32_t));
	auto debug = easyvk::Buffer(device, 1, sizeof(uint32_t));

	out.clear();
	std::vector<easyvk::Buffer> bufs = {out, partition, debug};

	std::vector<uint32_t> spvCode = 
	#include "build/test.cinit"
	;
	auto program = easyvk::Program(device, spvCode, bufs);

	program.setWorkgroups(numWorkgroups);
	program.setWorkgroupSize(workgroupSize);

	// Run the kernel.
	program.initialize("test");

	float time = program.runWithDispatchTiming();

	std::cout << "debug: " << debug.load<uint>(0) << "\n";

	// Check the output.
	if (checkResults) {
		for (int i = 0; i < size; i++) {
			std::cout << "out[" << i << "]: " << out.load<uint>(i) << "\n";
		}
	}


	// Cleanup.
	program.teardown();
	out.teardown();
	partition.teardown();
	debug.teardown();
	device.teardown();
	instance.teardown();
	return 0;
}
