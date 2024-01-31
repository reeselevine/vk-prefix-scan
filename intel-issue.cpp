#include <vector>
#include <iostream>
#include <easyvk.h>
#include <cassert>
#include <vector>
#include <unistd.h>

int main(int argc, char* argv[]) {
  int workgroupSize = 32;
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
	auto out = easyvk::Buffer(device, 2, sizeof(uint32_t));

	out.clear();
	std::vector<easyvk::Buffer> bufs = {out};

	std::vector<uint32_t> spvCode = 
	#include "build/intel-issue.cinit"
	;
	auto program = easyvk::Program(device, spvCode, bufs);

	program.setWorkgroups(numWorkgroups);
	program.setWorkgroupSize(workgroupSize);

	// Run the kernel.
	program.initialize("test");

	float time = program.runWithDispatchTiming();
	std::cout << "Thread 16 recorded subgroup size: " << out.load<uint>(0) << "\n";
	std::cout << "Thread 16 recorded subgroup id: " << out.load<uint>(1) << "\n";

	// Cleanup.
	program.teardown();
	out.teardown();
	device.teardown();
	instance.teardown();
	return 0;
}
