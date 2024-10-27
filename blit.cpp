#include <vector>
#include <iostream>
#include <easyvk.h>
#include <cassert>
#include <vector>
#include <unistd.h>
#include <numeric>

#define BATCH_SIZE 1

int main(int argc, char* argv[]) {
  int workgroupSize = 1024;
  int numWorkgroups = 1024;
  int deviceID = 1;
  bool enableValidationLayers = false;
  bool checkResults = false;
  int c;

    while ((c = getopt (argc, argv, "vct:w:d:b:p:")) != -1)
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
	auto sizeBytes = numWorkgroups * workgroupSize * BATCH_SIZE * sizeof(uint);
	// Initialize instance.
	auto instance = easyvk::Instance(enableValidationLayers);
	// Get list of available physical devices.
	auto physicalDevices = instance.physicalDevices();
	// Create device from first physical device.
	auto device = easyvk::Device(instance, physicalDevices.at(deviceID));
	std::cout << "Using device: " << device.properties.deviceName << "\n";
    std::cout << "Device subgroup size: " << device.subgroupSize() << "\n";
	// Define the buffers to use in the kernel. 
	
	std::vector<uint> hostIn(size, 0);
	std::vector<uint> hostOut(size, 0);

	//std::iota(std::begin(hostIn), std::end(hostIn), 0); // fill with increasing numbers till end 
	

	auto in = easyvk::Buffer(device, sizeBytes, true);
	in.fill(1U);

	auto out = easyvk::Buffer(device, sizeBytes, true);
	out.fill(0U); 
	 

	std::vector<easyvk::Buffer> bufs = {in, out};


	std::vector<uint32_t> spvCode = 
	#include "build/blit.cinit"
	;
	auto program = easyvk::Program(device, spvCode, bufs);

	program.setWorkgroups(numWorkgroups);
	program.setWorkgroupSize(workgroupSize);
	program.setWorkgroupMemoryLength(workgroupSize*sizeof(uint), 0);

	// Run the kernel.
	program.initialize("blit");
	float time = program.runWithDispatchTiming();
	  
  out.load(hostOut.data(), sizeBytes);

	bool ans = hostOut[size - 1] == 1 ? 1 : 0;
	if (checkResults) {
		for (int i = 0; i < size; i++) {
			//std::cout << "out[" << i << "]: " << hostOut[i] << 
			std::cout << "out[" << i << "]: " << hostOut[i] << ", ref:" << 1U << "\n";
			assert(hostOut[i] == 1U);
		}
	}

	

	std::cout << "debug: " << ans << "\n";
	// time is returned in ns, so don't need to divide by bytes to get GBPS
    std::cout << "GPU Time: " << time / 1000000 << " ms\n";
	std::cout << "Throughput: " << (((long) size) * 4 * 2)/(time) << " GBPS\n";

	// Cleanup.
	program.teardown();
	in.teardown();
	out.teardown();
	device.teardown();
	instance.teardown();
	return 0;
}
