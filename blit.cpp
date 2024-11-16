#include <vector>
#include <iostream>
#include <easyvk.h>
#include <cassert>
#include <vector>
#include <unistd.h>
#include <numeric>

//#define BATCH_SIZE 64

void computeReferencePrefixSum(uint32_t* ref, int size, int overflow, int start) {
  ref[0] = start;
  if (overflow){
	for (int i = 1; i < size; i++) {
    	ref[i] = ref[i - 1] + i;
		// may be incorrect with diff types of the implementation var
  	}
  }else{
	for (int i = 1; i < size; i++) {
		//std::cout << "hi\n";
    	ref[i] = ref[i-1] + start;
  	}
  }
}

int main(int argc, char* argv[]) {
  int workgroupSize = 1024;
  int numWorkgroups = 1024;
  int deviceID = 1;
  bool enableValidationLayers = false;
  bool checkResults = false;
  int c;
  int p = 1;
  char alg = 'a';
  int BATCH_SIZE = 8;
  int alt = 1;

    while ((c = getopt (argc, argv, "vct:w:d:b:p:a:s:")) != -1)
    switch (c)
      {
      case 'a':
        alt = atoi(optarg);
        break;
	  case 's':
        BATCH_SIZE = atoi(optarg);
		break;
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
	case 'p':
		p = atoi(optarg);
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
	

	std::vector<uint> hostOut(size, 0);
	std::vector<uint> ref(size, 0);

	// auto partitionCtr = easyvk::Buffer(device, sizeof(uint), true);
	// partitionCtr.fill(0U);

	auto in = easyvk::Buffer(device, sizeBytes, true);
	in.fill((uint)alt);

	auto out = easyvk::Buffer(device, sizeBytes, true);
	out.fill(0U); 
	 

	std::vector<easyvk::Buffer> bufs = {in, out};
	//std::vector<easyvk::Buffer> bufs = {in, out, prefixStates, debug};

std::vector<uint32_t> spvCode;

	if (BATCH_SIZE == 1) {
		spvCode = 
		#include "batch_size/blit1.cinit"
		;
	}else if(BATCH_SIZE == 2) {
		spvCode = 
		#include "batch_size/blit2.cinit"
		;
	}else if(BATCH_SIZE == 4) {
		spvCode = 
		#include "batch_size/blit4.cinit"
		;
	}else if(BATCH_SIZE == 8) {
		spvCode = 
		#include "batch_size/blit8.cinit"
		;
	}else if(BATCH_SIZE == 16) {
		spvCode = 
		#include "batch_size/blit16.cinit"
		;
	}else if(BATCH_SIZE == 32) {
		spvCode = 
		#include "batch_size/blit32.cinit"
		;
	}else if(BATCH_SIZE == 64) {
		spvCode = 
		#include "batch_size/blit64.cinit"
		;
	}
	auto program = easyvk::Program(device, spvCode, bufs);

	program.setWorkgroups(numWorkgroups);
	program.setWorkgroupSize(workgroupSize);
	//program.setWorkgroupMemoryLength(workgroupSize*sizeof(uint), 0);

	// Run the kernel.
	program.initialize("blit");
	float time = program.runWithDispatchTiming();
	out.load(hostOut.data(), sizeBytes);

	
	uint cs = hostOut[size - 1] == alt ? 1 : 0;
	
	//std::cout << "debug: " << hostDebug[0] << "\n";
	if (checkResults) {
		computeReferencePrefixSum(ref.data(), size, false, alt);
		for (int i = 0; i < size; i++) {
			//std::cout << "out[" << i << "]: " << hostOut[i] << 
			std::cout << "out[" << i << "]: " << hostOut[i] << ", ref:" << alt << "\n";
			assert(hostOut[i] == alt);
		}
	}

	

	std::cout << "debug: " << cs << "\n";
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
