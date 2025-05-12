#include <vector>
#include <iostream>
#include <easyvk.h>
#include <cassert>
#include <vector>
#include <unistd.h>
#include <numeric>

//#define BATCH_SIZE 8

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
  int mem_type = 4;
  int BATCH_SIZE = 4;
  char alg = 'a';
  int alt = 1;

    while ((c = getopt (argc, argv, "vct:w:d:b:p:a:s:m:")) != -1)
    switch (c)
      {
      case 'a':
        alt = atoi(optarg);
        break;
	  case 'm':
        mem_type = atoi(optarg);
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
    auto size = numWorkgroups * workgroupSize * BATCH_SIZE * mem_type;
	if (size > 1073741824) {
		for (int i = 0; i < 100; i++) {
			std::cout << "OVERFLOW ALERT" << "\n";
		}
	}
	auto sizeBytes = numWorkgroups * workgroupSize * BATCH_SIZE * (sizeof(uint)) * mem_type;
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
	std::vector<uint> hostDebug(3, 0);
	std::vector<uint> ref(size, 0);

	//std::iota(std::begin(hostIn), std::end(hostIn), 0); // fill with increasing numbers till end 

	
	hostDebug[0] = alg;
	
	hostDebug[1] = p;

	auto in = easyvk::Buffer(device, sizeBytes, true);
	//in.store(hostIn.data(), sizeBytes);
	in.fill((uint)alt);

	auto debug = easyvk::Buffer(device, sizeof(uint)*3, true);
	debug.store(hostDebug.data(), sizeof(uint)*3);

	auto out = easyvk::Buffer(device, sizeBytes, true);
	auto prefixStates = easyvk::Buffer(device, numWorkgroups*1*sizeof(uint), true);
	auto partitionCtr = easyvk::Buffer(device, sizeof(uint), true);
	

	partitionCtr.fill(0U);
	prefixStates.fill(0U);
	out.fill(0U); 
	 

	std::vector<easyvk::Buffer> bufs = {in, out, prefixStates, partitionCtr, debug};
	// std::vector<easyvk::Buffer> bufs = {in, out, prefixStates, debug};

	std::string dir = "batch_size/prefix-scan" + std::to_string(BATCH_SIZE) + ".cinit";

	std::vector<uint32_t> spvCode;

	if (mem_type == 4) {
		if (BATCH_SIZE == 1) {
			spvCode = 
			#include "batch_size/prefix-scan1_uint4.cinit"
			;
		}else if(BATCH_SIZE == 2) {
			spvCode = 
			#include "batch_size/prefix-scan2_uint4.cinit"
			;
		}else if(BATCH_SIZE == 4) {
			spvCode = 
			#include "batch_size/prefix-scan4_uint4.cinit"
			;
		}else if(BATCH_SIZE == 8) {
			spvCode = 
			#include "batch_size/prefix-scan8_uint4.cinit"
			;
		}else if(BATCH_SIZE == 16) {
			spvCode = 
			#include "batch_size/prefix-scan16_uint4.cinit"
			;
		}else if(BATCH_SIZE == 32) {
			spvCode = 
			#include "batch_size/prefix-scan32_uint4.cinit"
			;
		}else if(BATCH_SIZE == 64) {
			spvCode = 
			#include "batch_size/prefix-scan64_uint4.cinit"
			;
		}else if(BATCH_SIZE == 128) {
			spvCode = 
			#include "batch_size/prefix-scan128_uint4.cinit"
			;
		}else if(BATCH_SIZE == 256) {
			spvCode = 
			#include "batch_size/prefix-scan256_uint4.cinit"
			;
		}else if(BATCH_SIZE == 512) {
			spvCode = 
			#include "batch_size/prefix-scan512_uint4.cinit"
			;
		}else if(BATCH_SIZE == 1024) {
			spvCode = 
			#include "batch_size/prefix-scan1024_uint4.cinit"
			;
		}else if(BATCH_SIZE == 2048) {
			spvCode = 
			#include "batch_size/prefix-scan2048_uint4.cinit"
			;
		}
	}else if (mem_type == 2){
		if (BATCH_SIZE == 1) {
			spvCode = 
			#include "batch_size/prefix-scan1_uint2.cinit"
			;
		}else if(BATCH_SIZE == 2) {
			spvCode = 
			#include "batch_size/prefix-scan2_uint2.cinit"
			;
		}else if(BATCH_SIZE == 4) {
			spvCode = 
			#include "batch_size/prefix-scan4_uint2.cinit"
			;
		}else if(BATCH_SIZE == 8) {
			spvCode = 
			#include "batch_size/prefix-scan8_uint2.cinit"
			;
		}else if(BATCH_SIZE == 16) {
			spvCode = 
			#include "batch_size/prefix-scan16_uint2.cinit"
			;
		}else if(BATCH_SIZE == 32) {
			spvCode = 
			#include "batch_size/prefix-scan32_uint2.cinit"
			;
		}else if(BATCH_SIZE == 64) {
			spvCode = 
			#include "batch_size/prefix-scan64_uint2.cinit"
			;
		}else if(BATCH_SIZE == 128) {
			spvCode = 
			#include "batch_size/prefix-scan128_uint2.cinit"
			;
		}else if(BATCH_SIZE == 256) {
			spvCode = 
			#include "batch_size/prefix-scan256_uint2.cinit"
			;
		}else if(BATCH_SIZE == 512) {
			spvCode = 
			#include "batch_size/prefix-scan512_uint2.cinit"
			;
		}else if(BATCH_SIZE == 1024) {
			spvCode = 
			#include "batch_size/prefix-scan1024_uint2.cinit"
			;
		}else if(BATCH_SIZE == 2048) {
			spvCode = 
			#include "batch_size/prefix-scan2048_uint2.cinit"
			;
		}
	}

	// std::vector<uint32_t> spvCode = 
	// #include "build/prefix-scan.cinit"
	// ;
	auto program = easyvk::Program(device, spvCode, bufs);

	program.setWorkgroups(numWorkgroups);
	program.setWorkgroupSize(workgroupSize);
	program.setWorkgroupMemoryLength(workgroupSize*sizeof(uint), 0);

	// Run the kernel.
	program.initialize("prefix_scan");
	float time = program.runWithDispatchTiming();
	out.load(hostOut.data(), sizeBytes);
	debug.load(hostDebug.data(), sizeof(uint) * 3);

	
	if (alt == 1) {
		hostDebug[0] = hostOut[size - 1] == size ? 1 : 0;
	}else{
		hostDebug[0] = hostOut[size - 1] == size * alt ? 1 : 0;
	}
	
	if (checkResults) {
		computeReferencePrefixSum(ref.data(), size, false, alt);
		for (int i = 0; i < size; i++) {
			//std::cout << "out[" << i << "]: " << hostOut[i] << 
			std::cout << "out[" << i << "]: " << hostOut[i] << ", ref:" << ref[i] << "\n";
			//assert(hostOut[i] == ref[i]);
		}
	}
	std::cout << "debug check: " << hostDebug[2] << "\n";
	

	std::cout << "debug: " << hostDebug[0] << "\n";
	// time is returned in ns, so don't need to divide by bytes to get GBPS
    std::cout << "GPU Time: " << time / 1000000 << " ms\n";
	std::cout << "Throughput: " << (((long) size) * 4 * 2)/(time) << " GBPS\n";

	// Cleanup.
	program.teardown();
	in.teardown();
	out.teardown();
	prefixStates.teardown();
	partitionCtr.teardown();
	debug.teardown();
	device.teardown();
	instance.teardown();
	return 0;
}
