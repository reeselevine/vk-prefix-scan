#include <vector>
#include <iostream>
#include <easyvk.h>
#include <cassert>
#include <vector>
#include <unistd.h>

int main(int argc, char *argv[])
{
  int workgroupSize = 64;
  int numWorkgroups = 32;
  int deviceID = 0;
  bool enableValidationLayers = false;
  bool checkResults = false;
  int c;

  while ((c = getopt(argc, argv, "vct:w:d:")) != -1)
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
      abort();
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
  auto in = easyvk::Buffer(device, size, sizeof(uint32_t));
  auto out = easyvk::Buffer(device, size, sizeof(uint32_t));

  // Write initial values to the buffers.
  for (int i = 0; i < size; i++)
  {
    // The buffer provides an untyped view of the memory, so you must specify
    // the type when using the load/store method.
    in.store<uint32_t>(i, i);
  }
  out.clear();
  std::vector<easyvk::Buffer> bufs = {in, out};

  std::vector<uint32_t> spvCode =
  #include "build/blit.cinit"
  ;
  auto program = easyvk::Program(device, spvCode, bufs);

  program.setWorkgroups(numWorkgroups);
  program.setWorkgroupSize(workgroupSize);

  // Run the kernel.
  program.initialize("blit");

  float time = program.runWithDispatchTiming();

  // Check the output.
  if (checkResults)
  {
    for (int i = 0; i < size; i++)
    {
      //			std::cout << "in[" << i << "]: " << in.load<uint>(i) << "\n";
      //			std::cout << "out[" << i << "]: " << out.load<uint>(i) << "\n";
      assert(out.load<uint>(i) == in.load<uint>(i));
    }
  }

  // time is returned in ns, so don't need to divide by bytes to get GBPS
  std::cout << "GPU Time: " << time / 1000000 << " ms\n";
  std::cout << "Throughput: " << (((long)size) * 4 * 2) / (time) << " GBPS\n";

  // Cleanup.
  program.teardown();
  in.teardown();
  out.teardown();
  device.teardown();
  instance.teardown();
  return 0;
}
