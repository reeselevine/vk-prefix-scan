#define BATCH_SIZE 8

__kernel void memcpy(
  __global uint *in,
  __global uint *out
  //__global atomic_uint *partition
  ) {
  // __local uint part_id;
  // // first thread in each block gets its part by atomically incrementing the global partition variable.
  // if (get_local_id(0) == 0) {
  //   part_id = atomic_fetch_add(partition, 1);
  // }
  // //ensure that all threads in the block see the updated part_id
  // work_group_barrier(CLK_LOCAL_MEM_FENCE);

  // uint my_id = part_id * get_local_size(0) * BATCH_SIZE + get_local_id(0) * BATCH_SIZE;
  
  for (uint i = 0; i < BATCH_SIZE; i++) {
    out[(get_local_id(0)*8) + i] = in[(get_local_id(0) *8) + i];
  }
}
