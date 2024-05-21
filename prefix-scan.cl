#define BATCH_SIZE 8


__kernel void prefix_scan(
  __global atomic_uint *partition,
  __global uint * debug) {
    
  __local uint part_id;
  // workgroup ID
  if (get_local_id(0) == 0) {
    part_id = atomic_fetch_add(partition, 1);
  }

  // ensure all threads see workgroup ID
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  __local uint exclusive_prefix;

  if (get_local_id(0) == 0) {
    exclusive_prefix = 0;
  }

  // threads in the first subgroup of all workgroups except workgroup 0.
  if (part_id != 0 && get_sub_group_id() == 0) {
    
    // normally do some work above here
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint local_prefix = 5;
    uint scanned_prefix = sub_group_scan_inclusive_add(local_prefix);
    if (get_sub_group_local_id() == get_sub_group_size() - 1) {
          exclusive_prefix += scanned_prefix;
    }
  }


  // ensure all threads in the block see exclusive_prefix  
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  if (get_local_id(0) == 31) {
    debug[0] = exclusive_prefix;
  }

  if (get_local_id(0) == 32) {
    debug[1] = exclusive_prefix;
  }



}
