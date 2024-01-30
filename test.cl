__kernel void test(
  __global uint *out,
  __global atomic_uint *partition,
  __global uint * debug) {

  __local uint part_id;

  if (get_local_id(0) == 0) {
    part_id = atomic_fetch_add(partition, 1);
  }

  work_group_barrier(CLK_LOCAL_MEM_FENCE);
 
  out[part_id + get_local_id(0)] = 42;
}
