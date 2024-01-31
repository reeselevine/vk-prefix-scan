__kernel void test(
  __global uint *out,
  __global atomic_uint *partition) {
  if (get_sub_group_id() == 0) {
    bool done = false;
    while (!done) {
      uint flag = atomic_load_explicit(partition, memory_order_acquire);
      if (sub_group_all(flag)) {
        done = true;
      }
    }
  }
  if (get_local_id(0) == 0) {
    out[0] = 1;
  }
}
