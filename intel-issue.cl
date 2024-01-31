__kernel void test(__global uint *out) {
  if (get_local_id(0) == 0) {
    out[0] = get_sub_group_size();
    out[1] = get_sub_group_id();
  }

  if (get_local_id(0) == 16) {
    out[2] = get_sub_group_size();
    out[3] = get_sub_group_id();
  }
}
