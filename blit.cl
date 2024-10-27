__kernel void blit(__global uint *in, __global uint *out) {
  // uint my_id = get_global_id(0) * 8;
  // for (int i = 0; i < 8; i++) {
  //   out[my_id + i] = in[my_id + i];
  // }

  out[get_global_id(0)] = in[get_global_id(0)];

}
