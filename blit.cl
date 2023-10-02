__kernel void blit(__global uint *in, __global uint *out) {
  out[get_global_id(0)] = in[get_global_id(0)];
}
