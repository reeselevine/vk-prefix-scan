#define BATCH_SIZE 64
__kernel void blit(__global uint *in, __global uint *out) {
  uint my_id = get_global_id(0) * BATCH_SIZE;
  for (int i = 0; i < BATCH_SIZE; i++) {
    out[my_id + i] = in[my_id + i];
  }
}
