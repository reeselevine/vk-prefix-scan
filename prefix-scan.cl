#define BATCH_SIZE 8

__kernel void prefix_scan(
  __global uint *in, 
  __global uint *out,
  __local uint *scratch,
  __global atomic_uint *partition,
  __global uint * debug) {

  __local uint part_id;
  // first thread in each block gets its part by atomically incrementing the global partition variable.
  if (get_local_id(0) == 0) {
    part_id = atomic_fetch_add(partition, 1);
  }
  //ensure that all threads in the block see the updated part_id
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  __local uint temp;
  __local uint inclusive_scan;


  //for (uint i = 0; i < 100; i++) {
  int scan_type;
  scan_type = debug[0];

  // each thread works on items indexed on its partition and position in the block
  uint my_id = part_id * get_local_size(0) * BATCH_SIZE + get_local_id(0) * BATCH_SIZE;

  // load work into private memory and compute thread local prefix sum
  uint values[BATCH_SIZE];
  uint sum = in[my_id];
  values[0] = sum;
  for (uint i = 1; i < BATCH_SIZE; i++) {
    sum += in[my_id + i];
    values[i] = sum;
  }

  switch (scan_type)
  {
  case 'a':
    {
      // store inclusive thread prefix to local memory so that a block wide prefix can be computed
      scratch[get_local_id(0)] = sum;
      work_group_barrier(CLK_LOCAL_MEM_FENCE);

      // perform raking exclusive sum, where only threads in the first subgroup do any work
      if (get_sub_group_id() == 0) {
        // each thread rakes across a block of the local prefixes
        uint rake_batch_size = get_local_size(0)/get_sub_group_size();
        uint start = get_local_id(0) * rake_batch_size;
        for (uint i = start + 1; i < start + rake_batch_size; i++) {
          scratch[i] += scratch[i - 1];
        }
        uint partial_sum = scratch[start + rake_batch_size - 1];
        uint prefix = sub_group_scan_exclusive_add(partial_sum);
        for (uint i = start; i < start + rake_batch_size; i++) {
          scratch[i] += prefix;
        }
        // synchronize scratch memory across threads in subgroup 
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
      }

      break;
    }

  case 'b':
    {
      uint scan = sub_group_scan_inclusive_add(sum);

      if ( (get_local_id(0) % 32) == (32 - 1) ) {
        scratch[get_local_id(0) / 32] = scan;
      } 

      work_group_barrier(CLK_LOCAL_MEM_FENCE);
      if (get_local_id(0) < 32) {
        scratch[get_local_id(0)] = sub_group_scan_exclusive_add(scratch[get_local_id(0)]);
      }
      work_group_barrier(CLK_LOCAL_MEM_FENCE);
      scratch[get_local_id(0)] = scan + scratch[get_local_id(0)/ 32];
      break;
    }
    
  case 'c':
    {
      // load input into shared memory 
      uint BLOCK_SIZE = get_local_size(0);
      scratch[get_local_id(0)] = sum;
      
      work_group_barrier(CLK_LOCAL_MEM_FENCE);
      // build the sum in place up the tree
      const ushort ai = 2 * get_local_id(0) + 1;
      const ushort bi = 2 * get_local_id(0) + 2;

      // build the sum in place up the tree
      if (BLOCK_SIZE >=    2) {if (get_local_id(0) < (BLOCK_SIZE >>  1) ) {scratch[   1 * bi - 1] += scratch[   1 * ai - 1];} if ((BLOCK_SIZE >>  0) > 32) work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=    4) {if (get_local_id(0) < (BLOCK_SIZE >>  2) ) {scratch[   2 * bi - 1] += scratch[   2 * ai - 1];} if ((BLOCK_SIZE >>  1) > 32) work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=    8) {if (get_local_id(0) < (BLOCK_SIZE >>  3) ) {scratch[   4 * bi - 1] += scratch[   4 * ai - 1];} if ((BLOCK_SIZE >>  2) > 32) work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=   16) {if (get_local_id(0) < (BLOCK_SIZE >>  4) ) {scratch[   8 * bi - 1] += scratch[   8 * ai - 1];} if ((BLOCK_SIZE >>  3) > 32) work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=   32) {if (get_local_id(0) < (BLOCK_SIZE >>  5) ) {scratch[  16 * bi - 1] += scratch[  16 * ai - 1];} if ((BLOCK_SIZE >>  4) > 32) work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=   64) {if (get_local_id(0) < (BLOCK_SIZE >>  6) ) {scratch[  32 * bi - 1] += scratch[  32 * ai - 1];} }
      if (BLOCK_SIZE >=  128) {if (get_local_id(0) < (BLOCK_SIZE >>  7) ) {scratch[  64 * bi - 1] += scratch[  64 * ai - 1];} }
      if (BLOCK_SIZE >=  256) {if (get_local_id(0) < (BLOCK_SIZE >>  8) ) {scratch[ 128 * bi - 1] += scratch[ 128 * ai - 1];} }
      if (BLOCK_SIZE >=  512) {if (get_local_id(0) < (BLOCK_SIZE >>  9) ) {scratch[ 256 * bi - 1] += scratch[ 256 * ai - 1];} }
      if (BLOCK_SIZE >= 1024) {if (get_local_id(0) < (BLOCK_SIZE >> 10) ) {scratch[ 512 * bi - 1] += scratch[ 512 * ai - 1];} }
        
      // clear the last element
      if (get_local_id(0) == 0) { inclusive_scan = scratch[BLOCK_SIZE - 1]; scratch[BLOCK_SIZE - 1] = 0; }
      work_group_barrier(CLK_LOCAL_MEM_FENCE);
        
      // traverse down the tree building the scan in place
      if (BLOCK_SIZE >=    2){
          if (get_local_id(0) <    1) {
              scratch[(BLOCK_SIZE >>  1) * bi - 1] += scratch[(BLOCK_SIZE >>  1) * ai - 1];
              scratch[(BLOCK_SIZE >>  1) * ai - 1] = scratch[(BLOCK_SIZE >>  1) * bi - 1] - scratch[(BLOCK_SIZE >>  1) * ai - 1];
          }
      }

      if (BLOCK_SIZE >=    4){ if (get_local_id(0) <    2) {scratch[(BLOCK_SIZE >>  2) * bi - 1] += scratch[(BLOCK_SIZE >>  2) * ai - 1]; scratch[(BLOCK_SIZE >>  2) * ai - 1] = scratch[(BLOCK_SIZE >>  2) * bi - 1] - scratch[(BLOCK_SIZE >>  2) * ai - 1];} }
      if (BLOCK_SIZE >=    8){ if (get_local_id(0) <    4) {scratch[(BLOCK_SIZE >>  3) * bi - 1] += scratch[(BLOCK_SIZE >>  3) * ai - 1]; scratch[(BLOCK_SIZE >>  3) * ai - 1] = scratch[(BLOCK_SIZE >>  3) * bi - 1] - scratch[(BLOCK_SIZE >>  3) * ai - 1];} }
      if (BLOCK_SIZE >=   16){ if (get_local_id(0) <    8) {scratch[(BLOCK_SIZE >>  4) * bi - 1] += scratch[(BLOCK_SIZE >>  4) * ai - 1]; scratch[(BLOCK_SIZE >>  4) * ai - 1] = scratch[(BLOCK_SIZE >>  4) * bi - 1] - scratch[(BLOCK_SIZE >>  4) * ai - 1];} }
      if (BLOCK_SIZE >=   32){ if (get_local_id(0) <   16) {scratch[(BLOCK_SIZE >>  5) * bi - 1] += scratch[(BLOCK_SIZE >>  5) * ai - 1]; scratch[(BLOCK_SIZE >>  5) * ai - 1] = scratch[(BLOCK_SIZE >>  5) * bi - 1] - scratch[(BLOCK_SIZE >>  5) * ai - 1];} }
      if (BLOCK_SIZE >=   64){ if (get_local_id(0) <   32) {scratch[(BLOCK_SIZE >>  6) * bi - 1] += scratch[(BLOCK_SIZE >>  6) * ai - 1]; scratch[(BLOCK_SIZE >>  6) * ai - 1] = scratch[(BLOCK_SIZE >>  6) * bi - 1] - scratch[(BLOCK_SIZE >>  6) * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=  128){ if (get_local_id(0) <   64) {scratch[(BLOCK_SIZE >>  7) * bi - 1] += scratch[(BLOCK_SIZE >>  7) * ai - 1]; scratch[(BLOCK_SIZE >>  7) * ai - 1] = scratch[(BLOCK_SIZE >>  7) * bi - 1] - scratch[(BLOCK_SIZE >>  7) * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=  256){ if (get_local_id(0) <  128) {scratch[(BLOCK_SIZE >>  8) * bi - 1] += scratch[(BLOCK_SIZE >>  8) * ai - 1]; scratch[(BLOCK_SIZE >>  8) * ai - 1] = scratch[(BLOCK_SIZE >>  8) * bi - 1] - scratch[(BLOCK_SIZE >>  8) * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=  512){ if (get_local_id(0) <  256) {scratch[(BLOCK_SIZE >>  9) * bi - 1] += scratch[(BLOCK_SIZE >>  9) * ai - 1]; scratch[(BLOCK_SIZE >>  9) * ai - 1] = scratch[(BLOCK_SIZE >>  9) * bi - 1] - scratch[(BLOCK_SIZE >>  9) * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >= 1024){ if (get_local_id(0) <  512) {scratch[(BLOCK_SIZE >> 10) * bi - 1] += scratch[(BLOCK_SIZE >> 10) * ai - 1]; scratch[(BLOCK_SIZE >> 10) * ai - 1] = scratch[(BLOCK_SIZE >> 10) * bi - 1] - scratch[(BLOCK_SIZE >> 10) * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
          
      if (get_local_id(0) != BLOCK_SIZE - 1) { scratch[get_local_id(0)] = scratch[get_local_id(0) + 1];} else {scratch[BLOCK_SIZE - 1] = inclusive_scan;} 
      break;
    }
  default:
    {
      return;
    }  
  }

  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  uint total_exclusive_prefix = 0;
  // scratch contains an inclusive prefix per thread, so the exclusive prefix is grabbed from 
  // the previous thread's scratch location
  if (get_local_id(0) != 0) {
    total_exclusive_prefix = scratch[get_local_id(0) - 1];
  }

  for (uint i = 0; i < BATCH_SIZE; i++) {
    out[my_id + i] = values[i] + total_exclusive_prefix;
  }
 // }
}