#define U4
#define BATCH_SIZE 2048
#define FLG_A 1U
#define FLG_P 2U
#define ANTI_MASK 30

#define MASK ~(3 << ANTI_MASK)

// uint4 prefix_sum_inclusive(uint4 v) {
//     v.y += v.x;                  // v = ( v.x, v.x+v.y, v.z, v.w )
//     v.z += v.y;                  // v = ( v.x, v.x+v.y, v.x+v.y+v.z, v.w )
//     v.w += v.z;                  // v = ( v.x, v.x+v.y, v.x+v.y+v.z, v.x+v.y+v.z+v.w )
//     return v;
// }

inline void uint4_prefix_sum_inclusive_batch(uint4 *values) {
    // First, prefix scan the first uint4 element
    values[0].y += values[0].x;
    values[0].z += values[0].y;
    values[0].w += values[0].z;

    // Then, for remaining elements
    for (uint i = 1; i < BATCH_SIZE; i++) {
        // Add last component from previous
        uint prev = values[i - 1].w;

        values[i].x += prev;
        values[i].y += values[i].x;
        values[i].z += values[i].y;
        values[i].w += values[i].z;
    }
}


inline void uint2_prefix_sum_inclusive_batch(uint2 *values) {
    // First, prefix scan the first uint2 element
    values[0].y += values[0].x;

    // Then, for remaining elements
    for (uint i = 1; i < BATCH_SIZE; i++) {
        // Add last component from previous
        uint prev = values[i - 1].y;

        values[i].x += prev;
        values[i].y += values[i].x;
    }
}



__kernel void prefix_scan(
  __global uint4 *in, 
  __global uint4 *out,
  __local uint *scratch,
  __global atomic_uint *prefix_states,
  __global atomic_uint *partition,
  __global uint * debug) {
  __local uint part_id;
  // first thread in each block gets its part by atomically incrementing the global partition variable.
  if (get_local_id(0) == 0) {
    part_id = atomic_fetch_add(partition, 1);
  }
  //ensure that all threads in the block see the updated part_id
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  __local uint exclusive_prefix;
  __local uint temp;
  __local uint inclusive_scan;

  int scan_type;
  int p;

  scan_type = debug[0];
  p = debug[1];

  // each thread works on items indexed on its partition and position in the block
  uint my_id = part_id * get_local_size(0) * BATCH_SIZE + get_local_id(0) * BATCH_SIZE;

  #ifdef U4
      uint4 values[BATCH_SIZE];
      for (uint i = 0; i < BATCH_SIZE; i++) {
        values[i] = in[my_id + i];
      }
      uint4_prefix_sum_inclusive_batch(values);
      scratch[get_local_id(0)] = values[BATCH_SIZE - 1].w;
  #else
      uint2 values[BATCH_SIZE];
      for (uint i = 0; i < BATCH_SIZE; i++) {
        values[i] = in[my_id + i];
      }
      uint2_prefix_sum_inclusive_batch(values);
      scratch[get_local_id(0)] = values[BATCH_SIZE - 1].y;
  #endif


  

  switch (scan_type)
  {
  case 'a':
    {
      // store inclusive thread prefix to local memory so that a block wide prefix can be computed
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
        //sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        //debug[2] = scratch[3];
        
      }   
      break;
    }

  case 'b':
    {
    // copy values to shared memory
    
    // scratch[get_local_id(0)] = sum[3];
    // work_group_barrier(CLK_LOCAL_MEM_FENCE);

    // //  first warp reduces all values
    // if (get_sub_group_id() == 0){


    //     //  interleaved addressing to reduce values into 0...31
    //     for (ushort i = 1; i < get_local_size(0) / get_sub_group_size(); i++){
    //         scratch[get_local_id(0)] += scratch[get_local_id(0) + get_sub_group_size() * i];
    //     }

    //     // Perform the reduction using sub_group_reduce_add, which is like a shuffle-based reduction
    //     uint local_prefix = sub_group_reduce_add(scratch[get_local_id(0)]);
    
    //     // Optionally, broadcast the reduced value to all threads in the subgroup (if needed)
    //     local_prefix = sub_group_broadcast(local_prefix, 0);
        
        
    //     // raking write of results to shared memory
    //     for (short i = 0; i < BLOCK_SIZE / 32; i++){
    //         shared[lid + i * 32] = value;
    //     }

    //     debug[2] = scratch[0];
    // }
    // //work_group_barrier(CLK_LOCAL_MEM_FENCE);
    // // if (lid % 32 == 0){
    // //   value = shared[lid / 32];
    // //   value = simd_broadcast_first(value);
    // // }
      break;
    }
    
  case 'c':
    {
      // load input into shared memory 
      uint BLOCK_SIZE = get_local_size(0);
      const ushort sg_size = get_sub_group_size();
      
      work_group_barrier(CLK_LOCAL_MEM_FENCE);
      // build the sum in place up the tree
      const ushort ai = 2 * get_local_id(0) + 1;
      const ushort bi = 2 * get_local_id(0) + 2;

      // build the sum in place up the tree
      if (BLOCK_SIZE >=    2) {
        
        if (get_local_id(0) < (BLOCK_SIZE >>  1) ) {
          scratch[   1 * bi - 1] += scratch[   1 * ai - 1];
        } 
        
        if ((BLOCK_SIZE >>  0) > sg_size) { 
          work_group_barrier(CLK_LOCAL_MEM_FENCE);
        } 
      }
      if (BLOCK_SIZE >=    4) {if (get_local_id(0) < (BLOCK_SIZE >>  2) ) {scratch[   2 * bi - 1] += scratch[   2 * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=    8) {if (get_local_id(0) < (BLOCK_SIZE >>  3) ) {scratch[   4 * bi - 1] += scratch[   4 * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=   16) {if (get_local_id(0) < (BLOCK_SIZE >>  4) ) {scratch[   8 * bi - 1] += scratch[   8 * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=   32) {if (get_local_id(0) < (BLOCK_SIZE >>  5) ) {scratch[  16 * bi - 1] += scratch[  16 * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=   64) {if (get_local_id(0) < (BLOCK_SIZE >>  6) ) {scratch[  32 * bi - 1] += scratch[  32 * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=  128) {if (get_local_id(0) < (BLOCK_SIZE >>  7) ) {scratch[  64 * bi - 1] += scratch[  64 * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=  256) {if (get_local_id(0) < (BLOCK_SIZE >>  8) ) {scratch[ 128 * bi - 1] += scratch[ 128 * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=  512) {if (get_local_id(0) < (BLOCK_SIZE >>  9) ) {scratch[ 256 * bi - 1] += scratch[ 256 * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >= 1024) {if (get_local_id(0) < (BLOCK_SIZE >> 10) ) {scratch[ 512 * bi - 1] += scratch[ 512 * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
        
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

      if (BLOCK_SIZE >=    4){ if (get_local_id(0) <    2) {scratch[(BLOCK_SIZE >>  2) * bi - 1] += scratch[(BLOCK_SIZE >>  2) * ai - 1]; scratch[(BLOCK_SIZE >>  2) * ai - 1] = scratch[(BLOCK_SIZE >>  2) * bi - 1] - scratch[(BLOCK_SIZE >>  2) * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=    8){ if (get_local_id(0) <    4) {scratch[(BLOCK_SIZE >>  3) * bi - 1] += scratch[(BLOCK_SIZE >>  3) * ai - 1]; scratch[(BLOCK_SIZE >>  3) * ai - 1] = scratch[(BLOCK_SIZE >>  3) * bi - 1] - scratch[(BLOCK_SIZE >>  3) * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=   16){ if (get_local_id(0) <    8) {scratch[(BLOCK_SIZE >>  4) * bi - 1] += scratch[(BLOCK_SIZE >>  4) * ai - 1]; scratch[(BLOCK_SIZE >>  4) * ai - 1] = scratch[(BLOCK_SIZE >>  4) * bi - 1] - scratch[(BLOCK_SIZE >>  4) * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=   32){ if (get_local_id(0) <   16) {scratch[(BLOCK_SIZE >>  5) * bi - 1] += scratch[(BLOCK_SIZE >>  5) * ai - 1]; scratch[(BLOCK_SIZE >>  5) * ai - 1] = scratch[(BLOCK_SIZE >>  5) * bi - 1] - scratch[(BLOCK_SIZE >>  5) * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=   64){ if (get_local_id(0) <   32) {scratch[(BLOCK_SIZE >>  6) * bi - 1] += scratch[(BLOCK_SIZE >>  6) * ai - 1]; scratch[(BLOCK_SIZE >>  6) * ai - 1] = scratch[(BLOCK_SIZE >>  6) * bi - 1] - scratch[(BLOCK_SIZE >>  6) * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=  128){ if (get_local_id(0) <   64) {scratch[(BLOCK_SIZE >>  7) * bi - 1] += scratch[(BLOCK_SIZE >>  7) * ai - 1]; scratch[(BLOCK_SIZE >>  7) * ai - 1] = scratch[(BLOCK_SIZE >>  7) * bi - 1] - scratch[(BLOCK_SIZE >>  7) * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=  256){ if (get_local_id(0) <  128) {scratch[(BLOCK_SIZE >>  8) * bi - 1] += scratch[(BLOCK_SIZE >>  8) * ai - 1]; scratch[(BLOCK_SIZE >>  8) * ai - 1] = scratch[(BLOCK_SIZE >>  8) * bi - 1] - scratch[(BLOCK_SIZE >>  8) * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >=  512){ if (get_local_id(0) <  256) {scratch[(BLOCK_SIZE >>  9) * bi - 1] += scratch[(BLOCK_SIZE >>  9) * ai - 1]; scratch[(BLOCK_SIZE >>  9) * ai - 1] = scratch[(BLOCK_SIZE >>  9) * bi - 1] - scratch[(BLOCK_SIZE >>  9) * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
      if (BLOCK_SIZE >= 1024){ if (get_local_id(0) <  512) {scratch[(BLOCK_SIZE >> 10) * bi - 1] += scratch[(BLOCK_SIZE >> 10) * ai - 1]; scratch[(BLOCK_SIZE >> 10) * ai - 1] = scratch[(BLOCK_SIZE >> 10) * bi - 1] - scratch[(BLOCK_SIZE >> 10) * ai - 1];} work_group_barrier(CLK_LOCAL_MEM_FENCE); }
          
      uint temp_tree = select(inclusive_scan, scratch[get_local_id(0) + 1], get_local_id(0) != BLOCK_SIZE - 1);

      work_group_barrier(CLK_LOCAL_MEM_FENCE);

      scratch[get_local_id(0)] = temp_tree;

      break;
    }
  default:
    {
      return;
    }  
  }

  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  // one thread in each block updates the aggregate/flag
  if (get_sub_group_id() == 0 && get_sub_group_local_id() == 0) {
    
    atomic_store_explicit(&prefix_states[part_id], (FLG_A << ANTI_MASK) | (scratch[get_local_size(0) - 1] & MASK), memory_order_relaxed);
    
    // first block does not need to look back
    if (part_id == 0) {
      atomic_store_explicit(&prefix_states[part_id], (FLG_P << ANTI_MASK) | (scratch[get_local_size(0) - 1] & MASK), memory_order_relaxed);
    }
    // might as well initialize exclusive prefix here too
    exclusive_prefix = 0;
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  if (p){
  // lookback phase (parallelized), all threads in first subgroup participate
  if (part_id != 0 && get_sub_group_id() == 0) {
    // ensure all threads in the subgroup see exclusive_prefix initialized
    int lookback_id = part_id - (get_sub_group_size() - get_sub_group_local_id());
    bool done = false;
    uint flag = FLG_P;
    uint agg = 0;
    // spin and lookback until full prefix is set
    while (!done) {
      if (lookback_id >= 0) {
         uint flagg = atomic_load_explicit(&prefix_states[lookback_id], memory_order_relaxed);
         agg = flagg & 0x3FFFFFFF;
         flag = flagg >> ANTI_MASK; // can also just give flag as 1 if lookbackid in this thread is -1 
      }
      // check if all threads see a valid get_local_id(0) prefix
      if (sub_group_all(flag)) {
        uint local_prefix = 0;
        // check if any thread has an inclusive prefix
        if (sub_group_any(flag == FLG_P)) {
          // we will terminate after this iteration
          done = true;
          // we want to find the highest thread with an inclusive prefix
          uint inclusive = flag == FLG_P ? get_sub_group_local_id() : 0;
          // broadcast to  all threads in the subgroup the highest thread with inclusive prefix
          uint max_inclusive = sub_group_reduce_max(inclusive);
          // load thread with highest FLG_P and higher prefixes
          if (max_inclusive <= get_sub_group_local_id()) {
            local_prefix = agg;
          }
          //local_prefix = agg & -(max_inclusive <= get_sub_group_local_id());
        // if no thread has inclusive prefix, all threads load exclusive prefix
        } else {
          // every thread looks back another partition
          local_prefix = agg;
          lookback_id = lookback_id - get_sub_group_size();
        }
        uint scanned_prefix = sub_group_scan_inclusive_add(local_prefix);

        // last thread has the full prefix, update the workgroup level exclusive prefix
        if (get_sub_group_local_id() == get_sub_group_size() - 1) {
          exclusive_prefix += scanned_prefix;
        }
      }
    }

    // finally last thread in subgroup updates this workgroup's prefix/flag
    if (get_sub_group_local_id() == get_sub_group_size() - 1) {
      atomic_store_explicit(&prefix_states[part_id], (FLG_P << ANTI_MASK) | ((exclusive_prefix + scratch[get_local_size(0) - 1]) & MASK), memory_order_relaxed);
    }
    //sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    // dont need subgroup barrier because exlcusive_prefix is local to the workgroup and is always accessed by the same thread of the workgroup everytime 
  }
  }else{
  // lookback phase
  if (part_id != 0 && get_local_id(0) == 0) {
    int lookback_id = part_id - 1;
    // spin and lookback until full prefix is set
    while (lookback_id >= 0) {
      uint flagg = atomic_load_explicit(&prefix_states[lookback_id], memory_order_relaxed);     
      uint agg = flagg & 0x3FFFFFFF;  
      uint flag = flagg >> ANTI_MASK;

      if (flag == FLG_P) {
        exclusive_prefix += agg; 
        break;
      } else if (flag == FLG_A) {
        exclusive_prefix += agg;
        lookback_id -= 1;
      }
    }
    atomic_store_explicit(&prefix_states[part_id], (FLG_P << ANTI_MASK) | ((exclusive_prefix + scratch[get_local_size(0) - 1]) & MASK), memory_order_relaxed);
  }
  }
  // ensure all threads in the block see exclusive_prefix  
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  uint total_exclusive_prefix = exclusive_prefix;
  // scratch contains an inclusive prefix per thread, so the exclusive prefix is grabbed from 
  // the previous thread's scratch location
  if (get_local_id(0) != 0) {
    total_exclusive_prefix += scratch[get_local_id(0) - 1];

  }

  // values is an int4 but you can d0 this still
  for (uint i = 0; i < BATCH_SIZE; i++) {
    out[my_id + i] = values[i] + total_exclusive_prefix;
  }

}












