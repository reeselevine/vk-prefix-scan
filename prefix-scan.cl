//#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#define BATCH_SIZE 8

#define FLG_A 1U
#define FLG_P 2U
#define MASK ~(3 << 30)
#define ANTI_MASK 30

typedef struct PrefixState {
  uint inclusive_prefix;
  atomic_uint flagg;
} PrefixState;

uint calc_lookback_id(uint part_id, uint lookback_amt) {
  if (lookback_amt > part_id) {
    return 0;
  } else {
    return part_id - lookback_amt;
  }
}

__kernel void prefix_scan(
  __global uint *in, 
  __global uint *out,
  __local uint *scratch,
  __global PrefixState *prefix_states,
  __global atomic_uint *partition,
  __global uint * debug) {
  __local uint part_id;
  // first thread in each block gets its part by atomically incrementing the global partition variable.
  if (get_local_id(0) == 0) {
    part_id = atomic_fetch_add(partition, 1);
    //part_id = get_group_id(0);
  }
  //ensure that all threads in the block see the updated part_id
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  __local uint exclusive_prefix;
  __local uint temp;
  __local uint inclusive_scan;

  int scan_type;
  scan_type = 'b';




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
      work_group_barrier(CLK_LOCAL_MEM_FENCE);
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
      work_group_barrier(CLK_LOCAL_MEM_FENCE);
      break;
    }
  default:
    {
      return;
    }  
  }

  // one thread in each block updates the aggregate/flag
  if (get_local_id(0) == 0) {
    // bit packing the first most significant 2 bits with FLG_A
    // TODO maybe: we dont need an atomic here because only 1 memory per workgroup and 1 thread touching it
    
    atomic_store_explicit(&prefix_states[part_id].flagg, (FLG_A << ANTI_MASK) | (scratch[get_local_size(0) - 1] & MASK), memory_order_relaxed);
    
    // first block does not need to look back
    if (part_id == 0) {
      prefix_states[part_id].inclusive_prefix = scratch[get_local_size(0) - 1];
      atomic_store_explicit(&prefix_states[part_id].flagg, (FLG_P << ANTI_MASK) | (scratch[get_local_size(0) - 1] & MASK), memory_order_relaxed);
      //debug[0] = ((FLG_P << 30) | (scratch[get_local_size(0) - 1] & MASK));
    }
    // might as well initialize exclusive prefix here too
    exclusive_prefix = 0;
  }
  

  //work_group_barrier(CLK_LOCAL_MEM_FENCE);
  // lookback phase (parallelized), all threads in first subgroup participate
  if (part_id != 0 && get_sub_group_id() == 0) {
    // ensure all threads in the subgroup see exclusive_prefix initialized
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    uint lookback_id = calc_lookback_id(part_id, get_sub_group_size() - get_sub_group_local_id());
    bool done = false;
    // spin and lookback until full prefix is set
    while (!done) {
      uint flagg = atomic_load_explicit(&prefix_states[lookback_id].flagg, memory_order_acquire);     
      uint agg = flagg & 0x3FFFFFFF;  
      uint flag = flagg >> ANTI_MASK;
      sub_group_barrier(CLK_LOCAL_MEM_FENCE);

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
          // highest thread with inclusive prefix loads it
          if (get_sub_group_local_id() == max_inclusive) {
            local_prefix = prefix_states[lookback_id].inclusive_prefix;

          // threads with higher ids load exclusive prefix
          } else if (max_inclusive < get_sub_group_local_id()) {
            local_prefix = agg;
          }
        // if no thread has inclusive prefix, all threads load exclusive prefix
        } else {
          // every thread looks back another partition
          local_prefix = agg;
          lookback_id = calc_lookback_id(lookback_id, get_sub_group_size());
        }
        uint scanned_prefix = sub_group_scan_inclusive_add(local_prefix);

        // if (part_id == 1) {

        //   debug[0] = sub_group_scan_inclusive_add(scanned_prefix);
        // }
        // last thread has the full prefix, update the workgroup level exclusive prefix
        if (get_sub_group_local_id() == get_sub_group_size() - 1) {
          exclusive_prefix += scanned_prefix;
        }
      }
    }

    // finally last thread in subgroup updates this workgroup's prefix/flag
    if (get_sub_group_local_id() == get_sub_group_size() - 1) {
      prefix_states[part_id].inclusive_prefix = exclusive_prefix + scratch[get_local_size(0) - 1];

      // this part_id no longer needs agg so just a flag is necesarry
      atomic_store_explicit(&prefix_states[part_id].flagg, FLG_P << ANTI_MASK, memory_order_release);
    }
  }

  // ensure all threads in the block see exclusive_prefix  
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

    // if (get_local_id(0) == 0 && part_id > 254) {
    //   debug[0] = exclusive_prefix;
    // }  


  uint total_exclusive_prefix = exclusive_prefix;
  // scratch contains an inclusive prefix per thread, so the exclusive prefix is grabbed from 
  // the previous thread's scratch location
  if (get_local_id(0) != 0) {
    total_exclusive_prefix += scratch[get_local_id(0) - 1];

  }

  // if (get_local_id(0) - 1 && part_id == 1) {
  //     debug[0] = exclusive_prefix;
  // }  


  for (uint i = 0; i < BATCH_SIZE; i++) {
    out[my_id + i] = values[i] + total_exclusive_prefix;
  }



}