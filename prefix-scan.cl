#define BATCH_SIZE 8

#define FLG_A 1
#define FLG_P 2

typedef struct PrefixState {
  uint agg;
  uint inclusive_prefix;
  atomic_uint flag;
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
  }
  //ensure that all threads in the block see the updated part_id
  work_group_barrier(CLK_LOCAL_MEM_FENCE);


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

  __local uint exclusive_prefix;

  // one thread in each block updates the aggregate/flag
  if (get_local_id(0) == 0) {
    uint flag = FLG_A;
    prefix_states[part_id].agg = scratch[get_local_size(0) - 1];
    // first block does not need to look back
    if (part_id == 0) {
      flag = FLG_P;
      prefix_states[part_id].inclusive_prefix = scratch[get_local_size(0) - 1];
    }
    atomic_store_explicit(&prefix_states[part_id].flag, flag, memory_order_release);

    // might as well initialize exclusive prefix here too
    exclusive_prefix = 0;
  }

  // lookback phase (parallelized), all threads in first subgroup participate
  if (part_id != 0 && get_sub_group_id() == 0) {
    // ensure all threads in the subgroup see exclusive_prefix initialized
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    uint lookback_id = calc_lookback_id(part_id, get_sub_group_size() - get_sub_group_local_id());
    bool done = false;
    // spin and lookback until full prefix is set
    while (!done) {
      uint flag = atomic_load_explicit(&prefix_states[lookback_id].flag, memory_order_acquire);
      // check if all threads see a valid prefix
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
            local_prefix = prefix_states[lookback_id].agg;
          }
        // if no thread has inclusive prefix, all threads load exclusive prefix
        } else {
          // every thread looks back another partition
          local_prefix = prefix_states[lookback_id].agg;
          lookback_id = calc_lookback_id(lookback_id, get_sub_group_size());
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
      //debug[0] = exclusive_prefix + scratch[get_local_size(0) - 1];
      prefix_states[part_id].inclusive_prefix = exclusive_prefix + scratch[get_local_size(0) - 1];
      atomic_store_explicit(&prefix_states[part_id].flag, FLG_P, memory_order_release);
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



  uint total_exclusive_prefix = exclusive_prefix;

  // scratch contains an inclusive prefix per thread, so the exclusive prefix is grabbed from 
  // the previous thread's scratch location
  if (get_local_id(0) != 0) {
    //debug[get_local_id(0) - 1] = total_exclusive_prefix;
    total_exclusive_prefix += scratch[get_local_id(0) - 1];
    
  }


  // store final prefix back to memory
  for (uint i = 0; i < BATCH_SIZE; i++) {
    out[my_id + i] = values[i] + total_exclusive_prefix;
  }


 








}
