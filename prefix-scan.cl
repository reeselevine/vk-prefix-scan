#define BATCH_SIZE 8

#define FLG_A 1
#define FLG_P 2

typedef struct PrefixState {
  uint agg;
  atomic_uint flag;
} PrefixState;

__kernel void prefix_scan(
  __global uint *in, 
  __global uint *out,
  __local uint *scratch,
  __global PrefixState *prefix_states,
  __global atomic_uint *partition) {
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
  }

  __local uint exclusive_prefix;

  // one thread in each block updates the aggregate/flag (use first thread to avoid extra workgroup barrier)
  if (get_local_id(0) == 0) {
    uint flag = FLG_A;
    // first block does not need to look back
    if (part_id == 0) {
      flag = FLG_P;
    }
    prefix_states[part_id].agg = scratch[get_local_size(0) - 1];
    atomic_store_explicit(&prefix_states[part_id].flag, flag, memory_order_release);

    // might as well initialize exclusive prefix here too
    exclusive_prefix = 0;
  }
  
  // lookback phase, for now not decoupled to test above code
  if (part_id != 0 && get_local_id(0) == 0) {
    uint lookback_id = part_id - 1;
    // spin until inclusive prefix is set
    while (atomic_load_explicit(&prefix_states[lookback_id].flag, memory_order_acquire) != FLG_P);
    exclusive_prefix = prefix_states[lookback_id].agg;
    prefix_states[part_id].agg = exclusive_prefix + scratch[get_local_size(0) - 1];
    atomic_store_explicit(&prefix_states[part_id].flag, FLG_P, memory_order_release);
  }

  // ensure all threads in the block see exclusive_prefix  
  work_group_barrier(CLK_LOCAL_MEM_FENCE);

  uint total_exclusive_prefix = exclusive_prefix;
  // scratch contains an inclusive prefix per thread, so the exclusive prefix is grabbed from 
  // the previous thread's scratch location
  if (get_local_id(0) != 0) {
    total_exclusive_prefix += scratch[get_local_id(0)- 1];
  }

  // store final prefix back to memory
  for (uint i = 0; i < BATCH_SIZE; i++) {
    out[my_id + i] = values[i] + total_exclusive_prefix;
  }


 








}
