// Deterministic radix-select top-k: for each row b,
// select the kTopK largest scores in [row_starts[b], row_starts[b] + lengths[b])
// and write their indices relative to row_starts[b] in ascending order.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <bit>

namespace sglang {

namespace fast_topk_detail {

constexpr uint32_t kThreadsPerBlock = 1024;
constexpr int kShortUnorderedMaxLength = 16 * 1024;
constexpr int kShortDirectMaxLength = 4 * 1024;
// One 8K-entry candidate buffer covers the normal long-row cutoff set. The
// threshold bin is refined in-place, while an explicit overflow bit sends
// concentrated/adversarial rows through the exact full-row fallback.
constexpr size_t kSmemBytes = 8 * 1024 * sizeof(uint32_t);  // 32KB

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, kTopK]
  const int32_t* __restrict__ lengths;     // [B]
  int64_t input_stride;
};

SGL_DEVICE auto convert_to_uint8(float x) -> uint8_t {
  const __half h = __float2half_rn(x);
  const uint16_t bits = __half_as_ushort(h);
  const uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

SGL_DEVICE auto convert_to_uint32(float x) -> uint32_t {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// When length <= kTopK, write the indices directly.
template <int kTopK>
SGL_DEVICE void naive_topk(int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < kTopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// Stable top-k classification for the candidate collector.
struct StableTopKFlags {
  bool gt;
  bool eq;
};

template <bool kCoarse>
SGL_DEVICE StableTopKFlags classify_stable_topk(
    float value, int coarse_threshold, uint32_t pivot_key, int tie_take) {
  const int coarse = static_cast<int>(convert_to_uint8(value));
  if constexpr (kCoarse) {
    return {coarse > coarse_threshold, false};
  }
  if (coarse > coarse_threshold) {
    return {true, false};
  }
  if (coarse == coarse_threshold) {
    const uint32_t key = convert_to_uint32(value);
    return {key > pivot_key, tie_take > 0 && key == pivot_key};
  }
  return {false, false};
}

SGL_DEVICE int warp_exclusive_sum(int value) {
  const int lane = threadIdx.x & 31;
  int inclusive = value;
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset) {
      inclusive += other;
    }
  }
  return inclusive - value;
}

SGL_DEVICE int warp_total_sum(int value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

// Stable collection using the exact pivot. Each warp stores its selected
// logical indices in order, so the final emission touches only the selected set.
// A warp-local capacity overflow falls back to the same exact two-pass scan and
// never clips or reorders the result.
template <int kTopK, bool kCoarse>
SGL_DEVICE void stable_collect_candidate(
    const float* __restrict__ input,
    int* __restrict__ index,
    int row_start,
    int length,
    int coarse_threshold,
    uint32_t pivot_key,
    int tie_take,
    int* __restrict__ stable_storage) {
  constexpr int kWarps = kThreadsPerBlock / 32;
  constexpr int kItemsPerThread = 8;
  constexpr int kCandidateCapacity = kSmemBytes / sizeof(int);
  constexpr int kWarpCapacity = kCandidateCapacity / kWarps;
  __shared__ int s_gt_counts[kWarps];
  __shared__ int s_eq_counts[kWarps];
  __shared__ int s_gt_bases[kWarps];
  __shared__ int s_eq_bases[kWarps];
  __shared__ int s_output_count;
  __shared__ int s_overflow;

  const int tx = threadIdx.x;
  const int warp = tx >> 5;
  const int lane = tx & 31;
  const int chunk = (length + kWarps - 1) / kWarps;
  const int begin = min(warp * chunk, length);
  const int end = min(begin + chunk, length);
  const unsigned full_mask = 0xffffffffu;

  if (tx == 0) {
    s_overflow = 0;
  }
  __syncthreads();

  // One row-order pass. Each warp writes its selected items into a fixed
  // logical segment; no atomic arrival order enters the stored sequence.
  int stored_total = 0;
  int lane_gt_total = 0;
  int lane_eq_total = 0;
  for (int base = begin; base < end; base += 32 * kItemsPerThread) {
    bool selected_gt[kItemsPerThread];
    bool selected_eq[kItemsPerThread];
    int local_total = 0;
    int local_gt = 0;
    int local_eq = 0;
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
      const int i = base + lane * kItemsPerThread + item;
      const bool valid = i < end;
      const StableTopKFlags flags =
          valid ? classify_stable_topk<kCoarse>(
                      input[row_start + i], coarse_threshold, pivot_key, tie_take)
                : StableTopKFlags{false, false};
      selected_gt[item] = flags.gt;
      selected_eq[item] = flags.eq;
      local_gt += flags.gt ? 1 : 0;
      local_eq += flags.eq ? 1 : 0;
    }
    local_total = local_gt + local_eq;
    const int prefix = warp_exclusive_sum(local_total);
    const int group_total = warp_total_sum(local_total);
    int rank = stored_total + prefix;
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
      if (selected_gt[item] || selected_eq[item]) {
        if (rank < kWarpCapacity) {
          stable_storage[warp * kWarpCapacity + rank] =
              base + lane * kItemsPerThread + item;
        } else {
          ::atomicExch(&s_overflow, 1);
        }
        ++rank;
      }
    }
    stored_total += group_total;
    lane_gt_total += local_gt;
    lane_eq_total += local_eq;
  }
  const int warp_gt_total = warp_total_sum(lane_gt_total);
  const int warp_eq_total = warp_total_sum(lane_eq_total);
  if (lane == 0) {
    s_gt_counts[warp] = warp_gt_total;
    s_eq_counts[warp] = warp_eq_total;
  }
  __syncthreads();

  if (tx == 0) {
    int gt_running = 0;
    int eq_running = 0;
    for (int w = 0; w < kWarps; ++w) {
      s_gt_bases[w] = gt_running;
      s_eq_bases[w] = eq_running;
      gt_running += s_gt_counts[w];
      eq_running += s_eq_counts[w];
    }
    s_output_count = gt_running + min(eq_running, tie_take);
  }
  __syncthreads();

  if (s_overflow) {
    // Exact fallback count pass. This path is intentionally uncommon; it is
    // required for rows whose selected set is concentrated in one warp.
    lane_gt_total = 0;
    lane_eq_total = 0;
    for (int base = begin; base < end; base += 32 * kItemsPerThread) {
      int local_gt = 0;
      int local_eq = 0;
#pragma unroll
      for (int item = 0; item < kItemsPerThread; ++item) {
        const int i = base + lane * kItemsPerThread + item;
        if (i < end) {
          const StableTopKFlags flags =
              classify_stable_topk<kCoarse>(
                  input[row_start + i], coarse_threshold, pivot_key, tie_take);
          local_gt += flags.gt ? 1 : 0;
          local_eq += flags.eq ? 1 : 0;
        }
      }
      lane_gt_total += local_gt;
      lane_eq_total += local_eq;
    }
    const int warp_gt_total = warp_total_sum(lane_gt_total);
    const int warp_eq_total = warp_total_sum(lane_eq_total);
    if (lane == 0) {
      s_gt_counts[warp] = warp_gt_total;
      s_eq_counts[warp] = warp_eq_total;
    }
    __syncthreads();
    if (tx == 0) {
      int gt_running = 0;
      int eq_running = 0;
      for (int w = 0; w < kWarps; ++w) {
        s_gt_bases[w] = gt_running;
        s_eq_bases[w] = eq_running;
        gt_running += s_gt_counts[w];
        eq_running += s_eq_counts[w];
      }
      s_output_count = gt_running + min(eq_running, tie_take);
    }
    __syncthreads();

    int seen_gt = 0;
    int seen_eq = 0;
    for (int base = begin; base < end; base += 32 * kItemsPerThread) {
      bool selected_gt[kItemsPerThread];
      bool selected_eq[kItemsPerThread];
      int local_gt = 0;
      int local_eq = 0;
#pragma unroll
      for (int item = 0; item < kItemsPerThread; ++item) {
        const int i = base + lane * kItemsPerThread + item;
        const bool valid = i < end;
        const StableTopKFlags flags =
            valid ? classify_stable_topk<kCoarse>(
                        input[row_start + i], coarse_threshold, pivot_key, tie_take)
                  : StableTopKFlags{false, false};
        selected_gt[item] = flags.gt;
        selected_eq[item] = flags.eq;
        local_gt += flags.gt ? 1 : 0;
        local_eq += flags.eq ? 1 : 0;
      }
      const int prefix_gt = warp_exclusive_sum(local_gt);
      const int prefix_eq = warp_exclusive_sum(local_eq);
      const int group_gt = warp_total_sum(local_gt);
      const int group_eq = warp_total_sum(local_eq);
      int gt_rank = prefix_gt;
      int eq_rank = prefix_eq;
#pragma unroll
      for (int item = 0; item < kItemsPerThread; ++item) {
        const int i = base + lane * kItemsPerThread + item;
        const int gt_before = s_gt_bases[warp] + seen_gt + gt_rank;
        const int eq_before = s_eq_bases[warp] + seen_eq + eq_rank;
        const bool selected =
            selected_gt[item] || (selected_eq[item] && eq_before < tie_take);
        const int output_pos = gt_before + min(eq_before, tie_take);
        if (selected && output_pos < kTopK) {
          index[output_pos] = i;
        }
        gt_rank += selected_gt[item] ? 1 : 0;
        eq_rank += selected_eq[item] ? 1 : 0;
      }
      seen_gt += group_gt;
      seen_eq += group_eq;
    }
  } else if (s_eq_counts[warp] == 0) {
    // The overwhelmingly common random case has no exact pivot duplicates.
    // Exact ties may still have been selected by an earlier warp, so include
    // that stable prefix in this warp's output base.
    const int total = s_gt_counts[warp];
    const int output_base =
        s_gt_bases[warp] + min(s_eq_bases[warp], tie_take);
    for (int p = lane; p < total; p += 32) {
      index[output_base + p] =
          stable_storage[warp * kWarpCapacity + p];
    }
  } else {
    // Exact tie handling for the bounded candidate set.
    const int total = s_gt_counts[warp] + s_eq_counts[warp];
    int seen_gt = 0;
    int seen_eq = 0;
    for (int base = 0; base < total; base += 32 * kItemsPerThread) {
      bool selected_gt[kItemsPerThread];
      bool selected_eq[kItemsPerThread];
      int local_gt = 0;
      int local_eq = 0;
#pragma unroll
      for (int item = 0; item < kItemsPerThread; ++item) {
        const int p = base + lane * kItemsPerThread + item;
        const bool valid = p < total;
        const int i = valid ? stable_storage[warp * kWarpCapacity + p] : 0;
        const StableTopKFlags flags =
            valid ? classify_stable_topk<kCoarse>(
                        input[row_start + i], coarse_threshold, pivot_key, tie_take)
                  : StableTopKFlags{false, false};
        selected_gt[item] = flags.gt;
        selected_eq[item] = flags.eq;
        local_gt += flags.gt ? 1 : 0;
        local_eq += flags.eq ? 1 : 0;
      }
      const int prefix_gt = warp_exclusive_sum(local_gt);
      const int prefix_eq = warp_exclusive_sum(local_eq);
      const int group_gt = warp_total_sum(local_gt);
      const int group_eq = warp_total_sum(local_eq);
      int gt_rank = prefix_gt;
      int eq_rank = prefix_eq;
#pragma unroll
      for (int item = 0; item < kItemsPerThread; ++item) {
        const int p = base + lane * kItemsPerThread + item;
        const bool valid = p < total;
        const int i = valid ? stable_storage[warp * kWarpCapacity + p] : 0;
        const int gt_before = s_gt_bases[warp] + seen_gt + gt_rank;
        const int eq_before = s_eq_bases[warp] + seen_eq + eq_rank;
        const bool selected =
            valid && (selected_gt[item] ||
                      (selected_eq[item] && eq_before < tie_take));
        const int output_pos = gt_before + min(eq_before, tie_take);
        if (selected && output_pos < kTopK) {
          index[output_pos] = i;
        }
        gt_rank += selected_gt[item] ? 1 : 0;
        eq_rank += selected_eq[item] ? 1 : 0;
      }
      seen_gt += group_gt;
      seen_eq += group_eq;
    }
  }
  __syncthreads();
  for (int i = tx + s_output_count; i < kTopK; i += kThreadsPerBlock) {
    index[i] = -1;
  }
  __syncthreads();
}
// Recompute the refinement histograms over the full row when the threshold
// coarse bin does not fit in the bounded candidate buffers.
template <int kTopK>
SGL_DEVICE void full_row_refine_pivot(
    const float* __restrict__ input,
    int row_start,
    int length,
    int coarse_threshold,
    int remaining,
    int (*histogram_buf)[256 + 128],
    int* threshold_bin,
    uint8_t* threshold_bytes,
    uint32_t* pivot_key,
    int* tie_take) {
  constexpr int kRadix = 256;
  const int tx = threadIdx.x;

#pragma unroll
  for (int round = 0; round < 4; ++round) {
    if (tx < kRadix + 1) {
      histogram_buf[0][tx] = 0;
    }
    __syncthreads();

    const int shift = 24 - round * 8;
    for (int i = tx; i < length; i += kThreadsPerBlock) {
      const float value = input[row_start + i];
      if (static_cast<int>(convert_to_uint8(value)) != coarse_threshold) {
        continue;
      }
      const uint32_t key = convert_to_uint32(value);
      bool prefix_match = true;
#pragma unroll
      for (int previous = 0; previous < round; ++previous) {
        prefix_match =
            prefix_match &&
            (((key >> (24 - previous * 8)) & 0xFFu) == threshold_bytes[previous]);
      }
      if (prefix_match) {
        const int bin = static_cast<int>((key >> shift) & 0xFFu);
        atomicAdd(&histogram_buf[0][bin], 1);
      }
    }
    __syncthreads();

#pragma unroll
    for (int scan_round = 0; scan_round < 8; ++scan_round) {
      if (tx < kRadix) {
        const int offset = 1 << scan_round;
        const int source = scan_round & 1;
        int value = histogram_buf[source][tx];
        if (tx < kRadix - offset) {
          value += histogram_buf[source][tx + offset];
        }
        histogram_buf[source ^ 1][tx] = value;
      }
      __syncthreads();
    }

    if (tx == 0) {
      *threshold_bin = 0;
    }
    __syncthreads();
    if (tx < kRadix &&
        histogram_buf[0][tx] > remaining &&
        histogram_buf[0][tx + 1] <= remaining) {
      *threshold_bin = tx;
    }
    __syncthreads();

    const int threshold = *threshold_bin;
    remaining -= histogram_buf[0][threshold + 1];
    if (tx == 0) {
      threshold_bytes[round] = static_cast<uint8_t>(threshold);
    }
    __syncthreads();

    if (remaining == 0 || round == 3) {
      if (tx == 0) {
        uint32_t pivot = 0;
        for (int byte = 0; byte < 4; ++byte) {
          const uint32_t value =
              byte <= round ? threshold_bytes[byte] : 0xFFu;
          pivot |= value << (24 - byte * 8);
        }
        *pivot_key = pivot;
        *tie_take = remaining;
      }
      __syncthreads();
      return;
    }
  }
}

template <int kTopK, bool kCoarse>
SGL_DEVICE void stable_collect_short_direct(
    const float* __restrict__ input,
    int* __restrict__ index,
    int row_start,
    int length,
    int coarse_threshold,
    uint32_t pivot_key,
    int tie_take) {
  constexpr int kWarps = kThreadsPerBlock / 32;
  constexpr int kMaxItems = kShortDirectMaxLength / kThreadsPerBlock;
  __shared__ int s_gt_totals[kWarps];
  __shared__ int s_eq_totals[kWarps];
  __shared__ int s_gt_bases[kWarps];
  __shared__ int s_eq_bases[kWarps];
  const int tx = threadIdx.x;
  const int warp = tx >> 5;
  const int lane = tx & 31;
  const int items_per_thread =
      (length + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const int begin = tx * items_per_thread;
  bool gt_flags[kMaxItems];
  bool eq_flags[kMaxItems];
  int local_gt = 0;
  int local_eq = 0;

#pragma unroll
  for (int item = 0; item < kMaxItems; ++item) {
    const int relative = begin + item;
    const bool valid = item < items_per_thread && relative < length;
    bool gt = false;
    bool eq = false;
    if (valid) {
      const float value = input[row_start + relative];
      if constexpr (kCoarse) {
        gt = static_cast<int>(convert_to_uint8(value)) > coarse_threshold;
      } else {
        const uint32_t key = convert_to_uint32(value);
        gt = key > pivot_key;
        eq = tie_take > 0 && key == pivot_key;
      }
    }
    gt_flags[item] = gt;
    eq_flags[item] = eq;
    local_gt += gt ? 1 : 0;
    local_eq += eq ? 1 : 0;
  }

  const unsigned full_mask = 0xffffffffu;
  int inclusive_gt = local_gt;
  int inclusive_eq = local_eq;
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int other_gt = __shfl_up_sync(full_mask, inclusive_gt, offset);
    const int other_eq = __shfl_up_sync(full_mask, inclusive_eq, offset);
    if (lane >= offset) {
      inclusive_gt += other_gt;
      inclusive_eq += other_eq;
    }
  }
  const int warp_gt_prefix = inclusive_gt - local_gt;
  const int warp_eq_prefix = inclusive_eq - local_eq;
  if (lane == 31) {
    s_gt_totals[warp] = inclusive_gt;
    s_eq_totals[warp] = inclusive_eq;
  }
  __syncthreads();

  if (warp == 0) {
    const int gt_total = s_gt_totals[lane];
    const int eq_total = s_eq_totals[lane];
    int gt_inclusive = gt_total;
    int eq_inclusive = eq_total;
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other_gt = __shfl_up_sync(full_mask, gt_inclusive, offset);
      const int other_eq = __shfl_up_sync(full_mask, eq_inclusive, offset);
      if (lane >= offset) {
        gt_inclusive += other_gt;
        eq_inclusive += other_eq;
      }
    }
    s_gt_bases[lane] = gt_inclusive - gt_total;
    s_eq_bases[lane] = eq_inclusive - eq_total;
  }
  __syncthreads();

  int gt_before = s_gt_bases[warp] + warp_gt_prefix;
  int eq_before = s_eq_bases[warp] + warp_eq_prefix;
#pragma unroll
  for (int item = 0; item < kMaxItems; ++item) {
    const bool gt = gt_flags[item];
    const bool eq = eq_flags[item];
    const bool selected =
        item < items_per_thread && (gt || (eq && eq_before < tie_take));
    const int output_pos = gt_before + min(eq_before, tie_take);
    if (selected && output_pos < kTopK) {
      index[output_pos] = begin + item;
    }
    gt_before += gt ? 1 : 0;
    eq_before += eq ? 1 : 0;
  }
  __syncthreads();
}

// Short/medium-row pivot selection. Candidate arrival order is irrelevant to
// integer radix histograms, so retain only the coarse-threshold bin with one
// atomic counter and avoid stable in-place candidate refinement. Canonical
// ordering is applied exactly once by stable_collect_candidate after the pivot
// is known. Overflow remains exact via the full-row fallback.
template <int kTopK>
SGL_DEVICE void radix_select_short_unordered(
    const float* __restrict__ input,
    int* __restrict__ index,
    int row_start,
    int length,
    int coarse_threshold,
    int remaining,
    int (*histogram_buf)[256 + 128],
    int* threshold_bin,
    uint8_t* threshold_bytes,
    uint32_t* pivot_key,
    int* tie_take,
    int* __restrict__ workspace) {
  constexpr int kRadix = 256;
  constexpr int kCandidateCapacity = kSmemBytes / sizeof(int);
  __shared__ int s_candidate_count;
  __shared__ int s_overflow;
  const int tx = threadIdx.x;

  if (remaining == 0) {
    if (length <= kShortDirectMaxLength) {
      stable_collect_short_direct<kTopK, true>(
          input, index, row_start, length, coarse_threshold, 0, 0);
    } else {
      stable_collect_candidate<kTopK, true>(
          input, index, row_start, length, coarse_threshold, 0, 0, workspace);
    }
    return;
  }

  if (tx < kRadix + 1) {
    histogram_buf[0][tx] = 0;
  }
  if (tx == 0) {
    s_candidate_count = 0;
    s_overflow = 0;
  }
  __syncthreads();

  for (int i = tx; i < length; i += kThreadsPerBlock) {
    const float value = input[row_start + i];
    if (static_cast<int>(convert_to_uint8(value)) != coarse_threshold) {
      continue;
    }
    const int position = ::atomicAdd(&s_candidate_count, 1);
    if (position < kCandidateCapacity) {
      workspace[position] = i;
    } else {
      ::atomicExch(&s_overflow, 1);
    }
    const uint32_t key = convert_to_uint32(value);
    ::atomicAdd(&histogram_buf[0][(key >> 24) & 0xFFu], 1);
  }
  __syncthreads();

  if (s_overflow) {
    full_row_refine_pivot<kTopK>(
        input,
        row_start,
        length,
        coarse_threshold,
        remaining,
        histogram_buf,
        threshold_bin,
        threshold_bytes,
        pivot_key,
        tie_take);
    if (length <= kShortDirectMaxLength) {
      stable_collect_short_direct<kTopK, false>(
          input,
          index,
          row_start,
          length,
          coarse_threshold,
          *pivot_key,
          *tie_take);
    } else {
      stable_collect_candidate<kTopK, false>(
          input,
          index,
          row_start,
          length,
          coarse_threshold,
          *pivot_key,
          *tie_take,
          workspace);
    }
    return;
  }

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    if (round > 0) {
      if (tx < kRadix + 1) {
        histogram_buf[0][tx] = 0;
      }
      __syncthreads();
      const int shift = 24 - round * 8;
      const int count = s_candidate_count;
      for (int p = tx; p < count; p += kThreadsPerBlock) {
        const uint32_t key =
            convert_to_uint32(input[row_start + workspace[p]]);
        bool prefix_match = true;
#pragma unroll
        for (int previous = 0; previous < round; ++previous) {
          prefix_match =
              prefix_match &&
              (((key >> (24 - previous * 8)) & 0xFFu) ==
               threshold_bytes[previous]);
        }
        if (prefix_match) {
          ::atomicAdd(&histogram_buf[0][(key >> shift) & 0xFFu], 1);
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (int scan_round = 0; scan_round < 8; ++scan_round) {
      if (tx < kRadix) {
        const int offset = 1 << scan_round;
        const int source = scan_round & 1;
        int value = histogram_buf[source][tx];
        if (tx < kRadix - offset) {
          value += histogram_buf[source][tx + offset];
        }
        histogram_buf[source ^ 1][tx] = value;
      }
      __syncthreads();
    }

    if (tx == 0) {
      *threshold_bin = 0;
    }
    __syncthreads();
    if (tx < kRadix &&
        histogram_buf[0][tx] > remaining &&
        histogram_buf[0][tx + 1] <= remaining) {
      *threshold_bin = tx;
    }
    __syncthreads();

    const int threshold = *threshold_bin;
    remaining -= histogram_buf[0][threshold + 1];
    if (tx == 0) {
      threshold_bytes[round] = static_cast<uint8_t>(threshold);
    }
    __syncthreads();

    if (remaining == 0 || round == 3) {
      if (tx == 0) {
        uint32_t pivot = 0;
        for (int byte = 0; byte < 4; ++byte) {
          uint32_t value = byte <= round ? threshold_bytes[byte] : 0u;
          if (byte > round && remaining == 0) {
            value = 0xFFu;
          }
          pivot |= value << (24 - byte * 8);
        }
        *pivot_key = pivot;
        *tie_take = remaining;
      }
      __syncthreads();
      break;
    }
  }

  if (length <= kShortDirectMaxLength) {
    stable_collect_short_direct<kTopK, false>(
        input,
        index,
        row_start,
        length,
        coarse_threshold,
        *pivot_key,
        *tie_take);
  } else {
    stable_collect_candidate<kTopK, false>(
        input,
        index,
        row_start,
        length,
        coarse_threshold,
        *pivot_key,
        *tie_take,
        workspace);
  }
}

// Stable compaction over a row-order candidate set. The candidate set is
// partitioned by contiguous warp slices, so this avoids rescanning padding or
// the full long row after radix selection.
template <int kTopK, bool kCoarse>
SGL_DEVICE void stable_collect_buffer(
    const float* __restrict__ input,
    int* __restrict__ index,
    int row_start,
    int coarse_threshold,
    uint32_t pivot_key,
    int tie_take,
    int* __restrict__ candidate,
    const int* __restrict__ candidate_counts) {
  constexpr int kWarps = kThreadsPerBlock / 32;
  constexpr int kCandidateCapacity = kSmemBytes / sizeof(int);
  constexpr int kWarpCapacity = kCandidateCapacity / kWarps;
  __shared__ int s_gt_counts[kWarps];
  __shared__ int s_eq_counts[kWarps];
  __shared__ int s_gt_bases[kWarps];
  __shared__ int s_eq_bases[kWarps];
  __shared__ int s_output_count;

  const int tx = threadIdx.x;
  const int warp = tx >> 5;
  const int lane = tx & 31;
  const unsigned full_mask = 0xffffffffu;
  const int count = candidate_counts[warp];

  int local_gt = 0;
  int local_eq = 0;
  for (int base = 0; base < count; base += 32) {
    const int p = base + lane;
    const bool valid = p < count;
    const int relative = valid ? candidate[warp * kWarpCapacity + p] : 0;
    const StableTopKFlags flags =
        valid ? classify_stable_topk<kCoarse>(
                    input[row_start + relative], coarse_threshold, pivot_key, tie_take)
              : StableTopKFlags{false, false};
    const unsigned gt_mask = __ballot_sync(full_mask, flags.gt);
    const unsigned eq_mask = __ballot_sync(full_mask, flags.eq);
    if (lane == 0) {
      local_gt += __popc(gt_mask);
      local_eq += __popc(eq_mask);
    }
  }
  const int warp_gt_total = warp_total_sum(local_gt);
  const int warp_eq_total = warp_total_sum(local_eq);
  if (lane == 0) {
    s_gt_counts[warp] = warp_gt_total;
    s_eq_counts[warp] = warp_eq_total;
  }
  __syncthreads();

  if (tx == 0) {
    int gt_running = 0;
    int eq_running = 0;
    for (int w = 0; w < kWarps; ++w) {
      s_gt_bases[w] = gt_running;
      s_eq_bases[w] = eq_running;
      gt_running += s_gt_counts[w];
      eq_running += s_eq_counts[w];
    }
    s_output_count = gt_running + min(eq_running, tie_take);
  }
  __syncthreads();

  if constexpr (kCoarse) {
    for (int p = lane; p < count; p += 32) {
      index[s_gt_bases[warp] + p] =
          candidate[warp * kWarpCapacity + p];
    }
  } else {
    int seen_gt = 0;
    int seen_eq = 0;
    for (int base = 0; base < count; base += 32) {
      const int p = base + lane;
      const bool valid = p < count;
      const int relative = valid ? candidate[warp * kWarpCapacity + p] : 0;
      const StableTopKFlags flags =
          valid ? classify_stable_topk<kCoarse>(
                      input[row_start + relative], coarse_threshold, pivot_key, tie_take)
                : StableTopKFlags{false, false};
      const unsigned gt_mask = __ballot_sync(full_mask, flags.gt);
      const unsigned eq_mask = __ballot_sync(full_mask, flags.eq);
      const int gt_before =
          s_gt_bases[warp] + seen_gt +
          __popc(gt_mask & ((1u << lane) - 1u));
      const int eq_before =
          s_eq_bases[warp] + seen_eq +
          __popc(eq_mask & ((1u << lane) - 1u));
      const bool selected =
          flags.gt || (flags.eq && eq_before < tie_take);
      const int output_pos = gt_before + min(eq_before, tie_take);
      if (selected && output_pos < kTopK) {
        index[output_pos] = relative;
      }
      seen_gt += __popc(gt_mask);
      seen_eq += __popc(eq_mask);
    }
  }
  __syncthreads();
  for (int i = tx + s_output_count; i < kTopK; i += kThreadsPerBlock) {
    index[i] = -1;
  }
  __syncthreads();
}

// Exact radix selection with an order-preserving coarse candidate set. Values
// above the coarse cutoff and values in the cutoff bin are stored together in
// logical order. Only the cutoff-bin subset is refined; final compaction then
// scans that bounded candidate set rather than the full row.
template <int kTopK>
SGL_DEVICE void radix_select_topk(
    const float* __restrict__ input,
    int* __restrict__ index,
    int row_start,
    int length) {
  constexpr int kBlockSize = kThreadsPerBlock;
  constexpr int kRadix = 256;
  constexpr int kWarps = kThreadsPerBlock / 32;
  constexpr int kItemsPerThread = 4;
  constexpr int kCandidateCapacity = kSmemBytes / sizeof(int);
  constexpr int kWarpCapacity = kCandidateCapacity / kWarps;

  __shared__ int s_histogram_buf[2][kRadix + 128];
  __shared__ int s_threshold_bin_id;
  __shared__ int s_overflow;
  __shared__ int s_num_input_warp[kWarps];
  __shared__ uint8_t s_threshold_bytes[4];
  __shared__ uint32_t s_pivot_key;
  __shared__ int s_tie_take;
  auto& s_histogram = s_histogram_buf[0];
  extern __shared__ int s_input_idx[];
  const int tx = threadIdx.x;

  if (tx < kRadix + 1) {
    s_histogram_buf[0][tx] = 0;
    s_histogram_buf[1][tx] = 0;
  }
  if (tx < kWarps) {
    s_num_input_warp[tx] = 0;
  }
  if (tx == 0) {
    s_overflow = 0;
  }
  __syncthreads();

  for (int idx = tx; idx < length; idx += kBlockSize) {
    const int bin = static_cast<int>(convert_to_uint8(input[row_start + idx]));
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  auto run_cumsum = [&] {
#pragma unroll 8
    for (int round = 0; round < 8; ++round) {
      if (tx < kRadix) {
        const int offset = 1 << round;
        const int source = round & 1;
        int value = s_histogram_buf[source][tx];
        if (tx < kRadix - offset) {
          value += s_histogram_buf[source][tx + offset];
        }
        s_histogram_buf[source ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx == 0) {
    s_threshold_bin_id = 0;
  }
  __syncthreads();
  if (tx < kRadix &&
      s_histogram[tx] > kTopK &&
      s_histogram[tx + 1] <= kTopK) {
    s_threshold_bin_id = tx;
  }
  __syncthreads();
  const int coarse_threshold = s_threshold_bin_id;
  int remaining = kTopK - s_histogram[coarse_threshold + 1];

  if (length <= kShortUnorderedMaxLength) {
    radix_select_short_unordered<kTopK>(
        input,
        index,
        row_start,
        length,
        coarse_threshold,
        remaining,
        s_histogram_buf,
        &s_threshold_bin_id,
        s_threshold_bytes,
        &s_pivot_key,
        &s_tie_take,
        s_input_idx);
    return;
  }

  // Store the coarse candidate set in row order. The threshold-bin subset is
  // refined in-place in this same buffer.
  const int chunk = (length + kWarps - 1) / kWarps;
  const int warp = tx >> 5;
  const int lane = tx & 31;
  const int begin = min(warp * chunk, length);
  const int end = min(begin + chunk, length);
  int selected_total = 0;
  for (int base = begin; base < end; base += 32 * kItemsPerThread) {
    bool selected_items[kItemsPerThread];
    int local_selected = 0;
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
      const int i = base + lane * kItemsPerThread + item;
      const bool valid = i < end;
      const int bin = valid
          ? static_cast<int>(convert_to_uint8(input[row_start + i]))
          : 0;
      // If no exact refinement is needed, keep only the strict coarse tail.
      // Otherwise keep the complete coarse candidate set and refine the
      // threshold-bin portion in-place below.
      const bool selected = remaining == 0
          ? (valid && bin > coarse_threshold)
          : (valid && bin >= coarse_threshold);
      selected_items[item] = selected;
      local_selected += selected ? 1 : 0;
    }
    const int prefix = warp_exclusive_sum(local_selected);
    const int group_total = warp_total_sum(local_selected);
    int rank = selected_total + prefix;
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
      const int i = base + lane * kItemsPerThread + item;
      if (selected_items[item]) {
        if (rank < kWarpCapacity) {
          s_input_idx[warp * kWarpCapacity + rank] = i;
        } else {
          ::atomicExch(&s_overflow, 1);
        }
        ++rank;
      }
    }
    selected_total += group_total;
  }
  const int warp_selected_total = selected_total;
  if (lane == 0) {
    s_num_input_warp[warp] = warp_selected_total;
  }
  __syncthreads();

  if (s_overflow) {
    if (remaining == 0) {
      stable_collect_candidate<kTopK, true>(
          input, index, row_start, length, coarse_threshold, 0, 0, s_input_idx);
    } else {
      full_row_refine_pivot<kTopK>(
          input,
          row_start,
          length,
          coarse_threshold,
          remaining,
          s_histogram_buf,
          &s_threshold_bin_id,
          s_threshold_bytes,
          &s_pivot_key,
          &s_tie_take);
      stable_collect_candidate<kTopK, false>(
          input,
          index,
          row_start,
          length,
          coarse_threshold,
          s_pivot_key,
          s_tie_take,
          s_input_idx);
    }
    return;
  }

  if (remaining == 0) {
    stable_collect_buffer<kTopK, true>(
        input,
        index,
        row_start,
        coarse_threshold,
        0,
        0,
        s_input_idx,
        s_num_input_warp);
    return;
  }

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    if (tx < kRadix + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();
    const int shift = 24 - round * 8;
    const int count = s_num_input_warp[warp];
    for (int p = lane; p < count; p += 32) {
      const int idx = s_input_idx[warp * kWarpCapacity + p];
      const uint32_t key = convert_to_uint32(input[row_start + idx]);
      if (static_cast<int>(convert_to_uint8(input[row_start + idx])) ==
          coarse_threshold) {
        bool prefix_match = true;
#pragma unroll
        for (int previous = 0; previous < round; ++previous) {
          prefix_match =
              prefix_match &&
              (((key >> (24 - previous * 8)) & 0xFFu) ==
               s_threshold_bytes[previous]);
        }
        if (!prefix_match) {
          continue;
        }
        ::atomicAdd(&s_histogram[(key >> shift) & 0xFFu], 1);
      }
    }
    __syncthreads();
    run_cumsum();
    if (tx == 0) {
      s_threshold_bin_id = 0;
    }
    __syncthreads();
    if (tx < kRadix &&
        s_histogram[tx] > remaining &&
        s_histogram[tx + 1] <= remaining) {
      s_threshold_bin_id = tx;
    }
    __syncthreads();
    const int threshold = s_threshold_bin_id;
    remaining -= s_histogram[threshold + 1];
    if (tx == 0) {
      s_threshold_bytes[round] = static_cast<uint8_t>(threshold);
    }
    __syncthreads();

    if (remaining == 0 || round == 3) {
      if (tx == 0) {
        uint32_t pivot = 0;
        for (int byte = 0; byte < 4; ++byte) {
          uint32_t value = byte <= round ? s_threshold_bytes[byte] : 0u;
          if (byte > round && remaining == 0) {
            value = 0xFFu;
          }
          pivot |= value << (24 - byte * 8);
        }
        s_pivot_key = pivot;
        s_tie_take = remaining;
      }
      __syncthreads();
      break;
    }

    int local_count = 0;
    for (int base = 0; base < count; base += 32) {
      const int p = base + lane;
      const bool valid = p < count;
      const int idx = valid ? s_input_idx[warp * kWarpCapacity + p] : 0;
      const uint32_t key = valid ? convert_to_uint32(input[row_start + idx]) : 0;
      const int coarse = valid
          ? static_cast<int>(convert_to_uint8(input[row_start + idx]))
          : 0;
      bool prefix_match = true;
#pragma unroll
      for (int previous = 0; previous < round; ++previous) {
        prefix_match =
            prefix_match &&
            (((key >> (24 - previous * 8)) & 0xFFu) ==
             s_threshold_bytes[previous]);
      }
      const bool keep = valid &&
          (coarse > coarse_threshold ||
           (coarse == coarse_threshold &&
            (!prefix_match ||
             static_cast<int>((key >> shift) & 0xFFu) >= threshold)));
      __syncwarp();
      const unsigned mask = __ballot_sync(0xffffffffu, keep);
      if (keep) {
        const int rank =
            local_count + __popc(mask & ((1u << lane) - 1u));
        s_input_idx[warp * kWarpCapacity + rank] = idx;
      }
      local_count += __popc(mask);
    }
    if (lane == 0) {
      s_num_input_warp[warp] = local_count;
    }
    __syncthreads();
  }
  stable_collect_buffer<kTopK, false>(
      input,
      index,
      row_start,
      coarse_threshold,
      s_pivot_key,
      s_tie_take,
      s_input_idx,
      s_num_input_warp);
}

// Exact radix selection followed by one stable logical-order compaction.
// The radix phase computes the coarse cutoff, the exact 32-bit pivot key, and
// the number of low-index ties to keep.  The collector then emits indices in
// row order without atomics or a post-selection sort.
template <int kTopK, bool kUsePDL>
__global__ __launch_bounds__(fast_topk_detail::kThreadsPerBlock) void fast_topk_kernel(
    const fast_topk_detail::FastTopKParams __grid_constant__ params) {
  using namespace fast_topk_detail;
  device::PDLWaitPrimary<kUsePDL>();

  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = params.row_starts[bid];
  const auto length = params.lengths[bid];
  const auto indice = params.indices + bid * kTopK;
  const auto score = params.input + bid * params.input_stride;
  if (length <= kTopK) {
    naive_topk<kTopK>(indice, length);
  } else {
    radix_select_topk<kTopK>(score, indice, row_start, length);
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace fast_topk_detail

/**
 * \brief Per-row top-k selection over ragged rows of a fp32 score matrix.
 *
 * Row b selects the kTopK largest values in
 * score[b, row_starts[b] : row_starts[b] + lengths[b]) and writes their
 * indices (relative to row_starts[b]) into indices[b]. Unfilled slots are
 * -1 when lengths[b] < kTopK. Scores may be finite or +/- infinity; NaN is
 * outside the kernel contract because the fp16 coarse key canonicalizes it.
 */
template <int kTopK, bool kUsePDL>
struct FastTopKKernel {
  static constexpr auto kernel = fast_topk_detail::fast_topk_kernel<kTopK, kUsePDL>;

  template <std::size_t kDynamicSmem>
  static void setup_dynamic_smem() {
    static const auto result = [] {
      const auto fptr = std::bit_cast<const void*>(kernel);
      return ::cudaFuncSetAttribute(
          fptr, ::cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(kDynamicSmem));
    }();
    host::RuntimeDeviceCheck(result);
  }

  static void
  run(const tvm::ffi::TensorView score,
      const tvm::ffi::TensorView row_starts,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView lengths) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto L = SymbolicSize{"length"};
    auto S = SymbolicSize{"input_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, L})  // score
        .with_strides({S, 1})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(score);
    TensorMatcher({B})  // row_starts
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(row_starts);
    TensorMatcher({B, kTopK})  // indices
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(indices);
    TensorMatcher({B})  // lengths
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(lengths);

    const auto params = fast_topk_detail::FastTopKParams{
        .input = static_cast<const float*>(score.data_ptr()),
        .row_starts = static_cast<const int32_t*>(row_starts.data_ptr()),
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .input_stride = S.unwrap(),
    };

    const auto num_rows = static_cast<uint32_t>(B.unwrap());
    setup_dynamic_smem<fast_topk_detail::kSmemBytes>();
    LaunchKernel(num_rows, fast_topk_detail::kThreadsPerBlock, device.unwrap(), fast_topk_detail::kSmemBytes)
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
