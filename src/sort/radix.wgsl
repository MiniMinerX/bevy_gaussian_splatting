#import bevy_gaussian_splatting::bindings::{
    view,
    globals,
    gaussian_uniforms,
    sorting_pass_index,
    sorting,
    status_counters,
    draw_indirect,
    input_entries,
    output_entries,
    sorted_entries,
    DrawIndirect,
    Entry,
}
#import bevy_gaussian_splatting::transform::{
    world_to_clip,
    in_frustum,
}

#ifdef PACKED_F32
#import bevy_gaussian_splatting::packed::get_position
#else

#ifdef BUFFER_STORAGE
#import bevy_gaussian_splatting::planar::get_position
#endif

#endif

#ifdef BUFFER_TEXTURE
#import bevy_gaussian_splatting::texture::get_position
#endif

struct SortingGlobal {
    digit_histogram: array<array<atomic<u32>, #{RADIX_BASE}>, #{RADIX_DIGIT_PLACES}>,
    assignment_counter: atomic<u32>,
    digit_tile_head: array<atomic<u32>, #{RADIX_BASE}>,
}

@group(3) @binding(0) var<uniform> sorting_pass_index: u32;
@group(3) @binding(1) var<storage, read_write> sorting: SortingGlobal;
// Per-tile temporary storage for radix pass C.
@group(3) @binding(2) var<storage, read_write> status_counters: array<array<atomic<u32>, #{RADIX_BASE}>>;
@group(3) @binding(3) var<storage, read_write> draw_indirect: DrawIndirect;
@group(3) @binding(4) var<storage, read_write> input_entries: array<Entry>;
@group(3) @binding(5) var<storage, read_write> output_entries: array<Entry>;

//
// The following three functions (`radix_reset`, `radix_sort_a`, `radix_sort_b`)
// form a standard three-phase GPU sort setup and were already correct.
// They are included here without changes.
//

// Per-`radix_sort_a` workgroup histogram: accumulate here, then flush to `sorting.digit_histogram`
// once per workgroup (reduces global atomic traffic vs 4 atomics per splat on global memory).
var<workgroup> wg_digit_hist: array<array<atomic<u32>, #{RADIX_BASE}>, #{RADIX_DIGIT_PLACES}>;

// One row of `radix_sort_b` (256 threads / workgroup × one workgroup per digit place).
var<workgroup> radix_b_row: array<u32, #{RADIX_BASE}>;

@compute @workgroup_size(#{RADIX_BASE}, #{RADIX_DIGIT_PLACES})
fn radix_reset(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
){
    let b = local_id.x;
    let p = local_id.y;
    atomicStore(&sorting.digit_histogram[p][b], 0u);
    if (p == 0u) {
        atomicStore(&sorting.digit_tile_head[b], 0u);
    }
    if (global_id.x == 0u && global_id.y == 0u) {
        atomicStore(&sorting.assignment_counter, 0u);
        draw_indirect.instance_count = 0u;
    }
}

@compute @workgroup_size(#{RADIX_BASE}, #{RADIX_DIGIT_PLACES})
fn radix_sort_a(
    @builtin(local_invocation_id) gl_LocalInvocationID: vec3<u32>,
    @builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>,
) {
    if (gl_LocalInvocationID.x == 0u && gl_LocalInvocationID.y == 0u && gl_GlobalInvocationID.x == 0u) {
        draw_indirect.vertex_count = 4u;
        atomicStore(&draw_indirect.instance_count, gaussian_uniforms.count);
    }
    workgroupBarrier();

    // One thread per histogram cell (256×4).
    atomicStore(&wg_digit_hist[gl_LocalInvocationID.y][gl_LocalInvocationID.x], 0u);
    workgroupBarrier();

    let thread_index = gl_GlobalInvocationID.x * #{RADIX_DIGIT_PLACES}u + gl_GlobalInvocationID.y;
    let start_entry_index = thread_index * #{ENTRIES_PER_INVOCATION_A}u;
    let end_entry_index = start_entry_index + #{ENTRIES_PER_INVOCATION_A}u;

    for (var entry_index = start_entry_index; entry_index < end_entry_index; entry_index += 1u) {
        if (entry_index >= gaussian_uniforms.count) { continue; }
        var key: u32 = 0xFFFFFFFFu;
        let position = vec4<f32>(get_position(entry_index), 1.0);
        let transformed_position = (gaussian_uniforms.transform * position).xyz;
        let clip_space_pos = world_to_clip(transformed_position);
        let diff = transformed_position - view.world_position;
        let dist2 = dot(diff, diff);
        let dist_bits = bitcast<u32>(dist2);
        let key_distance = 0xFFFFFFFFu - dist_bits;
        if (in_frustum(clip_space_pos.xyz)) {
            key = key_distance;
        }
        input_entries[entry_index].key = key;
        input_entries[entry_index].value = entry_index;
        for(var shift = 0u; shift < #{RADIX_DIGIT_PLACES}u; shift += 1u) {
            let digit = (key >> (shift * #{RADIX_BITS_PER_DIGIT}u)) & (#{RADIX_BASE}u - 1u);
            atomicAdd(&wg_digit_hist[shift][digit], 1u);
        }
    }

    workgroupBarrier();
    let local_count = atomicLoad(&wg_digit_hist[gl_LocalInvocationID.y][gl_LocalInvocationID.x]);
    atomicAdd(&sorting.digit_histogram[gl_LocalInvocationID.y][gl_LocalInvocationID.x], local_count);
}

// Parallel exclusive prefix per histogram row: O(log RADIX_BASE) depth vs serial O(RADIX_BASE).
@compute @workgroup_size(#{RADIX_BASE}, 1, 1)
fn radix_sort_b(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let p = workgroup_id.x;
    let i = local_id.x;
    let v = atomicLoad(&sorting.digit_histogram[p][i]);
    radix_b_row[i] = v;
    workgroupBarrier();

    for (var d = 0u; d < #{RADIX_BITS_PER_DIGIT}u; d += 1u) {
        workgroupBarrier();
        let stride = 1u << d;
        let t = radix_b_row[i];
        if (i >= stride) {
            radix_b_row[i] = radix_b_row[i - stride] + t;
        }
    }
    workgroupBarrier();
    let exclusive = radix_b_row[i] - v;
    atomicStore(&sorting.digit_histogram[p][i], exclusive);
}

// --- SHARED MEMORY for radix pass C ---
var<workgroup> tile_input_entries: array<Entry, #{WORKGROUP_ENTRIES_C}>;
var<workgroup> sorted_tile_entries: array<Entry, #{WORKGROUP_ENTRIES_C}>;
var<workgroup> tile_digit_counts: array<atomic<u32>, #{RADIX_BASE}>;
var<workgroup> local_digit_counts: array<u32, #{RADIX_BASE}>;
var<workgroup> local_digit_offsets: array<u32, #{RADIX_BASE}>;
// Atomic offsets used during parallel intra-tile scatter
var<workgroup> scatter_offsets: array<atomic<u32>, #{RADIX_BASE}>;
var<workgroup> tile_entry_count_ws: u32;
// Scratch for parallel prefix in `radix_sort_c_scatter` step 2 (same size as bin count).
var<workgroup> tile_prefix_scan: array<u32, #{RADIX_BASE}>;
const INVALID_KEY: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(#{WORKGROUP_INVOCATIONS_C})
fn radix_sort_c_count_tiles(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let tile_size = #{WORKGROUP_ENTRIES_C}u;
    let threads = #{WORKGROUP_INVOCATIONS_C}u;
    let global_entry_offset = workgroup_id.y * tile_size;

    if (tid < #{RADIX_BASE}u) {
        atomicStore(&tile_digit_counts[tid], 0u);
    }
    workgroupBarrier();

    for (var i = tid; i < tile_size; i += threads) {
        let idx = global_entry_offset + i;
        if (idx >= gaussian_uniforms.count) {
            continue;
        }

        let entry = input_entries[idx];
        let digit = (entry.key >> (sorting_pass_index * #{RADIX_BITS_PER_DIGIT}u)) & (#{RADIX_BASE}u - 1u);
        atomicAdd(&tile_digit_counts[digit], 1u);
    }
    workgroupBarrier();

    if (tid < #{RADIX_BASE}u) {
        let count = atomicLoad(&tile_digit_counts[tid]);
        atomicStore(&status_counters[workgroup_id.y][tid], count);
    }
}

@compute @workgroup_size(1)
fn radix_sort_c_scan_tiles(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let digit = global_id.y;
    if (digit >= #{RADIX_BASE}u) {
        return;
    }

    let tile_size = #{WORKGROUP_ENTRIES_C}u;
    let tile_count = (gaussian_uniforms.count + tile_size - 1u) / tile_size;

    var sum = atomicLoad(&sorting.digit_histogram[sorting_pass_index][digit]);
    for (var tile = 0u; tile < tile_count; tile += 1u) {
        let count = atomicLoad(&status_counters[tile][digit]);
        atomicStore(&status_counters[tile][digit], sum);
        sum += count;
    }
}

@compute @workgroup_size(#{WORKGROUP_INVOCATIONS_C})
fn radix_sort_c_scatter(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let tile_size = #{WORKGROUP_ENTRIES_C}u;
    let threads = #{WORKGROUP_INVOCATIONS_C}u;
    let global_entry_offset = workgroup_id.y * tile_size;

    // Step 1: Parallel load + parallel digit count.
    if (tid < #{RADIX_BASE}u) {
        atomicStore(&tile_digit_counts[tid], 0u);
    }
    workgroupBarrier();

    for (var i = tid; i < tile_size; i += threads) {
        let idx = global_entry_offset + i;
        if (idx < gaussian_uniforms.count) {
            let entry = input_entries[idx];
            tile_input_entries[i] = entry;
            let digit = (entry.key >> (sorting_pass_index * #{RADIX_BITS_PER_DIGIT}u)) & (#{RADIX_BASE}u - 1u);
            atomicAdd(&tile_digit_counts[digit], 1u);
        } else {
            tile_input_entries[i] = Entry(INVALID_KEY, INVALID_KEY);
        }
    }
    workgroupBarrier();

    // Step 2: Parallel exclusive prefix over per-bin counts (Hillis–Steele, log₂(RADIX_BASE) rounds).
    var orig_i = 0u;
    if (tid < #{RADIX_BASE}u) {
        orig_i = atomicLoad(&tile_digit_counts[tid]);
        tile_prefix_scan[tid] = orig_i;
    }
    workgroupBarrier();

    for (var d = 0u; d < #{RADIX_BITS_PER_DIGIT}u; d += 1u) {
        workgroupBarrier();
        if (tid < #{RADIX_BASE}u) {
            let stride = 1u << d;
            let t = tile_prefix_scan[tid];
            if (tid >= stride) {
                tile_prefix_scan[tid] = tile_prefix_scan[tid - stride] + t;
            }
        }
    }
    workgroupBarrier();

    if (tid < #{RADIX_BASE}u) {
        let inclusive = tile_prefix_scan[tid];
        let excl = inclusive - orig_i;
        local_digit_counts[tid] = orig_i;
        local_digit_offsets[tid] = excl;
    }
    if (tid == #{RADIX_BASE}u - 1u) {
        tile_entry_count_ws = tile_prefix_scan[tid];
    }
    workgroupBarrier();

    // Step 3: Parallel scatter into sorted_tile_entries using atomic offsets.
    if (tid < #{RADIX_BASE}u) {
        atomicStore(&scatter_offsets[tid], local_digit_offsets[tid]);
    }
    workgroupBarrier();

    for (var i = tid; i < tile_size; i += threads) {
        let entry = tile_input_entries[i];
        if (entry.value != INVALID_KEY) {
            let digit = (entry.key >> (sorting_pass_index * #{RADIX_BITS_PER_DIGIT}u)) & (#{RADIX_BASE}u - 1u);
            let dest_idx = atomicAdd(&scatter_offsets[digit], 1u);
            sorted_tile_entries[dest_idx] = entry;
        }
    }
    workgroupBarrier();

    // Step 4: Parallel write from the locally-sorted tile to global memory.
    for (var i = tid; i < tile_size; i += threads) {
        if (i < tile_entry_count_ws) {
            let entry = sorted_tile_entries[i];
            let digit = (entry.key >> (sorting_pass_index * #{RADIX_BITS_PER_DIGIT}u)) & (#{RADIX_BASE}u - 1u);

            let bin_start_offset = local_digit_offsets[digit];
            let rank_in_bin = i - bin_start_offset;
            let global_base = atomicLoad(&status_counters[workgroup_id.y][digit]);
            let dst = global_base + rank_in_bin;

            if (dst < gaussian_uniforms.count) {
                output_entries[dst] = entry;
            }
        }
    }
    if (sorting_pass_index == #{RADIX_DIGIT_PLACES}u - 1u && tid == 0u) {
        atomicStore(&draw_indirect.instance_count, gaussian_uniforms.count);
    }
}
