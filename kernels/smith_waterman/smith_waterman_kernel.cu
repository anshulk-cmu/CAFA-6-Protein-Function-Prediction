// smith_waterman_kernel.cu
// GPU-Accelerated Smith-Waterman Local Sequence Alignment
// Phase 2A: CAFA6 Project - Educational Implementation

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// ============================================================================
// BLOSUM62 Substitution Matrix (Constant Memory for Fast Access)
// ============================================================================
// Stores amino acid similarity scores - accessed frequently by all threads
__constant__ int d_blosum62[24][24];  // 20 amino acids + 4 special chars

// Amino acid encoding: A=0, R=1, N=2, D=3, C=4, Q=5, E=6, G=7, H=8, I=9,
//                      L=10, K=11, M=12, F=13, P=14, S=15, T=16, W=17,
//                      Y=18, V=19, B=20, Z=21, X=22, *=23

// Standard BLOSUM62 matrix (used for protein alignment scoring)
const int h_blosum62[24][24] = {
    // A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   B   Z   X   *
    {  4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4}, // A
    { -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4}, // R
    { -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4}, // N
    { -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4}, // D
    {  0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4}, // C
    { -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4}, // Q
    { -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4}, // E
    {  0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4}, // G
    { -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4}, // H
    { -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4}, // I
    { -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4}, // L
    { -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4}, // K
    { -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4}, // M
    { -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4}, // F
    { -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4}, // P
    {  1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4}, // S
    {  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4}, // T
    { -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4}, // W
    { -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4}, // Y
    {  0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4}, // V
    { -2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4}, // B
    { -1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4}, // Z
    {  0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4}, // X
    { -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}  // *
};

// Gap penalties
#define GAP_OPEN -10
#define GAP_EXTEND -1

// Maximum sequence length (for shared memory sizing)
#define MAX_SEQ_LEN 1024
#define TILE_SIZE 16  // 16x16 shared memory tiles

// ============================================================================
// Helper Functions
// ============================================================================

// Encode amino acid character to index (0-23)
__device__ __forceinline__ int aa_to_index(char aa) {
    switch(aa) {
        case 'A': return 0;  case 'R': return 1;  case 'N': return 2;
        case 'D': return 3;  case 'C': return 4;  case 'Q': return 5;
        case 'E': return 6;  case 'G': return 7;  case 'H': return 8;
        case 'I': return 9;  case 'L': return 10; case 'K': return 11;
        case 'M': return 12; case 'F': return 13; case 'P': return 14;
        case 'S': return 15; case 'T': return 16; case 'W': return 17;
        case 'Y': return 18; case 'V': return 19; case 'B': return 20;
        case 'Z': return 21; case 'X': return 22; case '*': return 23;
        default: return 22;  // Unknown -> X
    }
}

// ============================================================================
// Smith-Waterman Kernel (Anti-diagonal Parallelization)
// ============================================================================

__global__ void smith_waterman_kernel(
    const char* sequences_a,     // Query sequences (batch)
    const char* sequences_b,     // Database sequences (batch)
    const int* seq_lengths_a,    // Length of each query
    const int* seq_lengths_b,    // Length of each database seq
    const int* seq_offsets_a,    // Offset into sequences_a buffer
    const int* seq_offsets_b,    // Offset into sequences_b buffer
    float* scores,               // Output: alignment scores
    int num_pairs                // Number of sequence pairs to align
) {
    // Each block processes one sequence pair
    int pair_idx = blockIdx.x;
    if (pair_idx >= num_pairs) return;

    // Thread index within block
    int tid = threadIdx.x;

    // Get sequence information for this pair
    int len_a = seq_lengths_a[pair_idx];
    int len_b = seq_lengths_b[pair_idx];
    int offset_a = seq_offsets_a[pair_idx];
    int offset_b = seq_offsets_b[pair_idx];

    const char* seq_a = sequences_a + offset_a;
    const char* seq_b = sequences_b + offset_b;

    // Shared memory for storing tiles of the scoring matrix
    __shared__ int tile_current[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
    __shared__ int tile_previous[TILE_SIZE][TILE_SIZE + 1];
    __shared__ int max_score_shared[256];  // For reduction to find max score

    // Each thread will track maximum score seen
    int local_max_score = 0;

    // Number of tiles needed for each dimension
    int tiles_a = (len_a + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_b = (len_b + TILE_SIZE - 1) / TILE_SIZE;

    // Process anti-diagonals of tiles
    // Anti-diagonal index determines which tiles can be computed in parallel
    int max_diag = tiles_a + tiles_b - 1;

    for (int diag = 0; diag < max_diag; diag++) {
        // Determine which tile this thread will work on
        // Threads are distributed across the anti-diagonal

        int tile_row = tid / TILE_SIZE;  // Which tile row (0 to tiles_a-1)
        int tile_col = diag - tile_row;   // Which tile col (0 to tiles_b-1)

        // Check if this tile is valid for this anti-diagonal
        if (tile_row >= 0 && tile_row < tiles_a &&
            tile_col >= 0 && tile_col < tiles_b) {

            // Process 16x16 tile
            int local_row = tid % TILE_SIZE;
            int local_col = threadIdx.y;  // Assuming 2D block (16x16)

            int global_i = tile_row * TILE_SIZE + local_row;
            int global_j = tile_col * TILE_SIZE + local_col;

            if (global_i < len_a && global_j < len_b) {
                // Get amino acid indices
                int aa_a = aa_to_index(seq_a[global_i]);
                int aa_b = aa_to_index(seq_b[global_j]);

                // Get substitution score from BLOSUM62
                int match_score = d_blosum62[aa_a][aa_b];

                // Smith-Waterman recurrence relation
                int score_diag = 0;   // From diagonal (will load from previous tile)
                int score_up = 0;     // From above (gap in seq_b)
                int score_left = 0;   // From left (gap in seq_a)

                // TODO: Load scores from previous tiles/cells
                // This is simplified - full implementation needs proper boundary handling

                // Compute current cell score
                int score_match = score_diag + match_score;
                int score_gap_a = score_up + GAP_EXTEND;
                int score_gap_b = score_left + GAP_EXTEND;

                // Smith-Waterman: max of (match, gap_a, gap_b, 0)
                int cell_score = max(0, max(score_match, max(score_gap_a, score_gap_b)));

                // Store in shared memory tile
                tile_current[local_row][local_col] = cell_score;

                // Track maximum score for this thread
                local_max_score = max(local_max_score, cell_score);
            }
        }

        // Synchronize threads before moving to next anti-diagonal
        __syncthreads();

        // Swap tile buffers (current becomes previous for next iteration)
        if (tid < TILE_SIZE * TILE_SIZE) {
            int r = tid / TILE_SIZE;
            int c = tid % TILE_SIZE;
            tile_previous[r][c] = tile_current[r][c];
        }

        __syncthreads();
    }

    // ========================================================================
    // Parallel Reduction: Find Maximum Score Across All Threads
    // ========================================================================

    // Store each thread's max score in shared memory
    max_score_shared[tid] = local_max_score;
    __syncthreads();

    // Tree reduction to find global maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_score_shared[tid] = max(max_score_shared[tid],
                                       max_score_shared[tid + stride]);
        }
        __syncthreads();
    }

    // Thread 0 writes final result
    if (tid == 0) {
        // Normalize score to [0, 1] range (optional, for easier interpretation)
        float normalized_score = (float)max_score_shared[0] /
                                 (float)max(len_a, len_b);
        scores[pair_idx] = normalized_score;
    }
}

// ============================================================================
// Kernel Launch Wrapper (called from C++ wrapper)
// ============================================================================

extern "C" {

// Initialize BLOSUM62 matrix in constant memory
void init_blosum62() {
    cudaMemcpyToSymbol(d_blosum62, h_blosum62, sizeof(h_blosum62));
}

// Launch Smith-Waterman kernel
void launch_smith_waterman(
    const char* d_sequences_a,
    const char* d_sequences_b,
    const int* d_seq_lengths_a,
    const int* d_seq_lengths_b,
    const int* d_seq_offsets_a,
    const int* d_seq_offsets_b,
    float* d_scores,
    int num_pairs,
    cudaStream_t stream
) {
    // Configure kernel launch
    // One block per sequence pair, 256 threads per block
    dim3 blockDim(16, 16);  // 16x16 = 256 threads (tile-based)
    dim3 gridDim(num_pairs);

    // Launch kernel
    smith_waterman_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_sequences_a, d_sequences_b,
        d_seq_lengths_a, d_seq_lengths_b,
        d_seq_offsets_a, d_seq_offsets_b,
        d_scores, num_pairs
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

}  // extern "C"
