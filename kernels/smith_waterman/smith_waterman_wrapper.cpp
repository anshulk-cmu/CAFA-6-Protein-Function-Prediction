// smith_waterman_wrapper.cpp
// PyTorch C++ Extension Wrapper for Smith-Waterman CUDA Kernel
// Phase 2A: CAFA6 Project - Production Implementation

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <algorithm>

// Forward declarations of CUDA kernel launchers (defined in .cu file)
extern "C" {
    void init_blosum62();
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
    );
}

// ============================================================================
// Helper: Pack sequences into contiguous buffer
// ============================================================================
// Converts vector of strings into single char buffer for GPU transfer
void pack_sequences(
    const std::vector<std::string>& sequences,
    std::vector<char>& buffer,
    std::vector<int>& lengths,
    std::vector<int>& offsets
) {
    int num_seqs = sequences.size();
    lengths.resize(num_seqs);
    offsets.resize(num_seqs);

    // Calculate total size needed
    size_t total_size = 0;
    for (int i = 0; i < num_seqs; i++) {
        lengths[i] = sequences[i].length();
        offsets[i] = total_size;
        total_size += lengths[i];
    }

    // Pack all sequences into contiguous buffer
    buffer.resize(total_size);
    for (int i = 0; i < num_seqs; i++) {
        std::copy(sequences[i].begin(), sequences[i].end(),
                  buffer.begin() + offsets[i]);
    }
}

// ============================================================================
// Main Alignment Function (Python-callable)
// ============================================================================

/**
 * Align batches of protein sequence pairs using GPU
 *
 * @param sequences_a: Vector of query protein sequences
 * @param sequences_b: Vector of database protein sequences
 * @return: PyTorch tensor of alignment scores [num_pairs]
 */
torch::Tensor align_batch(
    std::vector<std::string> sequences_a,
    std::vector<std::string> sequences_b
) {
    // ========================================================================
    // STEP 1: Validate Input
    // ========================================================================

    int num_pairs = sequences_a.size();

    if (sequences_a.size() != sequences_b.size()) {
        throw std::runtime_error("sequences_a and sequences_b must have same length");
    }

    if (num_pairs == 0) {
        throw std::runtime_error("Empty sequence lists provided");
    }

    // ========================================================================
    // STEP 2: Pack Sequences into Contiguous Buffers (CPU)
    // ========================================================================

    std::vector<char> buffer_a, buffer_b;
    std::vector<int> lengths_a, lengths_b;
    std::vector<int> offsets_a, offsets_b;

    pack_sequences(sequences_a, buffer_a, lengths_a, offsets_a);
    pack_sequences(sequences_b, buffer_b, lengths_b, offsets_b);

    // ========================================================================
    // STEP 3: Allocate GPU Memory
    // ========================================================================

    char* d_sequences_a;
    char* d_sequences_b;
    int* d_lengths_a;
    int* d_lengths_b;
    int* d_offsets_a;
    int* d_offsets_b;
    float* d_scores;

    // Allocate device memory
    cudaMalloc(&d_sequences_a, buffer_a.size() * sizeof(char));
    cudaMalloc(&d_sequences_b, buffer_b.size() * sizeof(char));
    cudaMalloc(&d_lengths_a, num_pairs * sizeof(int));
    cudaMalloc(&d_lengths_b, num_pairs * sizeof(int));
    cudaMalloc(&d_offsets_a, num_pairs * sizeof(int));
    cudaMalloc(&d_offsets_b, num_pairs * sizeof(int));
    cudaMalloc(&d_scores, num_pairs * sizeof(float));

    // Check for allocation errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA malloc failed: ") +
                                cudaGetErrorString(err));
    }

    // ========================================================================
    // STEP 4: Copy Data from CPU to GPU (H2D Transfer)
    // ========================================================================

    cudaMemcpy(d_sequences_a, buffer_a.data(),
               buffer_a.size() * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sequences_b, buffer_b.data(),
               buffer_b.size() * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths_a, lengths_a.data(),
               num_pairs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths_b, lengths_b.data(),
               num_pairs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets_a, offsets_a.data(),
               num_pairs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets_b, offsets_b.data(),
               num_pairs * sizeof(int), cudaMemcpyHostToDevice);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Clean up before throwing
        cudaFree(d_sequences_a);
        cudaFree(d_sequences_b);
        cudaFree(d_lengths_a);
        cudaFree(d_lengths_b);
        cudaFree(d_offsets_a);
        cudaFree(d_offsets_b);
        cudaFree(d_scores);
        throw std::runtime_error(std::string("CUDA memcpy H2D failed: ") +
                                cudaGetErrorString(err));
    }

    // ========================================================================
    // STEP 5: Initialize BLOSUM62 Matrix (First Call Only)
    // ========================================================================

    static bool blosum_initialized = false;
    if (!blosum_initialized) {
        init_blosum62();
        blosum_initialized = true;
    }

    // ========================================================================
    // STEP 6: Launch CUDA Kernel
    // ========================================================================

    // Use default CUDA stream (can be parameterized for multi-stream)
    cudaStream_t stream = 0;

    launch_smith_waterman(
        d_sequences_a,
        d_sequences_b,
        d_lengths_a,
        d_lengths_b,
        d_offsets_a,
        d_offsets_b,
        d_scores,
        num_pairs,
        stream
    );

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Clean up
        cudaFree(d_sequences_a);
        cudaFree(d_sequences_b);
        cudaFree(d_lengths_a);
        cudaFree(d_lengths_b);
        cudaFree(d_offsets_a);
        cudaFree(d_offsets_b);
        cudaFree(d_scores);
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") +
                                cudaGetErrorString(err));
    }

    // ========================================================================
    // STEP 7: Copy Results from GPU to CPU (D2H Transfer)
    // ========================================================================

    // Allocate PyTorch tensor for output (on GPU initially)
    auto options = torch::TensorOptions()
                      .dtype(torch::kFloat32)
                      .device(torch::kCUDA, 0);
    torch::Tensor scores_tensor = torch::empty({num_pairs}, options);

    // Copy scores directly into PyTorch tensor's GPU memory
    cudaMemcpy(scores_tensor.data_ptr<float>(), d_scores,
               num_pairs * sizeof(float), cudaMemcpyDeviceToDevice);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_sequences_a);
        cudaFree(d_sequences_b);
        cudaFree(d_lengths_a);
        cudaFree(d_lengths_b);
        cudaFree(d_offsets_a);
        cudaFree(d_offsets_b);
        cudaFree(d_scores);
        throw std::runtime_error(std::string("CUDA memcpy D2H failed: ") +
                                cudaGetErrorString(err));
    }

    // ========================================================================
    // STEP 8: Clean Up GPU Memory
    // ========================================================================

    cudaFree(d_sequences_a);
    cudaFree(d_sequences_b);
    cudaFree(d_lengths_a);
    cudaFree(d_lengths_b);
    cudaFree(d_offsets_a);
    cudaFree(d_offsets_b);
    cudaFree(d_scores);

    // ========================================================================
    // STEP 9: Return Results to Python
    // ========================================================================

    // Move tensor to CPU for easier Python manipulation
    return scores_tensor.to(torch::kCPU);
}

// ============================================================================
// Optimized Batch Processing for Large Datasets
// ============================================================================

/**
 * Process very large alignment jobs in chunks to avoid GPU memory overflow
 *
 * @param sequences_a: Query sequences
 * @param sequences_b: Database sequences
 * @param chunk_size: Number of pairs to process per GPU batch (default: 10000)
 * @return: Tensor of all alignment scores
 */
torch::Tensor align_large_batch(
    std::vector<std::string> sequences_a,
    std::vector<std::string> sequences_b,
    int chunk_size = 10000
) {
    int total_pairs = sequences_a.size();

    if (total_pairs <= chunk_size) {
        // Small enough to process in one go
        return align_batch(sequences_a, sequences_b);
    }

    // Process in chunks
    std::vector<torch::Tensor> results;

    for (int start = 0; start < total_pairs; start += chunk_size) {
        int end = std::min(start + chunk_size, total_pairs);

        // Extract chunk
        std::vector<std::string> chunk_a(sequences_a.begin() + start,
                                         sequences_a.begin() + end);
        std::vector<std::string> chunk_b(sequences_b.begin() + start,
                                         sequences_b.begin() + end);

        // Process chunk
        torch::Tensor chunk_scores = align_batch(chunk_a, chunk_b);
        results.push_back(chunk_scores);
    }

    // Concatenate all results
    return torch::cat(results, 0);
}

// ============================================================================
// All-vs-All Alignment Matrix (For CAFA Training Data)
// ============================================================================

/**
 * Compute N×N similarity matrix (all-vs-all alignments)
 * Used for: 3000 training proteins → 3000×3000 = 9M alignments
 *
 * @param sequences: List of N protein sequences
 * @param chunk_size: Process this many pairs per GPU batch
 * @return: N×N similarity matrix as PyTorch tensor
 */
torch::Tensor align_all_vs_all(
    std::vector<std::string> sequences,
    int chunk_size = 10000
) {
    int n = sequences.size();

    // Pre-allocate result matrix
    torch::Tensor similarity_matrix = torch::zeros({n, n}, torch::kFloat32);

    // Generate all pairs (upper triangle only, matrix is symmetric)
    std::vector<std::string> pairs_a, pairs_b;
    std::vector<std::pair<int, int>> indices;

    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {  // j >= i for upper triangle
            pairs_a.push_back(sequences[i]);
            pairs_b.push_back(sequences[j]);
            indices.push_back({i, j});
        }
    }

    // Process in chunks
    torch::Tensor scores = align_large_batch(pairs_a, pairs_b, chunk_size);

    // Fill matrix (symmetric)
    auto scores_accessor = scores.accessor<float, 1>();
    for (size_t idx = 0; idx < indices.size(); idx++) {
        int i = indices[idx].first;
        int j = indices[idx].second;
        float score = scores_accessor[idx];

        similarity_matrix[i][j] = score;
        similarity_matrix[j][i] = score;  // Symmetric
    }

    return similarity_matrix;
}

// ============================================================================
// Python Module Bindings (pybind11)
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Smith-Waterman GPU-Accelerated Sequence Alignment (Phase 2A)";

    m.def("align_batch", &align_batch,
          "Align batch of sequence pairs on GPU",
          py::arg("sequences_a"),
          py::arg("sequences_b"));

    m.def("align_large_batch", &align_large_batch,
          "Align large batch with chunking to avoid GPU memory overflow",
          py::arg("sequences_a"),
          py::arg("sequences_b"),
          py::arg("chunk_size") = 10000);

    m.def("align_all_vs_all", &align_all_vs_all,
          "Compute N×N all-vs-all similarity matrix",
          py::arg("sequences"),
          py::arg("chunk_size") = 10000);
}
