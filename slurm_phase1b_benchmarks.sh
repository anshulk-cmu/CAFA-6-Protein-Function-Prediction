#!/bin/bash
#SBATCH --job-name=cafa6_phase1b
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/data/user_data/anshulk/cafa6/logs/slurm/phase1b_%j.out
#SBATCH --error=/data/user_data/anshulk/cafa6/logs/slurm/phase1b_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# Phase 1B: Comprehensive Benchmarking for CAFA 6 Project
# - CPU baseline benchmarks
# - GPU benchmarks
# - Profiling with torch.profiler
# - Embedding concatenation and validation

set -e  # Exit on error

echo "========================================================================"
echo "Phase 1B: Benchmarking and Analysis"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Setup environment
echo "Setting up environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cafa6

# Set paths
PROJECT_DIR="/home/anshulk/CAFA-6-Protein-Function-Prediction"
DATA_DIR="/data/user_data/anshulk/cafa6/data"
EMBEDDINGS_DIR="/data/user_data/anshulk/cafa6/embeddings"
BENCHMARK_DIR="$PROJECT_DIR/benchmark_results"  # Keep in home (small JSON files)
TRACES_DIR="/data/user_data/anshulk/cafa6/traces"
LOG_DIR="/data/user_data/anshulk/cafa6/logs"

cd $PROJECT_DIR

# Create necessary directories
mkdir -p $BENCHMARK_DIR
mkdir -p $TRACES_DIR
mkdir -p $DATA_DIR

# Check conda environment
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Create log directory
mkdir -p $LOG_DIR/slurm

# GPU info
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
fi

# Models to benchmark (representative subset)
MODELS=("esm2_3B" "esm_c_600m" "prot_t5_xl")

# ============================================================================
# Stage 1: Create Benchmark Dataset
# ============================================================================

echo "========================================================================"
echo "Stage 1: Creating Benchmark Dataset (1000 proteins)"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

python create_benchmark_dataset.py \
    --input $DATA_DIR/train_sequences.fasta \
    --output $DATA_DIR/train_sequences_benchmark_1k.fasta \
    --metadata $DATA_DIR/train_sequences_benchmark_1k_metadata.json \
    --n-total 1000 \
    --n-short 200 \
    --n-medium 600 \
    --n-long 200 \
    --seed 42

if [ $? -eq 0 ]; then
    echo "✓ Benchmark dataset created successfully"
else
    echo "✗ Failed to create benchmark dataset"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Stage 2: CPU Baseline Benchmarks"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

# Run CPU benchmarks sequentially for each model
for model in "${MODELS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Running CPU benchmark for $model"
    echo "----------------------------------------------------------------------"

    python benchmark_embeddings_cpu.py \
        --model $model \
        --input $DATA_DIR/train_sequences_benchmark_1k.fasta \
        --output-dir $BENCHMARK_DIR \
        --num-threads 16

    if [ $? -eq 0 ]; then
        echo "✓ CPU benchmark for $model completed"
    else
        echo "✗ CPU benchmark for $model failed"
    fi

    echo ""
done

echo "CPU benchmarks completed at: $(date)"
echo ""

# ============================================================================
# Stage 3: GPU Benchmarks
# ============================================================================

echo "========================================================================"
echo "Stage 3: GPU Benchmarks"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

# Run GPU benchmarks sequentially for each model
for model in "${MODELS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Running GPU benchmark for $model"
    echo "----------------------------------------------------------------------"

    python benchmark_embeddings_gpu.py \
        --model $model \
        --input $DATA_DIR/train_sequences_benchmark_1k.fasta \
        --output-dir $BENCHMARK_DIR \
        --gpu-id 0

    if [ $? -eq 0 ]; then
        echo "✓ GPU benchmark for $model completed"
    else
        echo "✗ GPU benchmark for $model failed"
    fi

    echo ""
done

echo "GPU benchmarks completed at: $(date)"
echo ""

# ============================================================================
# Stage 4: Profiling with torch.profiler
# ============================================================================

echo "========================================================================"
echo "Stage 4: Profiling with torch.profiler"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

# Profile each model to identify kernel-level bottlenecks
for model in "${MODELS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Profiling $model"
    echo "----------------------------------------------------------------------"

    # Set batch size based on model (smaller for larger models)
    if [ "$model" = "esm2_3B" ]; then
        BATCH_SIZE=24
    elif [ "$model" = "esm_c_600m" ]; then
        BATCH_SIZE=32
    else
        BATCH_SIZE=32
    fi

    python utils/profile_embeddings.py \
        --model $model \
        --input $DATA_DIR/train_sequences_benchmark_1k.fasta \
        --batch-size $BATCH_SIZE \
        --num-batches 3 \
        --output-dir $TRACES_DIR \
        --gpu-id 0

    if [ $? -eq 0 ]; then
        echo "✓ Profiling for $model completed"
    else
        echo "✗ Profiling for $model failed"
    fi

    echo ""
done

echo "Profiling completed at: $(date)"
echo ""

# ============================================================================
# Stage 5: Embedding Concatenation
# ============================================================================

echo "========================================================================"
echo "Stage 5: Concatenating Embeddings"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

# Check if all 5 model embeddings exist
echo "Checking for required embedding files..."
REQUIRED_MODELS=("esm2_3B" "esm_c_600m" "esm1b" "prot_t5_xl" "prost_t5")
ALL_EXISTS=true

for split in train test; do
    echo "  Checking $split split..."
    for model in "${REQUIRED_MODELS[@]}"; do
        filepath="$EMBEDDINGS_DIR/${split}_embeddings_${model}.pt"
        if [ -f "$filepath" ]; then
            echo "    ✓ $model: found"
        else
            echo "    ✗ $model: NOT FOUND ($filepath)"
            ALL_EXISTS=false
        fi
    done
done

if [ "$ALL_EXISTS" = true ]; then
    echo ""
    echo "All embedding files found. Proceeding with concatenation..."
    echo ""

    # Concatenate train embeddings
    echo "----------------------------------------------------------------------"
    echo "Concatenating train embeddings"
    echo "----------------------------------------------------------------------"

    python utils/concatenate_embeddings.py \
        --split train \
        --embeddings-dir $EMBEDDINGS_DIR \
        --output-dir $EMBEDDINGS_DIR \
        --normalize l2

    if [ $? -eq 0 ]; then
        echo "✓ Train embeddings concatenated"
    else
        echo "✗ Train concatenation failed"
    fi

    echo ""

    # Concatenate test embeddings
    echo "----------------------------------------------------------------------"
    echo "Concatenating test embeddings"
    echo "----------------------------------------------------------------------"

    python utils/concatenate_embeddings.py \
        --split test \
        --embeddings-dir $EMBEDDINGS_DIR \
        --output-dir $EMBEDDINGS_DIR \
        --normalize l2

    if [ $? -eq 0 ]; then
        echo "✓ Test embeddings concatenated"
    else
        echo "✗ Test concatenation failed"
    fi

    echo ""

    # ========================================================================
    # Stage 6: Validation
    # ========================================================================

    echo "======================================================================"
    echo "Stage 6: Validating Concatenated Embeddings"
    echo "======================================================================"
    echo "Start time: $(date)"
    echo ""

    # Validate train
    echo "----------------------------------------------------------------------"
    echo "Validating train embeddings"
    echo "----------------------------------------------------------------------"

    python utils/validate_concatenated_embeddings.py \
        --split train \
        --embeddings-dir $EMBEDDINGS_DIR

    if [ $? -eq 0 ]; then
        echo "✓ Train validation completed"
    else
        echo "✗ Train validation failed"
    fi

    echo ""

    # Validate test
    echo "----------------------------------------------------------------------"
    echo "Validating test embeddings"
    echo "----------------------------------------------------------------------"

    python utils/validate_concatenated_embeddings.py \
        --split test \
        --embeddings-dir $EMBEDDINGS_DIR

    if [ $? -eq 0 ]; then
        echo "✓ Test validation completed"
    else
        echo "✗ Test validation failed"
    fi

else
    echo ""
    echo "⚠ WARNING: Not all embedding files found. Skipping concatenation."
    echo "   Make sure Phase 1A embedding generation completed successfully."
fi

echo ""

# ============================================================================
# Final Summary
# ============================================================================

echo "========================================================================"
echo "Phase 1B Complete"
echo "========================================================================"
echo "End time: $(date)"
echo ""

# List generated files
echo "Generated Files:"
echo "----------------------------------------------------------------------"

echo "Benchmark Dataset:"
if [ -f "$DATA_DIR/train_sequences_benchmark_1k.fasta" ]; then
    echo "  ✓ $DATA_DIR/train_sequences_benchmark_1k.fasta"
else
    echo "  ✗ $DATA_DIR/train_sequences_benchmark_1k.fasta (missing)"
fi

echo ""
echo "Benchmark Results:"
for model in "${MODELS[@]}"; do
    cpu_file="$BENCHMARK_DIR/${model}_cpu_1k.json"
    gpu_file="$BENCHMARK_DIR/${model}_gpu_1k.json"

    if [ -f "$cpu_file" ]; then
        echo "  ✓ $cpu_file"
    else
        echo "  ✗ $cpu_file (missing)"
    fi

    if [ -f "$gpu_file" ]; then
        echo "  ✓ $gpu_file"
    else
        echo "  ✗ $gpu_file (missing)"
    fi
done

echo ""
echo "Profiling Traces:"
for model in "${MODELS[@]}"; do
    if ls $TRACES_DIR/${model}_profile*.json 1> /dev/null 2>&1; then
        echo "  ✓ ${model}_profile*.json"
    else
        echo "  ✗ ${model}_profile*.json (missing)"
    fi
done

echo ""
echo "Concatenated Embeddings:"
for split in train test; do
    concat_file="$EMBEDDINGS_DIR/${split}_embeddings_concatenated.pt"
    meta_file="$EMBEDDINGS_DIR/${split}_embeddings_metadata.json"

    if [ -f "$concat_file" ]; then
        size=$(du -h "$concat_file" | cut -f1)
        echo "  ✓ ${split}_embeddings_concatenated.pt ($size)"
    else
        echo "  ✗ ${split}_embeddings_concatenated.pt (missing)"
    fi

    if [ -f "$meta_file" ]; then
        echo "  ✓ ${split}_embeddings_metadata.json"
    else
        echo "  ✗ ${split}_embeddings_metadata.json (missing)"
    fi
done

echo ""
echo "Validation Reports:"
for split in train test; do
    report_file="$EMBEDDINGS_DIR/validation_report_${split}.txt"
    if [ -f "$report_file" ]; then
        echo "  ✓ validation_report_${split}.txt"
    else
        echo "  ✗ validation_report_${split}.txt (missing)"
    fi
done

echo ""
echo "========================================================================"
echo "Next Steps:"
echo "========================================================================"
echo "1. Review benchmark results in benchmark_results/"
echo "2. Analyze profiling traces in traces/"
echo "3. Check validation reports in $EMBEDDINGS_DIR"
echo "4. Generate performance visualizations"
echo "5. Write Phase 1B report"
echo ""
echo "========================================================================"
