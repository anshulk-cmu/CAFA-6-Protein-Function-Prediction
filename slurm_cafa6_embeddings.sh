#!/bin/bash
#SBATCH --job-name=cafa6_emb_full
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=/data/user_data/anshulk/cafa6/logs/slurm/cafa6_embeddings_%j.out
#SBATCH --error=/data/user_data/anshulk/cafa6/logs/slurm/cafa6_embeddings_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu
#SBATCH --requeue

set -e

mkdir -p /data/user_data/anshulk/cafa6/logs/{slurm,gpu_monitoring}

echo "=========================================="
echo "CAFA-6 Full Embedding Generation Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Time limit: 48 hours"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate cafa6

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment"
    exit 1
fi

echo "Environment Information:"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Set environment variables
export TRANSFORMERS_CACHE=/data/hf_cache/transformers
export HF_HOME=/data/hf_cache
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Environment Variables:"
echo "  HF_HOME: $HF_HOME"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

# Verify data files
echo "Verifying data files..."
if [ ! -f "/data/user_data/anshulk/cafa6/data/train_sequences.fasta" ]; then
    echo "ERROR: train_sequences.fasta not found"
    exit 1
fi
if [ ! -f "/data/user_data/anshulk/cafa6/data/testsuperset.fasta" ]; then
    echo "ERROR: testsuperset.fasta not found"
    exit 1
fi
echo "Data files verified"
echo "  Train: $(wc -l < /data/user_data/anshulk/cafa6/data/train_sequences.fasta) lines"
echo "  Test: $(wc -l < /data/user_data/anshulk/cafa6/data/testsuperset.fasta) lines"
echo ""

cd /home/anshulk/cafa6

echo "Working directory: $(pwd)"
echo ""

# Start GPU monitor in background for entire run
echo "Starting GPU monitor..."
python gpu_monitor.py > /data/user_data/anshulk/cafa6/logs/gpu_monitoring/monitor_${SLURM_JOB_ID}.log 2>&1 &
MONITOR_PID=$!
echo "GPU monitor started (PID: $MONITOR_PID)"
sleep 3
echo ""

GLOBAL_START_TIME=$(date +%s)

# ========================================
# PHASE 1: ESM Embeddings
# ========================================
echo "=========================================="
echo "PHASE 1: ESM Embedding Generation"
echo "=========================================="
echo "Models: ESM2-3B, ESM-C-600M, ESM1b"
echo "Config: config.yaml"
echo "Start time: $(date)"
echo ""

ESM_START_TIME=$(date +%s)

python generate_embeddings.py --config config.yaml 2>&1 | tee /data/user_data/anshulk/cafa6/logs/embeddings_esm_${SLURM_JOB_ID}.log

ESM_EXIT_STATUS=${PIPESTATUS[0]}

ESM_END_TIME=$(date +%s)
ESM_DURATION=$((ESM_END_TIME - ESM_START_TIME))
ESM_HOURS=$((ESM_DURATION / 3600))
ESM_MINUTES=$(((ESM_DURATION % 3600) / 60))
ESM_SECONDS=$((ESM_DURATION % 60))

echo ""
echo "=========================================="
echo "ESM Phase Summary"
echo "=========================================="
echo "Exit status: $ESM_EXIT_STATUS"
echo "Duration: ${ESM_HOURS}h ${ESM_MINUTES}m ${ESM_SECONDS}s"
echo ""

if [ $ESM_EXIT_STATUS -ne 0 ]; then
    echo "ERROR: ESM embedding generation failed with exit status $ESM_EXIT_STATUS"
    echo "Stopping GPU monitor..."
    kill $MONITOR_PID 2>/dev/null || true
    wait $MONITOR_PID 2>/dev/null || true
    exit $ESM_EXIT_STATUS
fi

echo "ESM embeddings completed successfully"
echo "Generated files:"
for model in esm2_3b esm_c_600m esm1b; do
    for split in train test; do
        file="/data/user_data/anshulk/cafa6/embeddings/${split}_embeddings_${model}.pt"
        if [ -f "$file" ]; then
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "  ${split}_${model}: ${size}"
        fi
    done
done
echo ""

# ========================================
# GPU Cleanup Between Phases
# ========================================
echo "=========================================="
echo "GPU Cleanup"
echo "=========================================="
echo "Clearing GPU memory before T5 phase..."
echo ""

# Kill any stray Python processes
pkill -f generate_embeddings.py || true
sleep 2

# Force GPU memory cleanup using Python
python -c "import torch; import gc; torch.cuda.empty_cache(); gc.collect(); print('GPU memory cleared')"

# Reset GPU if needed (this requires nvidia-smi)
nvidia-smi --gpu-reset || echo "GPU reset not available (requires root), continuing..."

# Show current GPU state
echo "GPU state after cleanup:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv
echo ""

# Wait a bit for GPU to stabilize
sleep 5

echo "GPU cleanup complete"
echo ""

# ========================================
# PHASE 2: T5 Embeddings
# ========================================
echo "=========================================="
echo "PHASE 2: T5 Embedding Generation"
echo "=========================================="
echo "Models: ProtT5-XL, ProstT5"
echo "Config: config_t5.yaml"
echo "Start time: $(date)"
echo ""

T5_START_TIME=$(date +%s)

python generate_embeddings_t5.py --config config_t5.yaml 2>&1 | tee /data/user_data/anshulk/cafa6/logs/embeddings_t5_${SLURM_JOB_ID}.log

T5_EXIT_STATUS=${PIPESTATUS[0]}

T5_END_TIME=$(date +%s)
T5_DURATION=$((T5_END_TIME - T5_START_TIME))
T5_HOURS=$((T5_DURATION / 3600))
T5_MINUTES=$(((T5_DURATION % 3600) / 60))
T5_SECONDS=$((T5_DURATION % 60))

echo ""
echo "=========================================="
echo "T5 Phase Summary"
echo "=========================================="
echo "Exit status: $T5_EXIT_STATUS"
echo "Duration: ${T5_HOURS}h ${T5_MINUTES}m ${T5_SECONDS}s"
echo ""

if [ $T5_EXIT_STATUS -ne 0 ]; then
    echo "ERROR: T5 embedding generation failed with exit status $T5_EXIT_STATUS"
else
    echo "T5 embeddings completed successfully"
    echo "Generated files:"
    for model in prot_t5_xl prost_t5; do
        for split in train test; do
            file="/data/user_data/anshulk/cafa6/embeddings/${split}_embeddings_${model}.pt"
            if [ -f "$file" ]; then
                size=$(ls -lh "$file" | awk '{print $5}')
                echo "  ${split}_${model}: ${size}"
            fi
        done
    done
fi
echo ""

# ========================================
# Stop GPU Monitor
# ========================================
echo "Stopping GPU monitor..."
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true
echo "GPU monitor stopped"
echo ""

# ========================================
# Final Summary
# ========================================
GLOBAL_END_TIME=$(date +%s)
GLOBAL_DURATION=$((GLOBAL_END_TIME - GLOBAL_START_TIME))
GLOBAL_HOURS=$((GLOBAL_DURATION / 3600))
GLOBAL_MINUTES=$(((GLOBAL_DURATION % 3600) / 60))
GLOBAL_SECONDS=$((GLOBAL_DURATION % 60))

echo "=========================================="
echo "FINAL JOB SUMMARY"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "Phase 1 (ESM):"
echo "  Status: $([[ $ESM_EXIT_STATUS -eq 0 ]] && echo 'SUCCESS' || echo 'FAILED')"
echo "  Duration: ${ESM_HOURS}h ${ESM_MINUTES}m ${ESM_SECONDS}s"
echo ""
echo "Phase 2 (T5):"
echo "  Status: $([[ $T5_EXIT_STATUS -eq 0 ]] && echo 'SUCCESS' || echo 'FAILED')"
echo "  Duration: ${T5_HOURS}h ${T5_MINUTES}m ${T5_SECONDS}s"
echo ""
echo "Total Duration: ${GLOBAL_HOURS}h ${GLOBAL_MINUTES}m ${GLOBAL_SECONDS}s"
echo "Started: $(date -d @$GLOBAL_START_TIME)"
echo "Completed: $(date -d @$GLOBAL_END_TIME)"
echo ""

# ========================================
# List All Generated Embeddings
# ========================================
echo "=========================================="
echo "ALL GENERATED EMBEDDINGS"
echo "=========================================="
ls -lh /data/user_data/anshulk/cafa6/embeddings/*.pt 2>/dev/null | awk '{print $9, $5}' || echo "No .pt files found"
echo ""

# ========================================
# Storage Usage
# ========================================
echo "=========================================="
echo "STORAGE USAGE"
echo "=========================================="
echo "Embeddings directory:"
du -sh /data/user_data/anshulk/cafa6/embeddings/ 2>/dev/null || echo "Directory not found"
echo ""
echo "Logs directory:"
du -sh /data/user_data/anshulk/cafa6/logs/ 2>/dev/null || echo "Directory not found"
echo ""

# ========================================
# Log File Locations
# ========================================
echo "=========================================="
echo "LOG FILES"
echo "=========================================="
echo "Main logs:"
echo "  ESM: /data/user_data/anshulk/cafa6/logs/embeddings_esm_${SLURM_JOB_ID}.log"
echo "  T5: /data/user_data/anshulk/cafa6/logs/embeddings_t5_${SLURM_JOB_ID}.log"
echo ""
echo "SLURM logs:"
echo "  Output: /data/user_data/anshulk/cafa6/logs/slurm/cafa6_embeddings_${SLURM_JOB_ID}.out"
echo "  Error: /data/user_data/anshulk/cafa6/logs/slurm/cafa6_embeddings_${SLURM_JOB_ID}.err"
echo ""
echo "GPU monitoring:"
echo "  Monitor: /data/user_data/anshulk/cafa6/logs/gpu_monitoring/monitor_${SLURM_JOB_ID}.log"
echo ""
echo "Worker logs:"
echo "  Location: /data/user_data/anshulk/cafa6/logs/*_*.log"
echo ""

# ========================================
# Final GPU State
# ========================================
echo "=========================================="
echo "FINAL GPU STATE"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv
echo ""

# ========================================
# Additional Statistics
# ========================================
echo "=========================================="
echo "ADDITIONAL STATISTICS"
echo "=========================================="
echo "For detailed job statistics after completion:"
echo "  seff $SLURM_JOB_ID"
echo "  sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize,AveCPU"
echo ""

# Determine final exit status
FINAL_EXIT_STATUS=0
if [ $ESM_EXIT_STATUS -ne 0 ]; then
    FINAL_EXIT_STATUS=$ESM_EXIT_STATUS
elif [ $T5_EXIT_STATUS -ne 0 ]; then
    FINAL_EXIT_STATUS=$T5_EXIT_STATUS
fi

if [ $FINAL_EXIT_STATUS -eq 0 ]; then
    echo "=========================================="
    echo "✓ ALL EMBEDDING GENERATION COMPLETED SUCCESSFULLY"
    echo "=========================================="
else
    echo "=========================================="
    echo "✗ SOME PHASES FAILED - CHECK LOGS"
    echo "=========================================="
fi

echo ""
exit $FINAL_EXIT_STATUS
