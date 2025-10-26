#!/bin/bash
#SBATCH --job-name=cafa6_embeddings
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/data/user_data/anshulk/cafa6/logs/slurm/cafa6_embeddings_%j.out
#SBATCH --error=/data/user_data/anshulk/cafa6/logs/slurm/cafa6_embeddings_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu
#SBATCH --requeue

set -e

mkdir -p /data/user_data/anshulk/cafa6/logs/{slurm,gpu_monitoring}

echo "=========================================="
echo "CAFA-6 Embedding Generation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Time limit: 24 hours"
echo ""

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
echo "Config file: $(ls -lh config.yaml)"
echo ""

echo "Starting GPU monitor..."
python gpu_monitor.py > /data/user_data/anshulk/cafa6/logs/gpu_monitoring/monitor_${SLURM_JOB_ID}.log 2>&1 &
MONITOR_PID=$!
echo "GPU monitor started (PID: $MONITOR_PID)"
sleep 3
echo ""

START_TIME=$(date +%s)
echo "=========================================="
echo "Starting Embedding Generation"
echo "=========================================="
echo "Start time: $(date)"
echo ""

python generate_embeddings.py --config config.yaml 2>&1 | tee /data/user_data/anshulk/cafa6/logs/embeddings_${SLURM_JOB_ID}.log

EXIT_STATUS=${PIPESTATUS[0]}

echo ""
echo "Stopping GPU monitor..."
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true
echo "GPU monitor stopped"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "Job Summary"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Exit status: $EXIT_STATUS"
echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Started: $(date -d @$START_TIME)"
echo "Completed: $(date -d @$END_TIME)"
echo ""

if [ $EXIT_STATUS -eq 0 ]; then
    echo "=========================================="
    echo "Embedding Generation Completed Successfully"
    echo "=========================================="
    echo ""
    echo "Output files:"
    ls -lh /data/user_data/anshulk/cafa6/embeddings/*.pt 2>/dev/null || echo "No .pt files found"
    echo ""
    echo "Generated embeddings:"
    for model in esm2_3b ankh_large prot_t5_xl prot_bert_bfd; do
        for split in train test; do
            file="/data/user_data/anshulk/cafa6/embeddings/${split}_embeddings_${model}.pt"
            if [ -f "$file" ]; then
                size=$(ls -lh "$file" | awk '{print $5}')
                echo "  ${split}_${model}: ${size}"
            fi
        done
    done
    echo ""
    echo "Log files:"
    echo "  Main log: /data/user_data/anshulk/cafa6/logs/embeddings_${SLURM_JOB_ID}.log"
    echo "  GPU monitor: /data/user_data/anshulk/cafa6/logs/gpu_monitoring/monitor_${SLURM_JOB_ID}.log"
    echo "  Worker logs: /data/user_data/anshulk/cafa6/logs/*_*.log"
else
    echo "=========================================="
    echo "Job Failed or Incomplete"
    echo "=========================================="
    echo "Exit status: $EXIT_STATUS"
    echo ""
    echo "Checkpoint files (for resume):"
    ls -lh /data/user_data/anshulk/cafa6/embeddings/*.chk 2>/dev/null || echo "No checkpoint files found"
    echo ""
    echo "Partial embeddings:"
    ls -lh /data/user_data/anshulk/cafa6/embeddings/*.pt 2>/dev/null || echo "No .pt files found yet"
    echo ""
    echo "To resume from checkpoint, rerun:"
    echo "  sbatch slurm_cafa6_embeddings.sh"
    echo ""
    echo "Check error logs:"
    echo "  Error: /data/user_data/anshulk/cafa6/logs/slurm/cafa6_embeddings_${SLURM_JOB_ID}.err"
    echo "  Main: /data/user_data/anshulk/cafa6/logs/embeddings_${SLURM_JOB_ID}.log"
fi

echo ""
echo "=========================================="
echo "Final GPU State"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv
echo ""

echo "=========================================="
echo "Storage Usage"
echo "=========================================="
echo "Embeddings directory:"
du -sh /data/user_data/anshulk/cafa6/embeddings/
echo ""
echo "Logs directory:"
du -sh /data/user_data/anshulk/cafa6/logs/
echo ""

echo "For detailed job statistics after completion:"
echo "  seff $SLURM_JOB_ID"
echo "  sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize,AveCPU"
echo ""

exit $EXIT_STATUS
