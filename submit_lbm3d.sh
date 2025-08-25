#!/bin/bash

# Set output/error directories
OUTDIR="/home/eric.vaizman/LBM3D_BB/slurm_out"
ERRDIR="/home/eric.vaizman/LBM3D_BB/slurm_err"
mkdir -p "$OUTDIR" "$ERRDIR"


# Create the SLURM batch script
cat <<EOF > /home/eric.vaizman/LBM3D_BB/eric_submit_lbm3d.sbatch
#!/bin/bash
#SBATCH --job-name=lbm3d_gpu_test
#SBATCH --output=$OUTDIR/slurm-%j.out
#SBATCH --error=$ERRDIR/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=01:45:00
#SBATCH --partition=l40s-shared
#SBATCH --qos=2h_2g
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-gpu=12

echo "================================================================"
echo "Timestamp : \$(date)"
echo "Host : \$(hostname)"
echo "SLURM Job ID : \$SLURM_JOB_ID"
echo "GPU(s) Allocated : \$CUDA_VISIBLE_DEVICES"
echo "================================================================"
echo ""
echo "--- Verifying GPU Accessibility Inside Container ---"
nvidia-smi
echo ""
# mpirun -np \$(( $TOTAL_GPUS + 1 )) julia --compile=all --project=. scripts/run_HPC.jl
EOF

# Submit the job
sbatch /home/eric.vaizman/LBM3D_BB/eric_submit_lbm3d.sbatch