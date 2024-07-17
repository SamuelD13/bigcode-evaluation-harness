#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=evaluate            # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=evaluate.log   # Log-file (important!)
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                # Number of GPU's
#SBATCH --time=23:00:00              # Max execution time
#SBATCH --partition=tesla           # Partition
#

# Activate your Anaconda environment
conda activate eval

# Run your Python script
cd /home/sam/bigcode-evaluation-harness
python main.py \
--model deepseek-ai/deepseek-coder-6.7b-instruct  \
--tasks humanevalexplainsynthesize-python \
--do_sample False \
--n_samples 1 \
--batch_size 1 \
--allow_code_execution \
--save_generations \
--trust_remote_code \
--prompt deepseek \
--load_data_path generations_humanevalexplaindescribepython_deepseek-coder-6.7b-instruct_humanevalexplaindescribe-python.json \
--save_generations_path generations_humanevalexplainsynthesizepython_deepseek-coder-6.7b-instruct.json \
--metric_output_path evaluation_humanevalexplainpython_deepseek-coder-6.7b-instruct.json \
--max_length_generation 2048 \
--precision bf16