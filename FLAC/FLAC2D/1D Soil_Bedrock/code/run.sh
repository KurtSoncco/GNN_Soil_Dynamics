#!/bin/bash
# Job name:
#SBATCH --job-name=Seq2Seq
#
# Account:
#SBATCH --account=kurtwal98
#
# Partition:
#SBATCH --partition=partition_name
#
# Wall clock limit:
#SBATCH --time=00:00:30
#
# Activate the virtual environment if needed
# source /path/to/your/virtualenv/bin/activate

# Navigate to the directory containing main.py
cd /path/to/your/project

# Run the main.py script
python main.py

# Deactivate the virtual environment if needed
# deactivate