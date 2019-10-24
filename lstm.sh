#!/bin/bash
#SBATCH -A p30991               # Allocation
#SBATCH -p short                # Queue
#SBATCH -t 04:00:00             # Walltime/duration of the job
#SBATCH -N 1                    # Number of Nodes
#SBATCH --mem=0               # Memory per node in GB needed for a job. Also see --mem-per-cpu
#SBATCH --ntasks-per-node=4     # Number of Cores (Processors)
#SBATCH --mail-user=mmachiobiorah2015@u.northwestern.edu  # Designate email address for job communications
#SBATCH --mail-type=BEGIN,END,REQUEUE     # Events options are job BEGIN, END, NONE, FAIL, REQUEUE
#SBATCH --output=/projects/p30991/list_of_foods/output.out    # Path for output must already exist
#SBATCH --error=/projects/p30991/list_of_foods/error.out    # Path for errors must already exist
#SBATCH --job-name="lstm"       # Name of job

# unload any modules that carried over from your command line session
module purge

# add a project directory to your PATH (if needed)
export PATH=$PATH:/projects/p30991/list_of_foods/

# load modules you need to use
module load python/anaconda3.6

#activate my virtual environment 
source dictionary/bin/activate

# Another command you actually want to execute, if needed:
python3 create_lstm_data_set.py