#!/bin/bash -l
#SBATCH --partition=k2-hipri         # queue
#SBATCH --job-name=acc_meta          # Job name
#SBATCH --ntasks=32                  # Number of cores
#SBATCH --time=3:00:00               # Time limit hrs:min:sec
#SBATCH --output=output.txt          # Standard output and error log
#SBATCH --error=error.txt            # Standard output and error log

source ~/envs/source_quip.sh

export DFTB_COMMAND="/mnt/scratch/chemistry-apps/dkb01416/dftbplus/installed/20.1/gnu-6.4.0/somp/bin/dftb+ > dftb.out"
export DFTB_SLAKO="/users/40247882/repository/DataBase/skf/parameterisation/PtCO/"

export VASP_COMMAND="mpirun -n 32 /mnt/scratch/chemistry-apps/dkb01416/vasp/installed/intel-2016/5.4.4/vasp_std"
export VASP_PP_PATH="/mnt/scratch/chemistry-apps/dkb01416/vasp/PseudoPotential"

export QUIP_COMMAND="quip"
export GAPFIT_EXEC="gap_fit"

export PYTHONPATH="$PYTHONPATH:/users/40247882/repository/accelerate-metadynamics"

echo `date "+%Y-%m-%d %H:%M:%S"` `pwd` >> $HOME/submitted
python -u ./run.py 2>&1 > acc.out
echo `date "+%Y-%m-%d %H:%M:%S"` `pwd` >> $HOME/finished
