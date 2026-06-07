which -a nvcc; echo "PATH=$PATH"
N=$(readlink -f "$(which nvcc)"); ls -l "$N"; stat -c '%A %U:%G' "$N"; id
findmnt -T "$N" -o TARGET,SOURCE,FSTYPE,OPTIONS    # noexec? foreign SOURCE?
env | grep -iE 'CUDA_HOME|CUDA_PATH|MODULEPATH|LMOD'
nvcc --version      
