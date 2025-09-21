# Check if the .env file exists
if [ -f .env ]; then
    # Load environment variables from the .env file
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found!"
fi
# Set the working directory to the current directory
export WORKDIR=$(pwd)
# Add the working directory to the PYTHONPATH
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
python onemax_dac/train_ddqn.py -c onemax_dac/configs/onemax_n100_ddqn.yml -s 1 --n-cpus 10 --gamma 0.99 --out-dir outputs 
# python onemax_dac/train_ppo.py --setting-file onemax_dac/configs/onemax_n100_ppo_sc.yml --seed 1 -c 10 --out-dir outputs 
# python onemax_dac/hpo_ppo.py -m --config-name hpo_ppo_7
