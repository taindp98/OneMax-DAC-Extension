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
python onemax_dac/train.py \
    --problem_size 200 \
    --reward_choice scaling \
    --seed 1 \
    --max_steps 100000 \
    --num_workers 4