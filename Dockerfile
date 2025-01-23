# Use the official miniconda3 base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Prevent tzdata from prompting for geographic area selection
ENV DEBIAN_FRONTEND=noninteractive

# Install git, OpenGL libraries, and other dependencies
RUN apt-get update && apt-get install -y git

RUN apt-get update && apt-get install -y \
    texlive \
    texlive-latex-extra \
    dvipng \
    cm-super \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a new Conda environment with Python 3.9
RUN conda create -n dacbench python=3.10 -y

# Install pip in the Conda environment
RUN conda run -n dacbench pip install --upgrade pip

# Set the environment path to use the newly created Conda environment
ENV PATH /opt/conda/envs/dacbench/bin:$PATH

# Copy the dacbench.txt file into the container at /app/dacbench.txt
COPY requirements.txt /app/

# Install the dependencies from dacbench.txt into the Conda environment
RUN conda run -n dacbench pip install --no-cache-dir -r /app/requirements.txt
 
# Install PyTorch with the specified CUDA and CUDNN version
# RUN conda run -n dacbench pip install torch==2.1.0+cu121 \
#     torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
#     -f https://download.pytorch.org/whl/torch_stable.html

# Use conda run to ensure commands run with the conda environment
SHELL ["conda", "run", "-n", "dacbench", "/bin/bash", "-c"]

# Default command to run when the container starts
CMD ["bash"]