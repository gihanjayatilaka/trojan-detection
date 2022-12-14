# Environments

1. cmsc828
2. cmsc828a
3. cmsc828b
4. cmsc828c -- This is from Punky. This is gonna be hugeeeee.


VULCAN SUB
keras
# kerasOld
conda create -n kerasOld python=2.7 -y
conda activate kerasOld
pip install setuptools==40
pip install wheel && GRPC_BUILD_WITH_BORING_SSL_ASM="" GRPC_PYTHON_BUILD_SYSTEM_RE2=true GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=true GRPC_PYTHON_BUILD_SYSTEM_ZLIB=true pip install grpcio==1.39
pip install imageio==2.5
conda install -c conda-forge cudatoolkit=9 cudnn=7
pip install tensorflow==1.12.0




watch -n0.1 nvidia-smi

cd /fs/class-projects/fall2022/cmsc828w/c828w018/trojan-detection/gihan/
srun --pty --partition=class --account=class --qos=default --gres=gpu:1 --time=4:00:00 bash
conda activate keras
jupyter notebook --no-browser --port=8889 --ip=0.0.0.0
ssh -N -f -L localhost:8888:vulcan31:8889 c828w018@nexusclass00.umiacs.umd.edu





cd /vulcanscratch/gihan/trojan
srun --pty --account=abhinav --qos=default --gres=gpu:1 --time=4:00:00 bash
conda activate keras
jupyter notebook --no-browser --port=8889 --ip=0.0.0.0
ssh -N -f -L localhost:8888:vulcan31:8889 gihan@vulcansub00.umiacs.umd.edu



conda create -n keras python=3.7 -y
conda activate keras
conda install -c conda-forge htop -y
conda install -c conda-forge cudatoolkit=11.7 cudnn=8.2.1 -y
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
pip install --upgrade pip
conda install -c nvidia cuda-nvcc
pip install tensorflow
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
pip install matplotlib
conda install -c conda-forge scikit-learn



<!-- TMUX -->

tmux
CTRLb "
CTRLb (up)

source activate keras

<!-- CUDA -->


module load cuda