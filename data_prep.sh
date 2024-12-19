### PREP DATA PROCESSING

export DEBIAN_FRONTEND=noninteractive

site_pkg_path=$(python3.10 -c 'import site; print(site.getsitepackages()[0])')

## Transformers QWEN-VL

pip install git+https://github.com/huggingface/transformers

sudo wget -O $site_pkg_path/transformers/models/qwen2_vl/modeling_qwen2_vl.py https://raw.githubusercontent.com/radna0/EasyAnimate/refs/heads/TPU/qwen/modeling_qwen2_vl.py




## Accelerate TPU all gather implementation & LAUNCHER & CONFIG

pip install git+https://github.com/huggingface/accelerate

sudo wget -O $site_pkg_path/accelerate/utils/operations.py https://raw.githubusercontent.com/radna0/EasyAnimate/refs/heads/TPU/accelerate/operations.py


sudo wget -O $site_pkg_path/accelerate/commands/launch.py https://raw.githubusercontent.com/radna0/EasyAnimate/refs/heads/TPU/accelerate/launch.py


echo 'export HF_HOME=/dev/shm' >> ~/.bashrc && source ~/.bashrc
accelerate config default --config_file /dev/shm/accelerate/default_config.yaml

sudo wget -O /dev/shm/accelerate/default_config.yaml https://raw.githubusercontent.com/radna0/EasyAnimate/refs/heads/TPU/accelerate/config.yaml




## Torch XLA 

sudo wget -O $site_pkg_path/torch_xla/core/xla_model.py https://raw.githubusercontent.com/radna0/EasyAnimate/refs/heads/TPU/torch_xla/xla_model.py
