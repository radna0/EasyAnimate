### PREP DATA PROCESSING

export DEBIAN_FRONTEND=noninteractive

site_pkg_path=$(python3.10 -c 'import site; print(site.getsitepackages()[0])')

## Transformers QWEN-VL

pip install git+https://github.com/huggingface/transformers

cd $site_pkg_path/transformers/models/qwen2_vl/
sudo rm -rf modeling_qwen2_vl.py
sudo wget -O modeling_qwen2_vl.py https://raw.githubusercontent.com/radna0/EasyAnimate/refs/heads/TPU/qwen/modeling_qwen2_vl.py


## Accelerate TPU all gather implementation & LAUNCHER & CONFIG

pip install git+https://github.com/huggingface/accelerate

cd $site_pkg_path/accelerate/utils/
sudo rm -rf operations.py
sudo wget -O operations.py https://raw.githubusercontent.com/radna0/EasyAnimate/refs/heads/TPU/accelerate/operations.py

cd $site_pkg_path/accelerate/commands/
sudo rm -rf launch.py
sudo wget -O launch.py https://raw.githubusercontent.com/radna0/EasyAnimate/refs/heads/TPU/accelerate/launch.py

accelerate config default --config_file /dev/shm/accelerate/default_config.yaml
cd /dev/shm/accelerate/
rm default_config.yaml
wget -O default_config.yaml https://raw.githubusercontent.com/radna0/EasyAnimate/refs/heads/TPU/accelerate/config.yaml


## Torch XLA 

cd $site_pkg_path/torch_xla/core/xla_model.py
sudo rm -rf xla_model.py
wget -O xla_model.py https://raw.githubusercontent.com/radna0/EasyAnimate/refs/heads/TPU/torch_xla/xla_model.py
