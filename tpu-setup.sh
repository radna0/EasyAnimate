cd ~
export DEBIAN_FRONTEND=noninteractive

sudo apt-get update -y
sudo apt-get install software-properties-common -y

DEBIAN_FRONTEND=noninteractive sudo add-apt-repository ppa:deadsnakes/ppa -y


sudo apt install python3.10 python3.10-venv python3.10-dev -y
python3.10 --version

sudo curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.10

sudo pip uninstall -y tensorflow tensorflow-cpu

sudo pip install accelerate diffusers transformers loguru peft

sudo apt-get install -y libgl1 libglib2.0-0 google-perftools



# EasyAnimate
rm -rf EasyAnimate

sudo git clone -b TPU https://github.com/radna0/EasyAnimate.git
sudo chmod -R 777 EasyAnimate

# enter EasyAnimate's dir
cd EasyAnimate

# download weights
mkdir models/Diffusion_Transformer
mkdir models/Motion_Module
mkdir models/Personalized_Model

wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/EasyAnimateV4-XL-2-InP.tar.gz -O models/Diffusion_Transformer/EasyAnimateV4-XL-2-InP.tar.gz

cd models/Diffusion_Transformer/
tar -zxvf EasyAnimateV4-XL-2-InP.tar.gz
rm EasyAnimateV4-XL-2-InP.tar.gz

cd ../../


# Video Caption
pip install -r requirements.txt

cd easyanimate/video_caption && pip install -r requirements.txt --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

site_pkg_path=$(python3.10 -c 'import site; print(site.getsitepackages()[0])')
cp -v easyocr_detection_patched.py $site_pkg_path/easyocr/detection.py

sudo apt install -y ffmpeg   

sudo pip install --upgrade accelerate

# TPU all gather implementation
cd $site_pkg_path/accelerate/utils/
sudo rm -rf operations.py
sudo wget -O operations.py https://raw.githubusercontent.com/radna0/EasyAnimate/TPU/accelerate/operations.py

#Accelerate config
cd ~/.cache/huggingface/accelerate/
wget -O default_config.yaml https://raw.githubusercontent.com/radna0/EasyAnimate/TPU/accelerate/config.yaml

# Pytorch XLA
pip uninstall torch torchvision -y
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl' -f https://storage.googleapis.com/libtpu-releases/index.html


# Xformers
cd ~
pip install ninja
sudo python3.10 -m pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers


# Cloud Storage Fuse
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update 
sudo apt-get -y install gcsfuse
sudo apt-get -y upgrade gcsfuse

# Mount Cloud Storage Bucket
gcloud auth activate-service-account --key-file=service_account.json

gcloud config set project easyanimate-431707
