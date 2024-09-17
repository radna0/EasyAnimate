
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
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

