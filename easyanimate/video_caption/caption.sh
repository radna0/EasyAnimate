# install VILA (video-caption)
wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/package/vila-1.0.0-torch2.4.0-py3-none-any.whl
mv vila-1.0.0-torch2.4.0-py3-none-any.whl vila-1.0.0-py3-none-any.whl
pip install vila-1.0.0-py3-none-any.whl --extra-index-url https://download.pytorch.org/whl/cu118
wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/package/flash_attn-2.6.3%2Bcu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# install llm-awq (video-caption)
git clone https://github.com/mit-han-lab/llm-awq /root/llm-awq
cd /root/llm-awq
pip install -e .
cd /root/llm-awq/awq/kernels
# https://github.com/mit-han-lab/llm-awq/issues/93#issuecomment-2144434686
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0+PTX" && python3 setup.py install

# install vllm (video-caption)
pip install https://github.com/vllm-project/vllm/releases/download/v0.5.4/vllm-0.5.4+cu118-cp310-cp310-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118

# install requirements (video-caption)
cd /dev/
sudo cp easyanimate/video_caption/requirements.txt /dev/requirements-video_caption.txt
sudo pip install -r /root/requirements-video_caption.txt
sudo rm /root/requirements-video_caption.txt
