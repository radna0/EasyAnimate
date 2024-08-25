# Install PyTorch and PyTorch XLA.
export DATE="+20240808"
pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly${DATE}-cp310-cp310-linux_x86_64.whl
pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly${DATE}-cp310-cp310-linux_x86_64.whl

# Install JAX and Pallas.
pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

# Install other build dependencies.
pip install -r requirements-tpu.txt
