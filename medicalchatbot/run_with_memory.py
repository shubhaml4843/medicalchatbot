import os
import gc
import torch

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Clear memory before starting
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Import and run the app
from app import app

if __name__ == "__main__":
    app.run(debug=True, port=8040)