# GAP-ST: Protein-Induced Group-Aware MoE for
High-Dimensional Gene Expression Prediction from Histology

This is our PyTorch implementation for the paper:


## Usage

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-enabled GPU (recommended)

Download HEST benchmark datasets and pretrained weights of UNI2-h models using the following script:
```
from huggingface_hub import snapshot_download, hf_hub_download

source_dataroot = ""./dataset/"
weights_root = "./dataset/weights_root"

snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type='dataset', local_dir=weights_root, allow_patterns=['fm_v1/*'])
snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type='dataset', local_dir=source_dataroot, ignore_patterns=['fm_v1/*'])
hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=os.path.join(weights_root, "uni/"))
hf_hub_download("prov-gigapath/prov-gigapath", filename="pytorch_model.bin", local_dir=os.path.join(weights_root, "gigapath/"))
```

Testing foundation models with the following script
```
$ python /hest/benchmark.py \
        --datasets all \
        --encoders uni_v2 \
        --weights_root /path/to/weights_root \
        --source_dataroot /path/to/source_dataroot \
        --embed_dataroot /path/to/embed_dataroot \
        --batch_size 128
```

Training GAP-ST with the following script:
```
$ python train.py
```

