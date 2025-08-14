
## Installation

First, clone the repository:

```bash
git clone ...
```

Then set up the submodules:

```bash
git submodule update --init --recursive
```

Create venv

```bash
python3.10 -m venv venv
```

Add following to `venv/bin/activate`
```
module load OpenSSL
module load libffi
export HF_TOKEN=...
export HF_HOME=...
export WANDB_DIR=...
```

Install env

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Run with
```bash
python train.py text/train.py --batch_size=32 --compile=False --model_type=aoGPT/sigmaGPT # one GPU
torchrun --standalone --nproc_per_node=4 text/train.py # 4 GPU
```

Possibly `sbatch` with `run.py`
