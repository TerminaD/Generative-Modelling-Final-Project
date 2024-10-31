# Generative Models HW 1
## Environment
Two versions of training source code are provided. `train.ipynb` runs in a Google Colab environment, while `train.py`, as well as the evaluation code, `evaluate.py`, runs in a Conda environment created as so:
```bash
conda create -n genmod_lang python=3.11
conda activate genmod_lang
pip install transformers datasets evaluate
pip install accelerate -U
```

## How to Run
### Training
For `train.ipynb`, simply run it in a Google Colab environment!

As for `train.py`, after you've set up the conda environment specified above, simply run it with
```bash
python3 codes/train.py
```

A note: please run the code with the **main directory** (i.e., the one containing `ckpts`, `codes`, etc.) **as the current working directory**. Otherwise, the paths gets confused!

### Evaluate
Change the prompt as you desire here:
```python
prompt = "Two households, both alike in dignity,"	# Set your prompt here
```

And run it with:
```bash
python3 codes/evaluate.py
```