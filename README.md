# Adversarial Robustness on MNIST

This project partially reproduces the Carlini & Wagner L2 attack on a convolutional neural network trained on MNIST. FGSM and PGD are included as additional comparisons.

## Setup

Python 3.13 is recommended. The experiments were run on Windows with an NVIDIA RTX 5070 Ti.

Extract the submitted ZIP file and open PowerShell inside the extracted project folder.

```powershell
py -3.13 -m venv .venv
```

For NVIDIA GPUs compatible with CUDA 12.8, install the CUDA build first:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only use, instead install:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux/macOS, use python3.13 to create the environment and .venv/bin/python in place of the Windows executable. Choose a compatible PyTorch build for your platform. CPU execution is supported but the full C&W experiment may be slow.

## Train the model

```powershell
.\.venv\Scripts\python.exe train.py
```

MNIST is downloaded automatically. Training runs for three epochs and saves weights to models/mnist_model.pth. If models/mnist_model.pth is included in the submission, skip training to use the original experimental weights. Otherwise, train once before running the evaluations. Training is not seeded, so exact results can vary.

## Run experiments

```powershell
.\.venv\Scripts\python.exe evaluate.py --attack fgsm
.\.venv\Scripts\python.exe evaluate.py --attack pgd
.\.venv\Scripts\python.exe test_cw.py --samples 10000 --steps 1000 --batch-size 1000 --search-steps 9
```

FGSM and PGD use epsilon 0.1. PGD uses 20 steps of size 0.01 without a random start. C&W uses the next digit modulo 10 as the target, initial c=1, and nine search rounds. Only initially correct images count toward attack success rates. Average distances are measured over successful attacks. Runtime includes the evaluation loop.

Fixed-c comparisons:

```powershell
.\.venv\Scripts\python.exe test_cw.py --samples 10000 --steps 1000 --batch-size 100 --c 1
.\.venv\Scripts\python.exe test_cw.py --samples 10000 --steps 10000 --batch-size 100 --c 1
```

For a quick test, use --samples 10. Reduce batch size if GPU memory is limited.

## Recorded results

Baseline test accuracy: 98.67%. Initially correct images: 9867.

| Attack | Successful | Success rate | Mean L2 | Mean Linf | Time (s) |
|---|---:|---:|---:|---:|---:|
| FGSM | 1535 | 15.56% | 2.1186 | 0.1000 | 6.5521 |
| PGD | 2999 | 30.39% | 1.8875 | 0.1000 | 4.1344 |
| C&W L2 (binary search) | 9867 | 100% | 2.1906 | Not measured | 340.3968 |

FGSM and PGD are untargeted, whereas C&W is targeted. These are single-run timings, not controlled speed benchmarks. The model architecture and experimental settings differ from the original paper.

## Demo

```powershell
.\.venv\Scripts\python.exe demo.py
.\.venv\Scripts\python.exe demo.py --successful
```

The first command saves results/demo.png using the first five correctly classified images. The second saves results/demo_successful.png using selected examples where all three attacks succeed. The selected examples are not representative of overall success rates.

## References

- Carlini and Wagner, Towards Evaluating the Robustness of Neural Networks (2017): https://arxiv.org/abs/1608.04644
- Goodfellow et al., Explaining and Harnessing Adversarial Examples (2015): https://arxiv.org/abs/1412.6572
- Madry et al., Towards Deep Learning Models Resistant to Adversarial Attacks (2018): https://arxiv.org/abs/1706.06083
