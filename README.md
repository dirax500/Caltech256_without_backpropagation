# SteamPunkDeep – Forward-Forward Network for Caltech-256

## Algorithm

The algorithm uses training model for Caltech-256 dataset by applying **Hinton's Forward-Forward Algorithm (2022)** with no global backpropagation through the feature layers. Instead, it uses:

1. Each conv layer trains itself with a **local binary loss on "goodness"**
   - Goodness = mean of squared post-ReLU activations
   - Positive data (correct label embedded) → push goodness **above** threshold θ
   - Negative data (wrong label embedded) → push goodness **below** θ
2. After FF training, a lightweight **classifier head** is trained on frozen features
   (backprop is used here, but only through the 2-layer MLP head)

## Architecture

```
Input 3×96×96
  └─ FFConvLayer  3→64   + AvgPool → 64×48×48
  └─ FFConvLayer 64→128  + AvgPool → 128×24×24
  └─ FFConvLayer 128→256 + AvgPool → 256×12×12
  └─ FFConvLayer 256→512 + AvgPool → 512×6×6
  └─ Flatten → 18432-d
  └─ Linear(18432→1024) → ReLU → Dropout → Linear(1024→257)
```

## Setup

```bash
pip install torch torchvision
```

## Dataset structure (ImageFolder format)

```
caltech256/
  001.ak47/
    001_0001.jpg
    ...
  002.american-flag/
    ...
  257.clutter/
    ...
```

Download: https://data.caltech.edu/records/nyy15-4j048

## Training (M1 Mac → auto-uses MPS)

```bash
python ff_caltech256.py \
  --data_dir   ./caltech256 \
  --epochs     30 \
  --cls_epochs 50 \
  --theta      2.0 \
  --batch_size 128
```

## Training on external CUDA server

```bash
python ff_caltech256.py \
  --data_dir   /path/to/caltech256 \
  --epochs     40 \
  --cls_epochs 60 \
  --batch_size 256 \
  --ff_lr      5e-4
```

The script auto-detects MPS → CUDA → CPU in that priority order.

## Evaluation only

```bash
python ff_caltech256.py \
  --data_dir  ./caltech256 \
  --eval_only \
  --model_path ./checkpoints/best_ff_model.pt
```

## Expected performance

| Setup                  | Val Top-1 | Val Top-5 |
|------------------------|-----------|-----------|
| 30 FF + 50 CLS epochs  | ~30–38%   | ~55–65%   |
| 40 FF + 60 CLS epochs  | ~35–43%   | ~60–70%   |

Caltech-256 is hard: even standard backprop CNNs of similar size reach ~55–65%.
FF networks trade some accuracy for biological plausibility.

## Key hyperparameters

| Param        | Default | Notes                                        |
|--------------|---------|----------------------------------------------|
| `--theta`    | 2.0     | Goodness threshold. Try 1.5–3.0              |
| `--ff_lr`    | 3e-4    | Per-layer Adam LR. Try 1e-4 on larger GPU    |
| `--epochs`   | 30      | FF training epochs. More = better features   |
| `--cls_epochs`| 50     | Head training epochs. Uses cosine annealing  |
| `--batch_size`| 128    | Reduce to 64 if OOM on M1                    |

## Saved model format

```python
checkpoint = {
    "epoch":       int,
    "model_state": OrderedDict,   # load with model.load_state_dict(...)
    "val_top1":    float,
    "val_top5":    float,
    "args":        dict,
}
```

Load for submission:
```python
model = ForwardForwardNet(num_classes=257)
ckpt  = torch.load("best_ff_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()
```
## Authors
Didar Rakhimbay, Beibars Ybraiakhyn, Alim Daniyarov, Dias Akimbay

Nazarbayev University MSc in Robotics students

## License

MIT — see [LICENSE](LICENSE).
