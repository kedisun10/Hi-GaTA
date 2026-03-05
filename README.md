# Hi-GaTA: Hierarchical Gated Temporal Aggregation Adapter
A deep learning framework for surgical report gengeration

## Project Structure

```
Hi-GaTA/
├── train_main/
│   ├── HPTA.py                 # Core hierarchical aggregation modules
│   ├── train_sur40k.py         # Training for Sur40k
│   ├── encode_video_sur40k.py  # Feature extraction and encoding
│   ├── eval.py                 # Evaluation script
│   ├── train_adapter.py        # Training for stage 1
│   ├── train_lora.py           # Training for stage 2
│   ├── models
│   └── results
└── Dataset/
    ├── label.csv
    ├── video_path.csv
    ├── train_feature
    └── test_feature            

```

## Requirements
python >= 3.11
torch == 2.7.1
torchaudio == 2.7.1
torchvision == 0.22.1
transformers == 4.57.0

