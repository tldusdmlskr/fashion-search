# Fashion Attribute Classifier — Demo

Fashion attribute classifier inference demo.  
Enter an image URL → get predicted tags + simulated Qdrant payload in the terminal.

## What it does

```
Image URL
    ↓
marqo-fashionSigLIP (shared backbone)
    ↓
┌─────────────────────────────────────────┐
│  Pattern Position  →  multi-label       │
│  Pattern Size      →  4 classes         │
│  Trim              →  5 classes         │
│  Bottom Length     →  8 classes (opt.)  │
│  Waist Rise        →  3 classes (opt.)  │
└─────────────────────────────────────────┘
    ↓
Console output + simulated Qdrant payload
```

## Setup

```bash
pip install open-clip-torch requests pillow torch
```

Model files required (not included in repo — download separately):

```
fashion_classifier_v4.pt      # main classifier
fashion_classifier_bottom.pt  # bottom classifier
```

Update paths in `demo_inference.py`:

```python
V4_MODEL_PATH     = r"C:\...\fashion_classifier_v4.pt"
BOTTOM_MODEL_PATH = r"C:\...\fashion_classifier_bottom.pt"
```

## Usage

**Interactive mode** — prompts for URL each run:

```bash
python demo_inference.py
```

**Single URL:**

```bash
python demo_inference.py --url https://example.com/image.jpg
```

**Bottom garment:**

```bash
python demo_inference.py --url https://example.com/pants.jpg --bottom
```

## Example output

```
────────────────────────────────────────────────────
  ✦ Classifier Output
────────────────────────────────────────────────────

  Image ID  869118

  ┌ Attributes ──────────────────────────────┐
  │  Pattern Position  allover
  │  Pattern Size      medium
  │  Trim              plain
  └──────────────────────────────────────────┘

  ┌ Confidence ──────────────────────────────┐
  │  Pattern pos   [████████████░░░░░░░░] 87.3%
  │  Pattern size  [██████████░░░░░░░░░░] 79.1%
  │  Trim          [████████████████░░░░] 91.2%
  └──────────────────────────────────────────┘

  ┌ Qdrant Payload (simulated) ──────────────┐
  │  {
  │      "pattern_position": ["allover"],
  │      "pattern_size": "medium",
  │      "trim": "plain"
  │  }
  └──────────────────────────────────────────┘

  ✔ Payload saved to Qdrant vector DB
```

## Model performance

| Classifier | Metric | Score |
|------------|--------|-------|
| Pattern Position | micro F1 | 0.79 |
| Pattern Size | accuracy | 80% |
| Trim | accuracy | 81.7% |
| Bottom Length | accuracy | 73% |
| Bottom Waist Rise | accuracy | 77% |