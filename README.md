# Efficient Track Anything Online with Multi Object
Run EfficientTAM on a live video stream. Additionally, it can track and segment multiple objects in a live video stream.

## Acknowledge
Thanks for their excellent works: [EfficientTAM](https://github.com/yformer/EfficientTAM) and [SAM2 real-time](https://github.com/Gy920/segment-anything-2-real-time).

## Demos
https://github.com/user-attachments/assets/8432b537-1bb5-4877-b292-4fb80c83918c

## Usage
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Download the checkpoints
```bash
cd checkpoints
./download_checkpoints.sh
```

3. Run the demo
```bash
python efficientTAM_realtime.py
```
