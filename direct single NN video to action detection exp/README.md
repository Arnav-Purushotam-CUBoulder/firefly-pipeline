# Direct single NN video → (x,y,w,h,t) detections (prototype)

This is an experimental prototype for *end-to-end* firefly detection that **learns temporal features directly from raw video**.

Instead of background subtraction + hand-tuned heuristics, this trains a single spatiotemporal neural network on:
- `video.mp4`
- `labels.csv` with columns `x,y,w,h,frame[,traj_id]` (same base format as `test1/tools/firefly flash annotation tool v2.py`)

At inference time it runs the model in a sliding-window over the video and writes a CSV of detections:
`x,y,w,h,frame` (where `frame` is your `t`).

## Data format
- CSV columns: `x,y,w,h,frame[,traj_id]` (0-based `frame` index)
- `x,y` are **top-left** pixel coordinates in the original video
- `traj_id` (optional) is a per-firefly trajectory id; if provided, the model is trained with an extra tracking objective so it must learn motion cues.

## Why sliding-window (clip) instead of whole-video input?
Whole-video end-to-end models are possible, but most practical systems train on fixed-length clips and slide across long videos for memory and data-efficiency. The *single* learned network still encodes temporal cues (flash rise/fall, noise dynamics) in its weights.

## Install deps (example)
Create an environment with PyTorch + OpenCV:

`pip install -r requirements.txt`

## Train
Example:

`python3 train_firefly_video_detector.py --video /path/to/video.mp4 --csv /path/to/gt.csv --out_dir runs/firefly_video_centernet`

## Inference
Example:

`python3 infer_firefly_video_detector.py --video /path/to/video.mp4 --ckpt runs/firefly_video_centernet/checkpoints/best.pt --out_csv pred.csv`

To also output a linked trajectory id:

`python3 infer_firefly_video_detector.py --video /path/to/video.mp4 --ckpt runs/firefly_video_centernet/checkpoints/best.pt --out_csv pred.csv --emit_traj_id`

## Notes / limitations (prototype)
- This is a **single-class** detector (firefly vs background).
- Output is per-frame detections (you can post-process into trajectories later).
- The model output stride is currently fixed at `4`, so `--resize_h/--resize_w` should be divisible by 4.
- Training from a single video can work for a demo, but generalization improves a lot with more varied videos.
