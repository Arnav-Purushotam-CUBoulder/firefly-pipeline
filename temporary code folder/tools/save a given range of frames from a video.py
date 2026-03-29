#!/usr/bin/env python3
"""
extract_frames.py

Given an input video, prints the total number of frames and saves a specified
range of frames as PNG images to a given output directory.
"""

import cv2
import os

# ─── GLOBAL CONSTANTS ──────────────────────────────────────────
INPUT_VIDEO_PATH  = r'/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/raw videos/20240606_cam1_GS010064.mp4'
OUTPUT_FRAMES_DIR = r'/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/raw data from drive/pyrallis gopro data/raw frames/20240606_cam1_GS010064'
START_FRAME       = 0      # first frame index to save (inclusive)
END_FRAME         = 5000      # last frame index to save (inclusive)

def main():
    # Open the video file
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: cannot open video file {INPUT_VIDEO_PATH}")
        return

    # Get and print total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames in video: {total_frames}")

    # Ensure output directory exists
    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

    current_frame = 0
    saved_count = 0

    # Read through the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # no more frames

        # If within the desired range, save as PNG
        if START_FRAME <= current_frame <= END_FRAME:
            filename = f"frame_{current_frame:06d}.png"
            output_path = os.path.join(OUTPUT_FRAMES_DIR, filename)
            cv2.imwrite(output_path, frame)
            saved_count += 1

        current_frame += 1

        # Stop once we've processed past END_FRAME
        if current_frame > END_FRAME:
            break

    cap.release()
    print(f"Saved {saved_count} frames (from {START_FRAME} to {END_FRAME}) into:")
    print(f"  {OUTPUT_FRAMES_DIR}")

if __name__ == "__main__":
    main()
