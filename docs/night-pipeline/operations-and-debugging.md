# Night Pipeline Operations and Debugging

## Preferred Run Path

Use the gateway for one-off videos:

```bash
python3 "Pipelines/Pipeline Gateway/gateway.py" \
  --input "/path/to/night_video.mp4" \
  --output-root "/mnt/Samsung_SSD_2TB/temp to delete/pipeline gateway inference output data" \
  --route-override night \
  --force-tests \
  --max-concurrent 1
```

Use direct orchestration when you need to control every stage toggle in `pipeline_params.py`.

## Direct Run Path

1. Put the video here:

```text
/mnt/Samsung_SSD_2TB/temp to delete/night time pipeline inference output data/original videos
```

2. Put GT here:

```text
/mnt/Samsung_SSD_2TB/temp to delete/night time pipeline inference output data/ground truth/gt.csv
```

3. Run:

```bash
python3 "Pipelines/night_time_pipeline/orchestrator.py"
```

## Minimal Smoke Test

For a fast smoke test:

1. Set `MAX_FRAMES = 300`.
2. Consider setting `STAGE1_VARIANT = "cc_cpu"` if GPU dependencies are uncertain.
3. Disable heavy renders/analysis if needed:
   - `RUN_STAGE10 = False`
   - `RUN_STAGE11 = False`
   - `RUN_STAGE12 = False`
   - `RUN_STAGE14 = False`
4. Run the orchestrator.
5. Confirm these exist:

```text
csv files/<stem>.csv
csv files/<stem>_fireflies_logits.csv
original 10px overlay annotated videos/<stem>_orig_10px.mp4
stage9 validation/<stem>/
```

## Output Review Order

When debugging a video, inspect in this order:

1. Stage 1 CSV row count: did detection find candidates?
2. Stage 2/3 row counts: did recenter or area filtering remove everything?
3. Stage 4 class/logit columns: did CNN classify expected flashes as firefly?
4. Stage 6 fixed 10px render: are candidates visually centered?
5. Stage 8 crops: are final centroids correct?
6. Stage 8.5 row count: did area pruning remove real flashes?
7. Stage 8.6 artifacts: did neighbor hunt add valid detections or noise?
8. Stage 8.7 replacements: did large-flash repair help or over-expand?
9. Stage 9 normalized GT: did GT convert and offset correctly?
10. Stage 10 overlay: do model/GT align spatially and temporally?

## Common Failures

### No videos found

Check:

```text
ROOT/original videos
```

and `VIDEO_EXTS`.

Gateway runs should check `--input`.

### cuCIM or CuPy import/runtime failure

Symptoms:

- ImportError from `stage1_detect_cucim.py`.
- CUDA/NVRTC/Jitify errors.
- GPU memory errors.

Actions:

1. Switch to `STAGE1_VARIANT = "cc_cpu"` for a smoke test.
2. Try `STAGE1_VARIANT = "cc_cuda"` if CuPy works but cuCIM blob functions fail.
3. Reduce `CUCIM_BATCH_SIZE`.
4. Confirm CUDA, CuPy, and cuCIM versions are compatible.

### Stage 1 finds too many candidates

Actions:

- Raise `CUCIM_THRESHOLD` or `CC_FIXED_THRESHOLD`.
- Increase area thresholds.
- Disable or tune CLAHE/top-hat/DoG preprocessing.
- Inspect early render videos before tuning CNN.

### Stage 1 misses dim flashes

Actions:

- Lower detection threshold.
- Lower min area.
- Enable or tune CLAHE.
- Check source video brightness and compression.

### CNN removes real fireflies

Check:

- `CNN_MODEL_PATH` and route correctness.
- `FIREFLY_CONF_THRESH`.
- `IMAGENET_NORMALIZE`.
- Whether the model was trained for the species/route.

If `DROP_BACKGROUND_ROWS=False`, background rows remain but validation filters by class.

### Final CSV coordinates look off

Check:

- Stage 8 crops.
- `STAGE8_GAUSSIAN_SIGMA`.
- Whether consumers understand `xy_semantics=center`.
- Whether Stage 8.7 replaced boxes with larger squares.

### GT validation fails with schema error

Night Stage 9 expects `x,y,t`. If GT is `x,y,w,h,frame`, Stage 8.9 must run first.

If Stage 8.9 skipped unexpectedly, inspect the GT header.

### Many FNs and FPs with obvious temporal shift

Check:

- `GT_T_OFFSET`.
- Whether Stage 8.9 wrote raw frame numbers to `t`.
- Whether Stage 9 subtracted the same offset.
- Gateway logs for inferred offset and max frames.

### GT points are spatially shifted

Likely cause:

- Annotation CSV `x,y` semantics do not match Stage 8.9 expectations.

Action:

- Open raw annotation CSV and overlay manually.
- Determine whether `x,y` are top-left or center.
- Convert before validation if needed.

### Stage 8.6 adds noise

Actions:

- Set `RUN_STAGE8_6=False` for comparison.
- Reduce `STAGE8_6_RUNS`.
- Increase `STAGE8_6_DEDUPE_PX`.
- Tighten Stage 1 thresholds used by Stage 8.6.

### Stage 8.7 over-expands large flashes

Actions:

- Raise `STAGE8_7_INTENSITY_THR`.
- Raise `STAGE8_7_MIN_SQUARE_AREA_PX`.
- Inspect replacement crops under `stage8.7/<stem>/replacements`.

## Safe Tuning Loop

1. Use one representative video and a short frame cap.
2. Make one parameter change at a time.
3. Compare row counts after Stage 1, 3, 4, 7, 8, 8.5, 8.6, and 8.7.
4. Render Stage 6 and Stage 10 overlays.
5. Keep final tuning notes with:
   - video name,
   - model path,
   - Stage 1 variant,
   - thresholds,
   - validation metrics.

