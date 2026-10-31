"""
Duplicate frame handler: freezes each tracked object's masks across runs of
frames where that object's source image is held (anime drawn on 2s/3s, or held
for many frames), so tracker wobble on unchanged footage doesn't produce a
different mask every frame.

Method: each frame is reduced to a grid of block-mean luma values (area
averaging cancels grain). Per object (= mask file name), only blocks fully
inside that object's mask are compared, so a moving background and mask wobble
can't register. A frame is a "hold" of a group's first frame (the anchor) if no
block differs by more than `hi` and fewer than FRAC of the blocks differ by more
than `lo` (the hi/lo/frac idea from ffmpeg's mpdecimate). `hi` is calibrated per
video from the footage's own noise level. Each group's masks are replaced with
their per-pixel median.
"""

import cv2
import numpy as np
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog, QApplication

# Resolve absolute path of file back to project root folder
utils_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(utils_dir, ".."))

frames_dir = os.path.join(project_root, "temp", "frames")
mask_dir = os.path.join(project_root, "temp", "masks")
backup_dir = os.path.join(project_root, "temp", "masks_backup")

BLOCK_PX_1080 = 8        # block edge in pixels at 1080p; scaled to the actual height
LO_RATIO = 0.4           # lo = hi * LO_RATIO (mpdecimate's defaults are ~0.42)
FRAC = 0.25              # fraction of blocks over `lo` that counts as a widespread change
MIN_INTERIOR_BLOCKS = 4  # fewer fully-inside blocks than this: object is "small" (union fallback)
FULL_COVER = 250         # block coverage (0-255) that counts as fully inside the mask (~98%)
DEFAULT_HI = 5.0         # fallback `hi` (gray levels of block-mean difference)
MIN_HI = 2.0
NOISE_PERCENTILE = 97    # percentile of the hold-score cluster taken as the noise ceiling
NOISE_MARGIN = 1.5       # hi = noise ceiling * this margin
HOLD_CAP_MULTIPLE = 3.0  # auto hi never exceeds this x the median hold score (a larger gap means
                         # small real changes were counted as noise); relative, so film grain is fine
MAX_HI = 40.0            # absolute backstop for the auto hi (before `sensitivity`)
SCORE_FLOOR = 0.5        # scores below this are treated as equal (perfect holds)
MEDIAN_MAX_FRAMES = 15   # cap on frames read when building a median mask


# ---------------------------------------------------------------------------
# Reading frames and masks
# ---------------------------------------------------------------------------

def load_frame_entry(frame_number):
    """Reduces one frame and its masks to {"blocks": float16 grid of block-mean
    luma, "objects": {matte file name: uint8 grid of that object's coverage of
    each block (0-255)}}. Objects with an empty mask are left out. Returns None
    if the frame has no masks (checked before the expensive PNG decode) or its
    image can't be read."""
    mask_folder = os.path.join(mask_dir, frame_number)
    mattes = sorted(os.listdir(mask_folder)) if os.path.isdir(mask_folder) else []
    if not mattes:
        return None
    frame = cv2.imread(os.path.join(frames_dir, frame_number + ".png"), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return None

    block_px = max(2, round(BLOCK_PX_1080 * frame.shape[0] / 1080))
    grid = (max(1, frame.shape[1] // block_px), max(1, frame.shape[0] // block_px))   # (w, h)
    blocks = cv2.resize(frame.astype(np.float32), grid, interpolation=cv2.INTER_AREA).astype(np.float16)

    objects = {}
    for matte in mattes:
        image = cv2.imread(os.path.join(mask_folder, matte), cv2.IMREAD_GRAYSCALE)
        if image is None or not (image > 127).any():
            continue                                  # unreadable, or the object is absent here
        coverage = cv2.resize((image > 127).astype(np.float32), grid, interpolation=cv2.INTER_AREA)
        objects[matte] = np.ceil(coverage * 255).astype(np.uint8)   # ceil: one stray pixel still marks its block
    return {"blocks": blocks, "objects": objects}


# ---------------------------------------------------------------------------
# Comparison (one object at a time)
# ---------------------------------------------------------------------------

def comparison_region(cover_a, cover_b):
    """Blocks to compare for one object on two frames, as (region, small).
    Normally the blocks fully inside BOTH masks, eroded by one block: none
    contains background, and mask wobble at the edge is ignored (but changes
    confined to the outermost band of the silhouette are invisible). An object
    with too few such blocks is "small": the blocks touched by EITHER mask,
    dilated by one, are used instead (background can leak in there)."""
    kernel = np.ones((3, 3), np.uint8)
    inside = cv2.erode(((cover_a >= FULL_COVER) & (cover_b >= FULL_COVER)).astype(np.uint8), kernel)
    if int(inside.sum()) >= MIN_INTERIOR_BLOCKS:
        return inside > 0, False
    return cv2.dilate(((cover_a > 0) | (cover_b > 0)).astype(np.uint8), kernel) > 0, True


def block_diffs(a, b, region):
    """a, b: (blocks, cover) tuples for the same object on two frames."""
    return np.abs(a[0][region].astype(np.float32) - b[0][region].astype(np.float32))


def find_hold_groups(timeline, hi, trace=None):
    """timeline: one (blocks, cover) tuple per frame, None where the object has
    no mask (a hard break). Returns groups of frame indices (2+ members) over
    which the object's image is held. Each group starts at an anchor and extends
    while frames stay a hold of that anchor; the first frame that fails becomes
    the next anchor. trace, if given, gets (anchor, candidate, decision,
    max_diff, frac_over_lo) tuples."""
    lo, groups, i = hi * LO_RATIO, [], 0
    while i < len(timeline):
        anchor = timeline[i]
        if anchor is None:
            i += 1
            continue
        members, j = [i], i + 1
        while j < len(timeline):
            if timeline[j] is None:
                if trace is not None:
                    trace.append((i, j, "object_missing_break", None, None))
                break
            d = block_diffs(anchor, timeline[j], comparison_region(anchor[1], timeline[j][1])[0])
            max_diff, frac_over_lo = float(d.max()), float((d > lo).mean())
            decision = ("rejected_block_over_hi" if max_diff > hi else
                        "rejected_widespread_over_lo" if frac_over_lo > FRAC else "grouped")
            if trace is not None:
                trace.append((i, j, decision, max_diff, frac_over_lo))
            if decision != "grouped":
                break
            members.append(j)
            j += 1
        if len(members) > 1:
            groups.append(members)
        i = j                                         # first frame that didn't join (or the end)
    return groups


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def object_pair_scores(timeline):
    """Worst-block difference for every adjacent pair of frames where the object
    is present on both. Pairs for "small" objects (no solid interior, so the
    union fallback would leak background in) are left out so they can't skew
    calibration. Returns (scores, number_of_pairs_left_out)."""
    scores, small = [], 0
    for prev, cur in zip(timeline, timeline[1:]):
        if prev is None or cur is None:
            continue
        region, is_small = comparison_region(prev[1], cur[1])
        if is_small:
            small += 1
        else:
            scores.append(float(block_diffs(prev, cur, region).max()))
    return np.array(scores, dtype=np.float64), small


def calibrate_hi(scores, sensitivity=1.0):
    """Adaptive `hi` from adjacent-pair scores, which form two clusters: holds
    (noise floor) and real changes (much larger). Split them with 1-D 2-means in
    log space, then set hi just above the noise cluster's upper tail so small
    real changes (a blink) still register. Falls back to DEFAULT_HI if the
    clusters aren't clearly separated (too few pairs, or e.g. footage animated
    on 1s). Capped at HOLD_CAP_MULTIPLE x the median hold score and at MAX_HI,
    then divided by `sensitivity`. Returns (hi, info)."""
    info = {"method": "fallback", "scores": int(len(scores))}
    hi = DEFAULT_HI

    if len(scores) >= 10:
        x = np.log(np.maximum(scores, SCORE_FLOOR))
        t = float(x.mean())
        for _ in range(100):
            low, high = x[x <= t], x[x > t]
            if len(low) == 0 or len(high) == 0:
                break
            t_new = (low.mean() + high.mean()) / 2.0
            if abs(t_new - t) < 1e-5:
                t = t_new
                break
            t = t_new
        low, high = x[x <= t], x[x > t]
        min_cluster = max(3, int(0.02 * len(x)))
        if len(low) >= min_cluster and len(high) >= min_cluster:
            separation = float(np.exp(high.mean() - low.mean()))
            info["separation"] = separation
            if separation >= 3.0:
                hold_scores = scores[x <= t]
                noise_ceiling = float(np.percentile(hold_scores, NOISE_PERCENTILE))
                split = float(np.exp(t))
                hi = float(np.clip(noise_ceiling * NOISE_MARGIN, MIN_HI, split))
                hold_median = float(np.median(hold_scores))
                info.update({"method": "auto", "noise_ceiling": noise_ceiling, "split": split,
                             "hold_median": hold_median, "hold_pairs": int(len(hold_scores)),
                             "change_pairs": int(len(x) - len(hold_scores))})
                relative_cap = max(MIN_HI, HOLD_CAP_MULTIPLE * hold_median)
                if hi > relative_cap:
                    info["capped_from"], info["cap"] = hi, f"{HOLD_CAP_MULTIPLE:g}x median hold score"
                    hi = relative_cap

    if hi > MAX_HI:
        info.setdefault("capped_from", hi)
        info["cap"] = "absolute maximum"
        hi = MAX_HI                                   # caps apply first, so `sensitivity` can still move it

    hi = max(MIN_HI * 0.5, hi / max(sensitivity, 1e-3))
    info["hi"] = hi
    return hi, info


# ---------------------------------------------------------------------------
# Mask file replacement / backup
# ---------------------------------------------------------------------------

def replace_object_masks(name, group_frames):
    """Makes one object's mask file identical across every frame in the group
    (other objects' files are untouched): the per-pixel median of the group's
    mattes, which averages out tracker wobble instead of trusting one frame.
    Needs 3+ frames of plain 8-bit single-channel mattes; otherwise the first
    frame's file is copied."""
    paths = [p for p in (os.path.join(mask_dir, f, name) for f in group_frames) if os.path.isfile(p)]
    if len(paths) < 2:
        return

    if len(paths) >= 3:
        sample = paths if len(paths) <= MEDIAN_MAX_FRAMES else [
            paths[i] for i in np.linspace(0, len(paths) - 1, MEDIAN_MAX_FRAMES).astype(int)]
        imgs = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in sample]
        imgs = [m for m in imgs if m is not None and m.ndim == 2 and m.dtype == np.uint8]
        imgs = [m for m in imgs if m.shape == imgs[0].shape]
        if len(imgs) % 2 == 0:
            imgs = imgs[1:]                           # odd count -> a real median, no averaged values
        if len(imgs) >= 3:
            median = np.partition(np.stack(imgs), len(imgs) // 2, axis=0)[len(imgs) // 2]
            for p in paths:
                cv2.imwrite(p, median)
            return

    for p in paths[1:]:
        shutil.copyfile(paths[0], p)


def backup_mattes(mask_dir, backup_dir):
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(mask_dir, backup_dir)


def restore_backup_mattes(mask_dir, backup_dir):
    if not os.path.exists(backup_dir):
        print("No backup masks folder was found.")
        return
    shutil.copytree(backup_dir, mask_dir, dirs_exist_ok=True)
    print("Original masks restored.")


def remove_backup_mattes():
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)


# ---------------------------------------------------------------------------
# Main dedup routine
# ---------------------------------------------------------------------------

def replace_similar_matte_frames(parent_window, sensitivity=1.0, result_info=None, comparison_log=None):
    """
    Freezes each object's masks across runs of frames where its source image is
    held. Objects are independent: each is identified by its mask file name and
    frozen on its own schedule. Only the span of frames that have masks is
    processed.

    sensitivity: scales the auto-calibrated threshold. >1 flags smaller changes
        (freezes less); <1 tolerates more (freezes more).
    result_info: optional dict receiving "hi", "lo", "calibration",
        "pair_scores", per-object "objects" stats, frame counts, and timing.
    comparison_log: optional list receiving one dict per comparison: "object",
        "base", "candidate", "decision" ("grouped", "rejected_<reason>" or
        "object_missing_break"), "max_diff", "frac_over_lo".
    Returns True on success, False if there was nothing to do.
    """
    start = time.perf_counter()

    if not os.path.exists(frames_dir):
        print("Could not find frames to dedupe.\nPlease load a video first.")
        return False
    if not os.path.exists(mask_dir):
        print("No masks directory found.\nPlease track objects first.")
        return False

    frame_numbers = sorted((os.path.splitext(f)[0] for f in os.listdir(frames_dir) if f.endswith(".png")),
                           key=lambda x: (0, int(x), x) if x.isdecimal() else (1, 0, x))

    # If a backup exists, deduplication has run before: restore the originals so
    # we always work from the source data. (The backup itself is created below,
    # right before the first mask is overwritten, so a run that finds nothing to
    # dedupe never leaves behind a stale backup.)
    if os.path.exists(backup_dir):
        restore_backup_mattes(mask_dir, backup_dir)

    progress_dialog = QProgressDialog("Deduplicating...", None, 0, 100, parent_window)
    progress_dialog.setWindowTitle("Progress")
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setAutoClose(True)
    progress_dialog.show()
    bar = tqdm(total=100, desc="Deduplicating mask frames...")
    shown = 0

    def set_progress(fraction):
        nonlocal shown
        pct = min(100, int(fraction * 100))
        if pct > shown:
            bar.update(pct - shown)
            shown = pct
            progress_dialog.setValue(pct)
            QApplication.processEvents()

    messages = []          # printed after the progress bar closes, so they don't garble it
    try:
        entries = []
        with ThreadPoolExecutor() as pool:             # cv2 releases the GIL while decoding
            for i, entry in enumerate(pool.map(load_frame_entry, frame_numbers)):
                entries.append(entry)
                set_progress(0.8 * (i + 1) / len(frame_numbers))

        # Only the span of frames that have masks. Gaps inside it stay as None
        # entries, so frames on either side of a gap are never treated as neighbors.
        with_masks = [i for i, e in enumerate(entries) if e is not None and e["objects"]]
        if not with_masks:
            messages.append("No valid masks found in any frames.")
            return False
        first, last = with_masks[0], with_masks[-1]
        outside_span = len(frame_numbers) - (last - first + 1)
        frame_numbers, entries = frame_numbers[first:last + 1], entries[first:last + 1]

        objects = sorted({name for e in entries if e is not None for name in e["objects"]})
        timelines = {name: [None if e is None or name not in e["objects"] else (e["blocks"], e["objects"][name])
                            for e in entries] for name in objects}

        scored = {name: object_pair_scores(timelines[name]) for name in objects}
        scores = np.concatenate([s for s, _ in scored.values()])      # pooled across objects
        hi, calibration = calibrate_hi(scores, sensitivity)

        if not os.path.exists(backup_dir):
            backup_mattes(mask_dir, backup_dir)

        frozen_total, per_object = 0, {}
        for oi, name in enumerate(objects):
            trace = [] if comparison_log is not None else None
            groups = find_hold_groups(timelines[name], hi, trace)
            for members in groups:
                replace_object_masks(name, [frame_numbers[i] for i in members])
            frozen = sum(len(g) - 1 for g in groups)
            frozen_total += frozen
            per_object[name] = {"groups": len(groups), "frames_frozen": frozen,
                                "frames_present": sum(t is not None for t in timelines[name]),
                                "small_object_pairs_excluded": scored[name][1]}
            if trace:
                comparison_log.extend(
                    {"object": name, "base": frame_numbers[i], "candidate": frame_numbers[j],
                     "decision": decision, "max_diff": max_diff, "frac_over_lo": frac_over_lo}
                    for i, j, decision, max_diff, frac_over_lo in trace)
            set_progress(0.8 + 0.2 * (oi + 1) / len(objects))

        messages.append(f"Deduplicated {frozen_total} mask frames across {len(objects)} object(s) "
                        f"(hi={hi:.2f}, {calibration['method']}"
                        + (f", capped from {calibration['capped_from']:.1f}" if "capped_from" in calibration else "")
                        + ")")
    finally:
        progress_dialog.setValue(100)
        bar.close()
        for message in messages:
            print(message)

    if result_info is not None:
        result_info.update({
            "hi": hi, "lo": hi * LO_RATIO, "calibration": calibration, "pair_scores": scores.tolist(),
            "objects": per_object, "deduped_frames_amount": frozen_total,
            "frames_processed": len(frame_numbers), "frames_outside_masked_span": outside_span,
            "frames_without_masks_in_span": sum(e is None for e in entries),
            "total_seconds": time.perf_counter() - start,
        })
    return True
