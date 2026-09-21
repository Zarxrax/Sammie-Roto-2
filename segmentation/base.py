"""Shared prompt, tracking, and mask state for segmentation engines."""
import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog, QApplication
from sammie import core
from sammie.settings_manager import get_settings_manager

class SamManager:
    def __init__(self):
        self.model = None
        self.loaded_model_name = None
        self.predictor = None
        self.inference_state = None
        self.propagated = False  # whether we have propagated the masks
        self.deduplicated = False  # whether we have deduplicated the masks
        self.callbacks = []  # Add callbacks for segmentation events

    def add_callback(self, callback):
        """Add callback for segmentation events"""
        self.callbacks.append(callback)

    def _notify(self, action, **kwargs):
        """Notify callbacks of changes"""
        for callback in self.callbacks:
            try:
                callback(action, **kwargs)
            except Exception as e:
                print(f"Callback error: {e}")

    def load_segmentation_model(self, model=None, parent_window=None):
        from segmentation.registry import get_engine, DEFAULT_ENGINE_ID
        if model is None:
            model = get_settings_manager().get_session_setting("sam_model", DEFAULT_ENGINE_ID)
        spec = get_engine(model)
        if spec is None:
            raise ValueError(f"Unknown segmentation engine: {model}")
        core.DeviceManager.clear_cache()
        predictor = spec.load_predictor(core.DeviceManager.get_device(), parent_window)
        if predictor is None:
            return False
        self.predictor = predictor
        self.loaded_model_name = spec.id
        return True

    def unload_segmentation_model(self):
        """Unload the SAM model and clear cache"""
        self.predictor = None
        self.inference_state = None
        core.DeviceManager.clear_cache()
        print("Unloaded Segmentation model")

    def offload_model_to_cpu(self):
        """Offload SAM2 model to CPU to free VRAM"""
        device = core.DeviceManager.get_device()
        if device.type == 'cpu':
            return  # Already on CPU, nothing to do

        if self.predictor is not None:
            self.predictor.to('cpu')
            core.DeviceManager.clear_cache()

    def load_model_to_device(self):
        """Load SAM2 model back to the active device"""
        device = core.DeviceManager.get_device()
        if device.type == 'cpu':
            return  # Already on CPU, nothing to do

        if self.predictor is not None:
            self.predictor.to(device)

    def initialize_predictor(self):
        self.inference_state = self.predictor.init_state(
            video_path=core.frames_dir, async_loading_frames=True, offload_video_to_cpu=True
        )

    def _clear_frame_if_tracked(self, object_id, frame_number):
        """Reset frame_number to a blank slate if it is been tracked,
        so new points generate a segment without using any memory from other frames
        """
        obj_idx = self.inference_state["obj_id_to_idx"].get(object_id)
        if obj_idx is None:
            return
        tracked = self.inference_state["frames_tracked_per_obj"][obj_idx]
        if frame_number not in tracked:
            return
        self.predictor.clear_all_prompts_in_frame(
            self.inference_state, frame_number, object_id, need_output=False
        )
        tracked.pop(frame_number, None)
        for d in (self.inference_state["output_dict_per_obj"][obj_idx],
                  self.inference_state["temp_output_dict_per_obj"][obj_idx]):
            d["non_cond_frame_outputs"].pop(frame_number, None)

    def _snapshot_frame_state(self, object_id, frame_number):
        """Capture everything _clear_frame_if_tracked (and add_new_points_or_box) could
        touch for this frame/object, so a preview can be undone without a trace.
        """
        obj_idx = self.inference_state["obj_id_to_idx"].get(object_id)
        if obj_idx is None:
            # Object doesn't exist yet - add_new_points_or_box() will auto-register it
            # as a side effect of running the preview (obj_id_to_idx/obj_idx_to_id/
            # obj_ids, plus empty entries in every per-object dict below). Remember
            # that so _restore_frame_state can undo the registration afterward,
            # instead of leaving a brand-new, populated object behind permanently.
            return {"object_id": object_id, "obj_idx": None, "was_registered": False}

        tracked = self.inference_state["frames_tracked_per_obj"][obj_idx]
        output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
        temp_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
        point_inputs = self.inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs = self.inference_state["mask_inputs_per_obj"][obj_idx]
        return {
            "object_id": object_id,
            "obj_idx": obj_idx,
            "was_registered": True,
            "was_tracked": frame_number in tracked,
            "tracked_entry": tracked.get(frame_number),
            "output_cond": output_dict["cond_frame_outputs"].get(frame_number),
            "output_noncond": output_dict["non_cond_frame_outputs"].get(frame_number),
            "temp_cond": temp_dict["cond_frame_outputs"].get(frame_number),
            "temp_noncond": temp_dict["non_cond_frame_outputs"].get(frame_number),
            "had_point_inputs": frame_number in point_inputs,
            "point_inputs": point_inputs.get(frame_number),
            "had_mask_inputs": frame_number in mask_inputs,
            "mask_inputs": mask_inputs.get(frame_number),
        }

    def _restore_frame_state(self, frame_number, snapshot):
        """Undo _snapshot_frame_state - restores frames_tracked_per_obj plus every
        cond/non-cond entry exactly as it was, regardless of what happened in between."""
        if snapshot is None:
            return

        object_id = snapshot["object_id"]

        if not snapshot["was_registered"]:
            # The object didn't exist before the preview ran - undo whatever
            # auto-registration add_new_points_or_box() performed as a side effect,
            # rather than leaving a phantom object (with real prompt/output data for
            # this frame) sitting in inference_state for the rest of the session.
            obj_idx = self.inference_state["obj_id_to_idx"].pop(object_id, None)
            if obj_idx is not None:
                self.inference_state["obj_idx_to_id"].pop(obj_idx, None)
                obj_ids = self.inference_state.get("obj_ids")
                if obj_ids is not None and object_id in obj_ids:
                    obj_ids.remove(object_id)
                for dict_name in (
                    "output_dict_per_obj", "temp_output_dict_per_obj",
                    "frames_tracked_per_obj", "point_inputs_per_obj", "mask_inputs_per_obj",
                ):
                    d = self.inference_state.get(dict_name)
                    if d is not None:
                        d.pop(obj_idx, None)
            return

        obj_idx = snapshot["obj_idx"]
        tracked = self.inference_state["frames_tracked_per_obj"][obj_idx]
        output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
        temp_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
        point_inputs = self.inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs = self.inference_state["mask_inputs_per_obj"][obj_idx]

        if snapshot["was_tracked"]:
            tracked[frame_number] = snapshot["tracked_entry"]
        else:
            tracked.pop(frame_number, None)

        for d, cond_key, noncond_key in (
            (output_dict, "output_cond", "output_noncond"),
            (temp_dict, "temp_cond", "temp_noncond"),
        ):
            for storage_key, snap_key in (("cond_frame_outputs", cond_key), ("non_cond_frame_outputs", noncond_key)):
                val = snapshot[snap_key]
                if val is not None:
                    d[storage_key][frame_number] = val
                else:
                    d[storage_key].pop(frame_number, None)

        if snapshot["had_point_inputs"]:
            point_inputs[frame_number] = snapshot["point_inputs"]
        else:
            point_inputs.pop(frame_number, None)

        if snapshot["had_mask_inputs"]:
            mask_inputs[frame_number] = snapshot["mask_inputs"]
        else:
            mask_inputs.pop(frame_number, None)

    def segment_image(self, frame_number, object_id, input_points, input_labels):
        extension = core.get_frame_extension()
        frame_filename = os.path.join(core.frames_dir, f"{frame_number:05d}.{extension}")
        if os.path.exists(frame_filename):
            self._clear_frame_if_tracked(object_id, frame_number)

            # Run segmentation function
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_number,
                obj_id=object_id,
                points=input_points,
                labels=input_labels,
                clear_old_points=True,
            )
            # Save the segmentation mask for the object we actually edited.
            for i, out_obj_id in enumerate(out_obj_ids):
                if out_obj_id != object_id:
                    continue
                mask_filename = os.path.join(core.mask_dir, f"{frame_number:05d}", f"{out_obj_id}.png")
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                mask = (mask * 255).astype(np.uint8)
                os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
                cv2.imwrite(mask_filename, mask)

            # Notify that segmentation is complete
            self._notify('segmentation_complete', frame=frame_number, object_id=object_id, out_obj_ids=out_obj_ids)

    def preview_point(self, frame_number, object_id, all_points, preview_x, preview_y, is_positive):
        """Run a preview using the video predictor, then revert the state."""
        if self.predictor is None or self.inference_state is None:
            return None

        snapshot = self._snapshot_frame_state(object_id, frame_number)

        try:
            self._clear_frame_if_tracked(object_id, frame_number)

            existing = [p for p in all_points
                        if p['frame'] == frame_number and p['object_id'] == object_id]

            # Build preview point set
            preview_points = np.array([[p['x'], p['y']] for p in existing] + [[preview_x, preview_y]], dtype=np.float32)
            preview_labels = np.array([1 if p['positive'] else 0 for p in existing] + [1 if is_positive else 0], dtype=np.int32)

            # run with preview point
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_number,
                obj_id=object_id,
                points=preview_points,
                labels=preview_labels,
                clear_old_points=True,
            )

            # Capture the preview mask
            preview_mask = None
            for i, oid in enumerate(out_obj_ids):
                if oid == object_id:
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    preview_mask = (mask * 255).astype(np.uint8)
                    break

            return preview_mask

        except Exception as e:
            print(f"Preview error: {e}")
            return None

        finally:
            self._restore_frame_state(frame_number, snapshot)

    def replay_points(self, points_list):
        """Replay all points incrementally to rebuild masks."""
        frame_count = core.VideoInfo.total_frames
        self.predictor.reset_state(self.inference_state)

        for frame_number in range(frame_count):
            frame_points = [p for p in points_list if p['frame'] == frame_number]
            if not frame_points:
                continue

            frame_object_ids = {p['object_id'] for p in frame_points}
            for object_id in frame_object_ids:
                filtered_points = [
                    (p['x'], p['y'], p['positive'])
                    for p in frame_points if p['object_id'] == object_id
                ]
                out_obj_ids, out_mask_logits = [], []  # guards the save loop below if every add fails
                for i in range(1, len(filtered_points) + 1):
                    # Replay one click at a time, in the order they were made - matches
                    # how segment_image() was actually called during live editing
                    # (each click resubmits the full point set so far as its own
                    # separate call). This matters for fidelity: e.g. undoing the
                    # last point needs to reproduce the mask state that existed right
                    # before that point was added, not the mask a single batched call
                    # with the remaining points would produce - those aren't the same.
                    subset = filtered_points[:i]
                    input_points = np.array([(x, y) for x, y, _ in subset], dtype=np.float32)
                    input_labels = np.array([1 if pos else 0 for _, _, pos in subset], dtype=np.int32)
                    try:
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=frame_number,
                            obj_id=object_id,
                            points=input_points,
                            labels=input_labels,
                            clear_old_points=True
                        )
                    except Exception as e:
                        print(f"Error during prediction for frame {frame_number}, object {object_id}, point {i}: {e}")
                        continue

                # Save the mask for the object we just replayed only
                for j, out_obj_id in enumerate(out_obj_ids):
                    if out_obj_id != object_id:
                        continue
                    mask_filename = os.path.join(core.mask_dir, f"{frame_number:05d}", f"{out_obj_id}.png")
                    mask = (out_mask_logits[j] > 0.0).cpu().numpy().squeeze()
                    mask = (mask * 255).astype(np.uint8)
                    try:
                        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
                        cv2.imwrite(mask_filename, mask)
                    except Exception as e:
                        print(f"Error saving mask for frame {frame_number}, object {out_obj_id}: {e}")

        self._notify('replay_complete')


    def _propagate(self, parent_window, start_frame_idx, max_frame_num_to_track, reverse=False,
                    show_progress=True):
        """Core propagation loop shared by all tracking functions.

        Args:
            parent_window: Used for the progress dialog and to nudge the frame slider as we go.
            start_frame_idx: Frame to start propagating from.
            max_frame_num_to_track: How many additional frames to propagate beyond the start
                frame, or None to propagate to the end (or beginning, if reverse=True) of the video.
            reverse: If True, propagate backward toward frame 0 instead of forward.
            show_progress: If False, skips the progress dialog - intended for single-frame steps
                where a modal dialog would just be visual noise.

        Returns:
            (last_frame_idx, cancelled) - last_frame_idx is the last frame actually processed
            (None if nothing was processed), cancelled is True if the user hit Cancel.
        """
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)
        total_frames = (max_frame_num_to_track + 1) if max_frame_num_to_track is not None else core.VideoInfo.total_frames

        progress_dialog = None
        if show_progress:
            progress_dialog = QProgressDialog("Tracking...", "Cancel", 0, 100, parent_window)
            progress_dialog.setWindowTitle("Progress")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setAutoClose(True)
            progress_dialog.show()

        last_frame_idx = None
        cancelled = False

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state, start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track, reverse=reverse):
            for i, out_obj_id in enumerate(out_obj_ids):
                mask_filename = os.path.join(core.mask_dir, f"{out_frame_idx:05d}", f"{out_obj_id}.png")
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                mask = (mask * 255).astype(np.uint8)
                os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
                cv2.imwrite(mask_filename, mask)

            last_frame_idx = out_frame_idx

            if progress_dialog is not None:
                frames_processed = abs(out_frame_idx - start_frame_idx) + 1
                progress_dialog.setValue(int(frames_processed * 100 / total_frames))

            # Update display at the specified frequency (always update for quick, dialog-less steps)
            if not show_progress or out_frame_idx % display_update_frequency == 0:
                try:
                    parent_window.frame_slider.setValue(out_frame_idx)
                except Exception as e:
                    print(f"Error updating display: {e}")

            QApplication.processEvents()
            if progress_dialog is not None and progress_dialog.wasCanceled():
                cancelled = True
                break

        if progress_dialog is not None:
            if cancelled:
                progress_dialog.close()
            else:
                progress_dialog.setValue(100)

        return last_frame_idx, cancelled

    def track_objects(self, parent_window):
        """Track all objects across the full in/out point range (or the entire video)."""
        frame_count = core.VideoInfo.total_frames
        settings_mgr = get_settings_manager()
        in_point = settings_mgr.get_session_setting("in_point", None)
        out_point = settings_mgr.get_session_setting("out_point", None)
        if in_point is None:
            in_point = 0
        frames_to_track = None
        total_frames = frame_count
        if out_point is not None:
            frames_to_track = out_point - in_point
            total_frames = frames_to_track + 1

        last_frame_idx, cancelled = self._propagate(
            parent_window, start_frame_idx=in_point, max_frame_num_to_track=frames_to_track, reverse=False)

        if not cancelled:
            self.propagated = (total_frames == frame_count)
            print("Tracking completed")
            return 1
        else:
            self.propagated = False
            print("Tracking cancelled")
            return 0

    def track_forward(self, parent_window, current_frame):
        """Track all objects forward from current_frame to the out point (or end of video)."""
        settings_mgr = get_settings_manager()
        out_point = settings_mgr.get_session_setting("out_point", None)
        last_frame = out_point if out_point is not None else core.VideoInfo.total_frames - 1
        if current_frame >= last_frame:
            print("Already at the last frame")
            return 1

        max_frame_num_to_track = max(last_frame - current_frame, 0)

        last_frame_idx, cancelled = self._propagate(
            parent_window, start_frame_idx=current_frame, max_frame_num_to_track=max_frame_num_to_track,
            reverse=False)

        if cancelled:
            print("Forward tracking cancelled")
            return 0
        print(f"Forward tracking completed up to frame {last_frame_idx}")
        return 1

    def track_backward(self, parent_window, current_frame):
        """Track all objects backward from current_frame to the in point (or start of video)."""
        settings_mgr = get_settings_manager()
        in_point = settings_mgr.get_session_setting("in_point", None)
        if in_point is None:
            in_point = 0
        if current_frame <= in_point:
            print("Already at the first frame")
            return 1

        max_frame_num_to_track = max(current_frame - in_point, 0)

        last_frame_idx, cancelled = self._propagate(
            parent_window, start_frame_idx=current_frame, max_frame_num_to_track=max_frame_num_to_track,
            reverse=True)

        if cancelled:
            print("Backward tracking cancelled")
            return 0
        print(f"Backward tracking completed back to frame {last_frame_idx}")
        return 1

    def track_one_frame_forward(self, parent_window, current_frame):
        """Track all objects one frame forward from current_frame. Returns the new frame index."""
        last_frame = core.VideoInfo.total_frames - 1
        if current_frame >= last_frame:
            print("Already at the last frame")
            return current_frame

        last_frame_idx, _ = self._propagate(
            parent_window, start_frame_idx=current_frame, max_frame_num_to_track=1,
            reverse=False, show_progress=False)

        return last_frame_idx if last_frame_idx is not None else current_frame

    def track_one_frame_backward(self, parent_window, current_frame):
        """Track all objects one frame backward from current_frame. Returns the new frame index."""
        if current_frame <= 0:
            print("Already at the first frame")
            return current_frame

        last_frame_idx, _ = self._propagate(
            parent_window, start_frame_idx=current_frame, max_frame_num_to_track=1,
            reverse=True, show_progress=False)

        return last_frame_idx if last_frame_idx is not None else current_frame

    def clear_tracking(self):
        """Clear tracking data by deleting all masks, this needs to be followed up by replay_points"""
        if os.path.exists(core.mask_dir):
            shutil.rmtree(core.mask_dir)
        os.makedirs(core.mask_dir)
        self.predictor.reset_state(self.inference_state)
        core.DeviceManager.clear_cache()
        if self.propagated:
            print("Tracking data cleared")
        self.propagated = False
        self.deduplicated = False

