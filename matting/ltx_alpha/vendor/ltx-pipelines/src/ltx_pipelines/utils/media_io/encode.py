"""Video/audio encode paths (SDR H.264 and HDR EXR+HLG)."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from queue import Queue

import av
import numpy as np
import torch
from tqdm import tqdm

from ltx_core.color.audio_mux import prepare_audio_stream as _prepare_audio_stream
from ltx_core.color.audio_mux import validate_audio_waveform
from ltx_core.color.audio_mux import write_audio as _write_audio
from ltx_core.color.hlg import encode_linear_hdr_frames_to_hlg_mp4
from ltx_core.color.primaries import Primaries
from ltx_core.color.yuv import FrameConverter, PixelFormat, yuv420p_bt709_converter_
from ltx_core.types import Audio
from ltx_pipelines.utils.media_io.color_config import EXRColorSpace, decode_hdr_video
from ltx_pipelines.utils.media_io.exr import exr_colorspace_tag, exr_dir_label, save_exr_tensor

logger = logging.getLogger(__name__)


def _encode_hdr_video_outputs(
    video: Iterator[torch.Tensor],
    output_path: Path,
    fps: float,
    color_space: EXRColorSpace,
    *,
    max_workers: int = 8,
    audio: Iterator[Audio] | None = None,
    audio_sampling_rate: int | None = None,
    device: torch.device | None = None,
) -> Path:
    """Write EXR frames plus a BT.2020/HLG master.
    One pass over ``video`` feeds two independent sinks:
    * **HLG** — always Rec.709 scene-linear (``decode_hdr_video`` default).
    * **EXR** — matches ``color_space``: log codes when :attr:`~EXRColorSpace.is_log_working`,
      else scene-linear in ``color_space.source_primaries`` (reuse the HLG tensor when that
      is already Rec.709).
    Video tensors flow directly from decode through both sinks. When ``audio`` is
    deferred, HLG opens its AAC stream from ``audio_sampling_rate`` and pulls
    the completed track only after this iterator is exhausted.
    Args:
        video: VAE decode output as an iterator of ``[F, H, W, C]`` float tensors in
            ``[0, 1]`` (the compressed working-space signal), e.g. from
            :func:`~ltx_pipelines.chunks.split_decoded_chunks`.
        output_path: Destination of the HLG master. EXR frames go to
            ``<stem>_<label>_exr/`` where ``label`` is :func:`~ltx_pipelines.utils.media_io.exr.exr_dir_label`.
        fps: Output frame rate.
        color_space: HDR colour space for EXR tags / linear EXR primaries.
        max_workers: Number of EXR writer threads.
        audio: Optional stereo track muxed into the HLG master. It may be filled
            lazily while ``video`` is consumed.
        audio_sampling_rate: AAC sample rate, required when ``audio`` is set.
        device: Device for the HLG colour conversion. Defaults to CUDA when available.
    Returns:
        The directory the EXR frames were written to.
    Raises:
        ValueError: If ``video`` yields nothing.
    """
    exr_dir = output_path.parent / f"{output_path.stem}_{exr_dir_label(color_space)}_exr"
    exr_dir.mkdir(parents=True, exist_ok=True)
    exr_primaries = color_space.source_primaries
    exr_color_space = exr_colorspace_tag(color_space)
    frames_written = 0

    def hlg_chunks() -> Iterator[torch.Tensor]:
        """Write EXR for each chunk, then yield Rec.709 linear for the HLG encoder."""
        nonlocal frames_written
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending: deque[Future[None]] = deque()
            for chunk in video:
                # HLG sink: always Rec.709 scene-linear.
                hlg_linear = decode_hdr_video(chunk)

                # EXR sink: independent of HLG (reuse only when pixels already match).
                if color_space.is_log_working:
                    exr_hdr = chunk.float()
                elif color_space.source_primaries is Primaries.REC709:
                    exr_hdr = hlg_linear
                else:
                    exr_hdr = decode_hdr_video(chunk, out_primaries=color_space.source_primaries)

                for frame in exr_hdr:
                    path = exr_dir / f"frame_{frames_written:05d}.exr"
                    pending.append(
                        executor.submit(
                            save_exr_tensor, frame.cpu().clone(), str(path), True, exr_primaries, exr_color_space
                        )
                    )
                    frames_written += 1
                # Bound the writer backlog so pinned CPU frames cannot grow to the whole clip.
                while len(pending) > 2 * max_workers:
                    pending.popleft().result()
                yield hlg_linear
            while pending:
                pending.popleft().result()

    logger.info(
        "Encoding BT.2020/HLG/10-bit master%s: %s",
        " (with audio)" if audio is not None else "",
        output_path,
    )
    # Empty ``video`` raises inside the HLG encoder (after EXR side effects are none).
    encode_linear_hdr_frames_to_hlg_mp4(
        hlg_chunks(),
        output_path,
        fps=fps,
        primaries=Primaries.REC709,
        audio=audio,
        device=device,
        audio_sampling_rate=audio_sampling_rate,
    )
    logger.info(
        "Wrote %d EXR frames (%s/%s) to %s",
        frames_written,
        exr_primaries.value,
        exr_color_space,
        exr_dir,
    )
    return exr_dir


def _open_video_stream(
    container: av.container.Container,
    fps: int,
    height: int,
    width: int,
    frame_converter: FrameConverter,
    crf: int,
    preset: str,
    thread_count: int,
) -> av.video.VideoStream:
    stream = container.add_stream("libx264", rate=int(fps), options={"crf": str(crf), "preset": preset})
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.codec_context.thread_count = thread_count
    stream.codec_context.thread_type = "FRAME"
    if frame_converter.color_space is not None:
        stream.codec_context.colorspace = frame_converter.color_space.av_colorspace
    if frame_converter.color_range is not None:
        stream.codec_context.color_range = frame_converter.color_range.av_color_range
    return stream


def encode_sdr_h264(
    video: Iterator[torch.Tensor],
    *,
    fps: int,
    output_path: str,
    video_chunks_number: int,
    audio: Iterator[Audio] | None = None,
    audio_sampling_rate: int | None = None,
    frame_converter: FrameConverter = yuv420p_bt709_converter_,
    crf: int = 19,
    preset: str = "veryfast",
    thread_count: int = 0,
) -> None:
    """Mux RGB frame chunks to SDR H.264 (optional AAC).
    Pass :func:`~ltx_pipelines.chunks.deferred_stitch_audio` (and ``audio_sampling_rate``)
    when audio is assembled while ``video`` is consumed. Pull the iterator only after
    video drain.
    """

    def convert(chunk: torch.Tensor) -> torch.Tensor:
        return frame_converter(chunk.movedim(-1, -3))

    first_raw_chunk = next(video, None)
    if first_raw_chunk is None:
        raise ValueError("video is empty; expected at least one frame chunk.")
    first_chunk = convert(first_raw_chunk)

    if frame_converter.pixel_format == PixelFormat.RGB24:
        height, width = first_chunk.shape[-3], first_chunk.shape[-2]
    else:
        height = first_chunk.shape[-2] * 2 // 3
        width = first_chunk.shape[-1]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    container = av.open(output_path, mode="w")
    success = False
    try:
        stream = _open_video_stream(container, fps, height, width, frame_converter, crf, preset, thread_count)
        audio_stream = None
        if audio is not None:
            if audio_sampling_rate is None:
                raise ValueError("audio_sampling_rate is required when muxing audio")
            audio_stream = _prepare_audio_stream(container, audio_sampling_rate)
        av_format = frame_converter.pixel_format.av_format

        def cpu_chunks() -> Generator[np.ndarray, None, None]:
            yield first_chunk.to("cpu").numpy()
            for chunk in video:
                yield convert(chunk).to("cpu").numpy()

        _encode_chunks_threaded(
            container=container,
            stream=stream,
            av_format=av_format,
            chunks=cpu_chunks(),
            progress_total=video_chunks_number,
        )

        mux_audio = next(audio, None) if audio is not None else None
        if mux_audio is not None:
            validate_audio_waveform(mux_audio)
            assert audio_stream is not None
            _write_audio(container, audio_stream, mux_audio)
        success = True
    finally:
        container.close()
        if not success:
            Path(output_path).unlink(missing_ok=True)
    logger.info("Video saved to %s", output_path)


def encode_video(  # noqa: PLR0913
    video: torch.Tensor | Iterator[torch.Tensor],
    fps: int,
    audio: Audio | Iterator[Audio] | None,
    output_path: str,
    video_chunks_number: int,
    frame_converter: FrameConverter = yuv420p_bt709_converter_,
    crf: int = 19,
    preset: str = "veryfast",
    thread_count: int = 0,
    *,
    color_space: EXRColorSpace | None = None,
    audio_sampling_rate: int | None = None,
) -> Path | None:
    """Encode RGB frames to video; SDR H.264 or HDR EXR+HLG when ``color_space`` is set.
    Args:
        video: RGB frames as a ``(F, H, W, C)`` float ``[0, 1]`` tensor, or an iterator of
            such per-chunk tensors (e.g. the VAE decoder output). An empty iterator raises.
            For HDR this is the compressed working-space signal from VAE decode.
        fps: Output frame rate.
        audio: Audio to mux, an iterator yielding audio after ``video`` is consumed,
            or None for a video-only file. Pass
            :func:`~ltx_pipelines.chunks.deferred_stitch_audio` when parts are collected
            while ``video`` is consumed, and set ``audio_sampling_rate`` so AAC can open
            before any video packets. Both SDR and HDR pull the iterator only after
            draining ``video``. Waveforms must be stereo ``(2, N)`` or ``(N, 2)``.
        output_path: Destination path. Parent directories are created if missing.
            Partial SDR output is removed if encoding fails. HDR writes EXR under
            ``<stem>_<label>_exr/`` (see :func:`~ltx_pipelines.utils.media_io.exr.exr_dir_label`) and an
        video_chunks_number: Number of chunks yielded by ``video``, for the progress bar
            (SDR path only).
        frame_converter: Float-to-pixel converter (default YUV420p BT.709); SDR only.
        crf: libx264 constant rate factor; lower is higher quality (0-51); SDR only.
        preset: libx264 speed/compression preset; SDR only.
        thread_count: libx264 thread count (0 = auto); SDR only.
        color_space: HDR colour space for the encode sink. ``None`` (default) is SDR.
            When set, writes half EXR frames + HLG master.
        audio_sampling_rate: AAC sample rate. Inferred when ``audio`` is an
            :class:`~ltx_core.types.Audio`; required for an audio iterator.
    Returns:
        EXR output directory when HDR, else ``None``.
    Raises:
        ValueError: On an empty ``video``, a non-stereo ``audio`` waveform, or conflicting
            explicit and embedded audio sampling rates.
    """
    if isinstance(audio, Audio):
        validate_audio_waveform(audio)
        if audio_sampling_rate is not None and audio_sampling_rate != audio.sampling_rate:
            raise ValueError(
                f"audio_sampling_rate {audio_sampling_rate} does not match audio sampling rate {audio.sampling_rate}"
            )
        audio_sampling_rate = audio.sampling_rate
        audio = iter((audio,))

    if isinstance(video, torch.Tensor):
        video = iter([video])

    if color_space is not None:
        return _encode_hdr_video_outputs(
            video,
            Path(output_path),
            fps=float(fps),
            color_space=color_space,
            audio=audio,
            audio_sampling_rate=audio_sampling_rate,
        )

    encode_sdr_h264(
        video,
        fps=fps,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
        audio=audio,
        audio_sampling_rate=audio_sampling_rate,
        frame_converter=frame_converter,
        crf=crf,
        preset=preset,
        thread_count=thread_count,
    )
    return None


def encode_audio(audio: Audio, output_path: str) -> None:
    """Save an audio waveform as a 16-bit PCM ``.wav`` file at the source sampling rate.
    Reuses :func:`_write_audio` (the same muxing path used by :func:`encode_video`);
    the only difference is a PCM (``pcm_s16le``) stream in a WAV container instead of
    the AAC stream used for muxed video.
    """
    validate_audio_waveform(audio)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    container = av.open(output_path, mode="w")
    audio_stream = container.add_stream("pcm_s16le", rate=audio.sampling_rate)
    audio_stream.codec_context.sample_rate = audio.sampling_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio.sampling_rate)
    try:
        _write_audio(container, audio_stream, audio)
    finally:
        container.close()
    logger.info(f"Audio saved to {output_path}")


def _encode_chunks_threaded(
    container: av.container.Container,
    stream: av.video.stream.VideoStream,
    av_format: str,
    chunks: Iterator[np.ndarray],
    progress_total: int,
) -> None:
    """Run libx264 frame.encode + container.mux on a background thread while
    the caller produces numpy chunks on the current thread. The 1-slot queue
    lets the producer get one chunk ahead (so the next VAE/gather chunk
    overlaps with libx264 encoding the previous chunk) without buffering more
    than one chunk in CPU memory.
    """
    chunk_queue: Queue[np.ndarray | None] = Queue(maxsize=1)
    encoder_error: list[BaseException] = []

    def encoder_worker() -> None:
        error: BaseException | None = None
        while True:
            arr = chunk_queue.get()
            if arr is None:
                break
            if error is not None:
                continue
            try:
                for frame_array in arr:
                    frame = av.VideoFrame.from_ndarray(frame_array, format=av_format)
                    for packet in stream.encode(frame):
                        container.mux(packet)
            except Exception as e:
                error = e
        if error is None:
            try:
                for packet in stream.encode():
                    container.mux(packet)
            except Exception as e:
                error = e
        if error is not None:
            encoder_error.append(error)

    encoder_thread = threading.Thread(target=encoder_worker, name="h264-encoder")
    encoder_thread.start()
    try:
        for arr in tqdm(chunks, total=progress_total):
            chunk_queue.put(arr)
    finally:
        chunk_queue.put(None)
        encoder_thread.join()

    if encoder_error:
        raise encoder_error[0]
