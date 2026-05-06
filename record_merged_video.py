"""
record_merged_video.py - Record episodes and merge them into one video.

This wrapper accepts the same user-facing arguments as record_video.py.

Usage:
    python3 record_merged_video.py --object cracker_box
    python3 record_merged_video.py --sequence
    python3 record_merged_video.py --sequence --episodes 5
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2


FRAME_DIR = Path("output/frames")
FPS = 30
OBJECT_ORDER = ("cracker_box", "mustard_bottle", "sugar_box")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run record_video.py and merge all successful episodes into one mp4."
    )
    parser.add_argument("--object", choices=OBJECT_ORDER, default=None,
                        help="Object to record in single-object mode.")
    parser.add_argument("--sequence", action="store_true",
                        help="Record sequence episodes.")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Successful episodes to record.")
    parser.add_argument("--use_gt", action="store_true",
                        help="Use simulator pose instead of camera perception for planning.")
    parser.add_argument("--sequence_perception", action="store_true",
                        help="Use camera perception during --sequence instead of simulator poses.")
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument("--top_view", action="store_true",
                            help="Record and merge the overhead camera view.")
    view_group.add_argument("--both_views", action="store_true",
                            help="Record and merge both normal and overhead camera views.")
    return parser.parse_args()


def record_video_args(args):
    cmd = [sys.executable, "-u", "record_video.py"]

    if args.object is not None:
        cmd.extend(["--object", args.object])
    if args.sequence:
        cmd.append("--sequence")
    if args.episodes != 5:
        cmd.extend(["--episodes", str(args.episodes)])
    if args.use_gt:
        cmd.append("--use_gt")
    if args.sequence_perception:
        cmd.append("--sequence_perception")
    if args.top_view:
        cmd.append("--top_view")
    if args.both_views:
        cmd.append("--both_views")

    return cmd


def output_stem(args):
    if args.sequence:
        return "video_sequence_merged"
    if args.object:
        return f"video_{args.object}_merged"
    return "video_merged"


def output_paths(args):
    stem = output_stem(args)
    if args.both_views:
        return {
            "main": Path(f"{stem}.mp4"),
            "top": Path(f"{stem}_top.mp4"),
        }
    if args.top_view:
        return {"top": Path(f"{stem}_top.mp4")}
    return {"main": Path(f"{stem}.mp4")}


def run_recorder(cmd):
    done_count = None
    done_re = re.compile(r"Done!\s+(\d+)\s+successful")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        match = done_re.search(line)
        if match:
            done_count = int(match.group(1))

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"record_video.py exited with code {return_code}")

    return done_count


def episode_frames(ep_num, view_name):
    if view_name == "main":
        return sorted(FRAME_DIR.glob(f"ep{ep_num}_frame_*.png"))
    return sorted(FRAME_DIR.glob(f"ep{ep_num}_{view_name}_frame_*.png"))


def infer_episode_count(max_episodes, view_name):
    count = 0
    for ep_num in range(max_episodes):
        if not episode_frames(ep_num, view_name):
            break
        count += 1
    return count


def merge_episode_frames(output_path, episode_count, view_name):
    if episode_count <= 0:
        raise RuntimeError("No successful episode frames were found to merge.")

    writer = None
    total_frames = 0

    try:
        for ep_num in range(episode_count):
            frames = episode_frames(ep_num, view_name)
            if not frames:
                raise RuntimeError(
                    f"Missing {view_name} frames for successful episode {ep_num}."
                )

            for frame_path in frames:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    raise RuntimeError(f"Could not read frame: {frame_path}")

                if writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create video file: {output_path}")

                writer.write(frame)
                total_frames += 1
    finally:
        if writer is not None:
            writer.release()

    return total_frames


def make_vscode_friendly(output_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(
            f"ffmpeg was not found, so {output_path} was left in OpenCV mp4v format."
        )
        return False

    temp_path = output_path.with_name(f"{output_path.stem}_h264_tmp{output_path.suffix}")
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(output_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(temp_path),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        if temp_path.exists():
            temp_path.unlink()
        print(f"ffmpeg conversion failed for {output_path}; keeping the OpenCV mp4v file.")
        return False

    temp_path.replace(output_path)
    return True


def main():
    args = parse_args()
    cmd = record_video_args(args)
    outputs = output_paths(args)

    print("Running:", " ".join(cmd), flush=True)
    done_count = run_recorder(cmd)

    first_view = next(iter(outputs))
    episode_count = (
        done_count if done_count is not None
        else infer_episode_count(args.episodes, first_view)
    )

    for view_name, output_path in outputs.items():
        total_frames = merge_episode_frames(output_path, episode_count, view_name)
        converted = make_vscode_friendly(output_path)
        codec_note = "H.264/yuv420p" if converted else "OpenCV mp4v"
        print(
            f"\nMerged {episode_count} successful episodes "
            f"({total_frames} {view_name} frames) into {output_path} ({codec_note})"
        )


if __name__ == "__main__":
    main()
