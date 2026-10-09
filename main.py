import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def show_progress(seconds: float, duration: float) -> None:
    fraction = min(seconds / duration, 1.0) if duration > 0 else 0.0
    width = 30
    filled = round(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(
        f"\r转写进度 [{bar}] {fraction:6.1%} "
        f"({seconds / 60:.1f}/{duration / 60:.1f} 分钟)"
    )
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将音频转写为文字；输入 MP4 时先提取同名 MP3 音轨。",
        epilog="输出为输入文件旁的同名 .txt；MP4 转换需要 PATH 中有 ffmpeg。",
    )
    parser.add_argument("input_file", type=Path, metavar="文件", help="音频文件或 MP4 视频文件")
    args = parser.parse_args()
    audio_path = args.input_file

    if not audio_path.is_file():
        mp3_path = audio_path.with_suffix(".mp3")
        if audio_path.suffix.lower() == ".mp4" and mp3_path.is_file():
            print(f"MP4 不存在，直接使用同名音轨: {mp3_path}")
            audio_path = mp3_path
        else:
            print(f"文件不存在: {audio_path}")
            return 1

    if audio_path.suffix.lower() == ".mp4":
        mp3_path = audio_path.with_suffix(".mp3")
        if mp3_path.exists():
            print(f"复用已有音轨: {mp3_path}")
        else:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                print("找不到 ffmpeg，请先安装并将其加入 PATH")
                return 1
            print(f"正在提取音轨: {audio_path.name} -> {mp3_path.name}")
            try:
                subprocess.run(
                    [ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-i", str(audio_path),
                     "-vn", "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path)],
                    check=True,
                )
            except subprocess.CalledProcessError:
                mp3_path.unlink(missing_ok=True)
                print("音轨提取失败，请检查视频是否包含可用音轨")
                return 1
        audio_path = mp3_path

    output_path = audio_path.with_suffix(".txt")
    checkpoint_path = audio_path.with_suffix(".checkpoint.json")
    source = audio_path.stat()
    fingerprint = {"size": source.st_size, "mtime_ns": source.st_mtime_ns}

    if checkpoint_path.exists():
        try:
            state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if (state["source"] != fingerprint or not output_path.exists()
                    or not isinstance(state["offset"], int) or state["offset"] < 0
                    or output_path.stat().st_size < state["offset"]
                    or not isinstance(state["seconds"], (int, float))
                    or state["seconds"] < 0):
                raise ValueError("检查点与音频或文本不匹配")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"无法续传: {exc}；请检查 {checkpoint_path} 和 {output_path}")
            return 1
        with output_path.open("r+b") as file:
            file.truncate(state["offset"])
        if state.get("complete"):
            print(f"转写已经完成: {output_path}")
            return 0
        print(f"从 {state['seconds']:.2f} 秒继续转写: {audio_path.name}")
    else:
        if output_path.exists():
            backup_path = output_path.with_name(output_path.name + ".bak")
            number = 1
            while backup_path.exists():
                backup_path = output_path.with_name(f"{output_path.name}.bak.{number}")
                number += 1
            output_path.rename(backup_path)
            print(f"已有文本没有检查点，已备份至: {backup_path}")
        state = {"source": fingerprint, "seconds": 0.0, "offset": 0, "complete": False}
        output_path.touch()
        save_checkpoint(checkpoint_path, state)

    print("正在加载模型...")
    from faster_whisper import WhisperModel

    model = WhisperModel("medium", device="cpu", compute_type="int8")
    print(f"开始转写: {audio_path.name}")
    options = {"language": "zh", "beam_size": 1, "vad_filter": True}
    if state["seconds"]:
        # 指定起点时 faster-whisper 会忽略 VAD，但能跳过已经处理的音频。
        options["clip_timestamps"] = str(state["seconds"])
    segments, info = model.transcribe(str(audio_path), **options)
    duration = info.duration
    show_progress(state["seconds"], duration)

    try:
        with output_path.open("ab") as file:
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                file.write((text + "\n").encode("utf-8"))
                file.flush()
                os.fsync(file.fileno())
                state["offset"] = file.tell()
                state["seconds"] = max(state["seconds"], segment.end)
                save_checkpoint(checkpoint_path, state)
                show_progress(state["seconds"], duration)
    finally:
        print()

    state["complete"] = True
    save_checkpoint(checkpoint_path, state)
    show_progress(duration, duration)
    print(f"\n转写完成: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
