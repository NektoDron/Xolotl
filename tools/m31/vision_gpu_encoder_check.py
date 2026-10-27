#!/usr/bin/env python3
"""Run only inside the coordinator's aictl/card0 + exclusive speed window.

Compare GPU single and batch PR mtmd APIs bitwise, report GPU/CPU drift, and
sample GPU0 memory. This is an encoder API/IPC check, not native LLM logits.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from vision_batch import compare
from vision_encoder import VisionEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, default=Path(__file__).resolve().parents[2]/'build-vision/m31_vision_adapter')
    parser.add_argument('--cpu-reference', type=Path, default=Path('/mnt/ai/logs/m31-port/vision-encode-v1'))
    args = parser.parse_args()
    if os.environ.get('M31_CARD_SESSION') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        parser.error('requires aictl card0 session (M31_CARD_SESSION=1 CUDA_VISIBLE_DEVICES=0)')
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path('/mnt/ai/models/MiniMax-M3.1-Flash-GGUF')
    model = root/'UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf'
    mmproj = root/'mmproj-BF16.gguf'
    stopped = threading.Event()
    samples = []
    def sample():
        while not stopped.is_set():
            result = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                used, free = map(int, result.stdout.strip().split(','))
                samples.append(dict(time=time.time(), used_mib=used, free_mib=free))
            stopped.wait(.1)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    results = []
    try:
        with VisionEncoder(args.adapter, model, mmproj, args.output/'cache', backend='cuda', threads=8, cpu_workers=1) as encoder:
            for label, filename in [('image', '0.png'), ('video', 'shapes-12s.mp4')]:
                reference = args.cpu_reference/(label+'-batch')/'manifest.json'
                prompt = (reference.parent/'prompt.txt').read_text()
                media = ['/mnt/ai/logs/m31-ref/media/'+filename]
                manifest = encoder.encode(prompt, media)
                ipc = manifest.parent.parent/'ipc'/'manifest.json'
                single_stats = encoder.last_stats.copy()
                encoder.close()  # one CUDA encoder at a time; avoid doubled VRAM
                spec = dict(model=str(model), mmproj=str(mmproj), backend='cuda',
                            prompt=prompt, media=media, video_fps=1.0, timestamp_interval_ms=5000,
                            encode=True, encoder_mode='batch', threads=8, cpu_workers=1,
                            embedding_cache_mb=0)
                spec_file = args.output/(label+'-batch.json')
                spec_file.write_text(json.dumps(spec)+'\n')
                batch_out = args.output/(label+'-batch')
                start = time.monotonic()
                with (args.output/(label+'-batch.log')).open('w') as log:
                    subprocess.run([str(args.adapter), str(spec_file), str(batch_out)], check=True,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=180)
                batch_seconds = time.monotonic()-start
                result = dict(case=label, single_stats=single_stats, batch_process_seconds=batch_seconds,
                              single_vs_batch=compare(ipc, batch_out/'manifest.json', 0, 0),
                              cpu_vs_gpu=compare(reference, ipc, 1e-3, 1e-3),
                              single_ipc=str(ipc), batch_ipc=str(batch_out/'manifest.json'))
                results.append(result)
                print(json.dumps(result), flush=True)
    finally:
        stopped.set(); monitor.join(timeout=10)
        (args.output/'memory.json').write_text(json.dumps(samples, indent=2)+'\n')
    summary = dict(scope='PR mtmd encoder API/IPC; no LLM inference', results=results,
                   gpu0_initial_used_mib=samples[0]['used_mib'] if samples else None,
                   gpu0_peak_used_mib=max(s['used_mib'] for s in samples) if samples else None,
                   single_batch_bitwise_pass=all(r['single_vs_batch']['pass'] for r in results))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    if not summary['single_batch_bitwise_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
