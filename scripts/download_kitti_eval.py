#!/usr/bin/env python3
"""Download the KITTI odometry test data the benchmark uses (sequences 09 and 10), laid out for scripts/eval.py:

    <out>/sequences/{09,10}/{image_2,image_3,velodyne}/
    <out>/calib/{09,10}/calib.txt

    python scripts/download_kitti_eval.py --out /data/kitti_vp2p_eval_data

Only these files are read out of KITTI's odometry archives, with HTTP range requests (about 10 GB). Interrupted runs
resume. KITTI is CC BY-NC-SA 3.0 (https://www.cvlibs.net/datasets/kitti/); by downloading you accept its terms.
"""

import argparse
import io
import os
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
SEQS = ("09", "10")


class HttpFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def open_zip(name: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpFile(BASE + name), buffer_size=1 << 20))


def fetch(zip_name: str, pairs: list, workers: int):
    """Extract (member, destination) pairs from a remote zip, skipping finished files, in parallel with retries."""
    todo = [(m, d) for m, d in pairs if not os.path.exists(d)]
    print(f"{zip_name}: {len(todo)} files to download ({len(pairs) - len(todo)} already there)", flush=True)
    done = [0]

    def run(chunk):
        for attempt in range(20):
            try:
                with open_zip(zip_name) as z:
                    while chunk:
                        member, dst = chunk[0]
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        with open(dst + ".part", "wb") as f:
                            f.write(z.read(member))
                        os.replace(dst + ".part", dst)
                        chunk.pop(0)
                        done[0] += 1
                        if done[0] % 500 == 0:
                            print(f"  {done[0]} / {len(todo)}", flush=True)
                return
            except Exception as e:  # dropped connection: reopen and continue where it stopped
                print(f"  retry {attempt + 1}: {type(e).__name__}: {e}", flush=True)
                time.sleep(min(60, 5 * (attempt + 1)))
        raise RuntimeError(f"{zip_name}: gave up with {len(chunk)} files left")

    chunks = [todo[i::workers] for i in range(workers)]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, [c for c in chunks if c]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Output folder, passed to scripts/eval.py --data")
    p.add_argument("--workers", type=int, default=16, help="Parallel connections (default: 16)")
    p.add_argument("--limit", type=int, default=0, help="Only the first N frames per sequence, for a quick test")
    args = p.parse_args()
    out = args.out

    fetch("data_odometry_calib.zip",
          [(f"dataset/sequences/{s}/calib.txt", os.path.join(out, "calib", s, "calib.txt")) for s in SEQS], 1)

    for zip_name, folders in (("data_odometry_color.zip", ("image_2", "image_3")),
                              ("data_odometry_velodyne.zip", ("velodyne",))):
        with open_zip(zip_name) as z:
            names = z.namelist()
        pairs = []
        for s in SEQS:
            for folder in folders:
                prefix = f"dataset/sequences/{s}/{folder}/"
                members = sorted(n for n in names if n.startswith(prefix) and not n.endswith("/"))
                if args.limit:
                    members = members[:args.limit]
                pairs += [(m, os.path.join(out, "sequences", s, folder, m[len(prefix):])) for m in members]
        fetch(zip_name, pairs, args.workers)

    for s in SEQS:
        counts = {f: len(os.listdir(os.path.join(out, "sequences", s, f))) for f in ("image_2", "image_3", "velodyne")}
        print(f"sequence {s}: {counts}")
    print("done:", out)


if __name__ == "__main__":
    main()
