# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
"""Download Oxford Spires from the Hugging Face Hub: the images and VILENS-SLAM outputs of every sequence, then the
core sequences in full.

    python -m wai_processing.scripts.conversion.download_spires --out /raw/oxford_spires
"""

import argparse

from huggingface_hub import snapshot_download

hf_repo_id = "ori-drs/oxford_spires_dataset"

allow_patterns = [
    "sequences/*/raw/images.zip",
    "sequences/*/processed/vilens-slam/**",
]

core_sequences = [
    "sequences/2024-03-12-keble-college-02/*",
    "sequences/2024-03-12-keble-college-03/*",
    "sequences/2024-03-12-keble-college-04/*",
    "sequences/2024-03-12-keble-college-05/*",
    "sequences/2024-03-13-observatory-quarter-01/*",
    "sequences/2024-03-13-observatory-quarter-02/*",
    "sequences/2024-03-14-blenheim-palace-01/*",
    "sequences/2024-03-14-blenheim-palace-02/*",
    "sequences/2024-03-14-blenheim-palace-05/*",
    "sequences/2024-03-18-christ-church-01/*",
    "sequences/2024-03-18-christ-church-02/*",
    "sequences/2024-03-18-christ-church-03/*",
    "sequences/2024-03-18-christ-church-05/*",
    "sequences/2024-05-20-bodleian-library-02/*",
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Folder to download into")
    local_dir = p.parse_args().out

    snapshot_download(repo_id=hf_repo_id, allow_patterns=allow_patterns, local_dir=local_dir, repo_type="dataset",
                      use_auth_token=False)
    for sequence in core_sequences:
        print(f"Downloading sequence: {sequence}")
        snapshot_download(repo_id=hf_repo_id, allow_patterns=sequence, local_dir=local_dir, repo_type="dataset",
                          use_auth_token=False)


if __name__ == "__main__":
    main()
