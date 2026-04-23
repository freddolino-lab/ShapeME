#!/usr/bin/env python3
import subprocess, sys, pathlib, os, shutil

features = ["EP", "HelT", "MGW", "ProT", "Roll"]
DEEPDNASHAPE = "/opt/conda/envs/deepdnashape/bin/deepDNAshape"

def run(fasta):
    fasta = str(pathlib.Path(fasta).resolve())

    for feat in features:
        # ShapeME expects this name:
        out_final = f"{fasta}.{feat}"

        # DeepDNAshape needs .fa or .fasta to emit FASTA:
        out_tmp = f"{out_final}.fa"

        cmd = [
            DEEPDNASHAPE,
            "--file", fasta,
            "--feature", feat,
            "--output", out_tmp,
        ]

        p = subprocess.run(cmd, text=True, capture_output=True)

        if p.returncode != 0:
            raise RuntimeError(
                f"Command failed: {' '.join(cmd)}\n"
                f"stderr:\n{p.stderr}\n"
                f"stdout:\n{p.stdout}\n"
            )

        if (not os.path.exists(out_tmp)) or os.path.getsize(out_tmp) == 0:
            raise RuntimeError(
                f"DeepDNAshape produced empty/missing output file: {out_tmp}\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{p.stderr}\n"
                f"stdout:\n{p.stdout}\n"
            )

        # rename to what ShapeME expects
        os.replace(out_tmp, out_final)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: calc_deepdnashape.py <fasta>", file=sys.stderr)
        sys.exit(2)
    run(sys.argv[1])
