import sys
import os
import subprocess
from pathlib import Path

this_path = Path(__file__).parent.absolute()
exec = os.path.join(this_path, "utils", "calc_shape.R")

cmd = f"Rscript {exec} {sys.argv[1]}"
res = subprocess.run(cmd, shell=True, capture_output=True)
if res.returncode != 0:
    sys.exit(
        f"ERROR: running\n\n{cmd}\n\nfailed with following stderr\n\n"\
        f"{res.stderr.decode()}\n\n"\
        f"and the following stdout\n\n"\
        f"{res.stdout.decode()}"
    )

