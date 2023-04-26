import sys
import os
import subprocess
from pathlib import Path

this_path = Path(__file__).parent.absolute()
exec = os.path.join(this_path, "utils", "calc_shape.R")

cmd = f"Rscript {exec} {sys.argv[1]}"
res = subprocess.run(cmd, shell=True)
if res.returncode != 0:
    sys.exit("ERROR: running\n\n{cmd}\n\nfailed.")

