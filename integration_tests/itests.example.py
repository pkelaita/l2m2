import os

# Not passing this flag assumes l2m2 has been installed locally from dist/
# If this is not the case, run `make build` in the root directory and
# pip install the built package
if any(arg in os.sys.argv for arg in ["--local", "-l"]):
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    root = file.parents[1]
    sys.path.append(str(root))


import l2m2

print("L2M2 Version:", (l2m2).__version__)

# Copy this file to ./itests.py, then write your integration tests here.
