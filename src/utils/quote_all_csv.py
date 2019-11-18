# Usage: add quotes to all fields in a csv.
# sys.argv[1] : name of file in which we want to quote all cells
# sys.argv[2] : name of new file to output

import pandas as pd
import csv
import sys

if len(sys.argv) != 3:
    print("Usage: add quotes to all fields in a csv.")
else:
    fin = pd.read_csv(sys.argv[1])
    fin.to_csv(sys.argv[2], index=False, quoting=csv.QUOTE_ALL)
