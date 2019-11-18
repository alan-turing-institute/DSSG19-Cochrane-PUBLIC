#!/usr/bin/python
# Usage: remove all leading/trailing quotes.
# sys.argv[1] : name of file in which we want to remove quotes from all cells
# sys.argv[2] : name of new file to output
import sys

if len(sys.argv) != 3:
    print("Usage: remove all leading/trailing quotes.")
else:
    with open(sys.argv[1], 'r') as fin, open(sys.argv[2], 'w') as fout:
        for line in fin:
            fout.write(line.replace('"', '').replace("'", ""))
