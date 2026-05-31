

import sys

for line in sys.stdin.readlines()[2:]:
  (row,col,value) = line.strip().split()
  if row==col:
    print row,col,value
  else:
    print row,col,value
    print col,row,value
