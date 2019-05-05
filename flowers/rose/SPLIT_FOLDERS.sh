#!/bin/bash
# outnum generates the name of the output directory
outnum=1
# n is the number of files we have moved
n=0
# Go through all JPG files in the current directory
for jpg in pwd; do
   # Create new output directory if first of new batch of 2000
   if [ $n -eq 0 ]; then
      outdir=roses$outnum
      mkdir $outdir
      ((outnum++))
   fi
   # Move the file to the new subdirectory
   mv "$jpg" "$outdir"

   # Count how many we have moved to there
   ((n++))

   # Start a new output directory if we have sent 2000
   [ $n -eq 100 ] && n=0
done
