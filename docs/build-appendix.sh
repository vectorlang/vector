#!/bin/sh

echo "##Appendix A - Compiler Source Code Listing"
echo

section=1

for infile in $@
do
    bname=$(basename $infile)
    echo "###A.$section $bname"
    echo
    awk '{print "    ", $0}' $infile
    section=$(($section+1))
done
