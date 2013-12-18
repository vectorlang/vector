#!/bin/bash

for file in $@
do
    cat $file
    echo
    echo "\\pagebreak"
    echo
done
