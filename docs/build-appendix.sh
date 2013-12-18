#!/bin/sh

guess_language () {
    case $1 in
        *.ml|*.mll|*.mly) echo ocaml;;
        *.hpp|*.cpp|*.cu) echo cpp;;
    esac
}

echo "##Appendix A - Compiler Source Code Listing"
echo

section=1

for infile in $@
do
    bname=$(basename $infile)
    echo "###A.$section $bname"
    echo
    echo "\`\`\`$(guess_language "$infile")"
    cat $infile
    echo '```'
    section=$(($section+1))
done
