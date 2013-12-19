#!/bin/sh

guess_language () {
    case $1 in
        *.ml|*.mll|*.mly) echo ocaml;;
        *.hpp|*.cpp|*.cu|*.vec) echo cpp;;
        *.sh) echo shell;;
        SConstruct|*SConscript) echo python;;
    esac
}

echo "##Appendix A - Compiler Source Code Listing"
echo

section=1

for infile in $@
do
    echo "###A.$section $infile"
    echo
    echo "\`\`\`$(guess_language "$infile")"
    cat $infile
    echo '```'
    section=$(($section+1))
done
