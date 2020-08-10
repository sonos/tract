#!/bin/sh

set -x

for bundle in *.txt
do
    prefix=$(echo $bundle | sed s/\.txt// | tr '.-' '__')
    cargo test -- --ignored $prefix::optim:: | grep '::optim::.* ok'  | cut -d ' ' -f 2 | sed "s/::/:/g" | cut -d : -f 3 > /tmp/$bundle
done

for bundle in *.txt
do
    cat $bundle >> /tmp/$bundle
    sort /tmp/$bundle > $bundle
    rm /tmp/$bundle
done


for bundle in *.txt
do
    cargo test -- --ignored $prefix::plain:: | grep '::plain::.* ok'  | cut -d ' ' -f 2 | sed "s/::/:/g" | cut -d : -f 3 | sed 's/$/ dynsize/'> /tmp/$bundle
done

for bundle in *.txt
do
    cat $bundle >> /tmp/$bundle
    sort /tmp/$bundle > $bundle
    rm /tmp/$bundle
done
