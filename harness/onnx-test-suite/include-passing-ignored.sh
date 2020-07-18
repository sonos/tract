#!/bin/sh

set -x

for bundle in *.txt
do
    prefix=$(echo $bundle | sed s/\.txt// | tr '.-' '__')
    cargo test -- --ignored $prefix::optim:: | grep '::optim::.* ok'  | cut -d ' ' -f 2 | sed "s/::/:/g" | cut -d : -f 3 > /tmp/new
    cat /tmp/new 
    cat $bundle >> /tmp/new
    sort /tmp/new > $bundle
    rm /tmp/new
    cargo test -- --ignored $prefix::plain:: | grep '::plain::.* ok'  | cut -d ' ' -f 2 | sed "s/::/:/g" | cut -d : -f 3 | sed 's/$/ dynsize/'> /tmp/new
    cat /tmp/new 
    cat $bundle >> /tmp/new
    sort /tmp/new > $bundle
    rm /tmp/new
done
