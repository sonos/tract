#!/bin/sh

device_name=$1
dataset_name=$2
platform=$3

[ -e venv ] || virtualenv venv
. venv/bin/activate

pip install -r requirements.txt

set -ex
mkdir -p tmp
(
cd tmp
aws s3 cp s3://tract-ci-builds/products/$device_name/$dataset_name.tgz .
tar zxf $dataset_name.tgz
data=`ls -1 $dataset_name.$device_name`
python ../train.py -N 15 --platform=$platform $dataset_name.$device_name/$data $platform.rs
)
mv tmp/$platform.rs .
