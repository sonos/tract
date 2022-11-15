#!/bin/sh

set -e

sudo apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

if [ ! -d $HOME/anaconda3 ]
then
    ANACONDA_SETUP=Anaconda3-2022.10-Linux-x86_64.sh
    [ -e $ANACONDA_SETUP ] || wget -q https://repo.anaconda.com/archive/$ANACONDA_SETUP
    bash $ANACONDA_SETUP -b
fi

. $HOME/anaconda3/bin/activate
echo $CONDA_EXE
if [ ! -d $HOME/anaconda3/envs/tf_37 ]
then
    conda env create -q -f environment.yml
fi
conda activate tf_37

cd `dirname $0`
jupyter nbconvert --to notebook --inplace --execute simple_model.ipynb
cargo run
cargo clean

if [ -n "$CI" ]
then
    conda deactivate
    conda env remove -n tf_37
fi
