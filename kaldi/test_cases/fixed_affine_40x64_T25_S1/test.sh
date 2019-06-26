#!/bin/bash
nnet3-compute --frames-per-chunk=26 --frame-subsampling-factor=1 model.raw.txt ark:features.ark ark,t:predicted.ark 2>/dev/null
diff outputs.ark predicted.ark || echo FAILED;
rm predicted.ark