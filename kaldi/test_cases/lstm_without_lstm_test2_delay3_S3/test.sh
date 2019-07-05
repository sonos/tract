#!/bin/bash
nnet3-compute --frames-per-chunk=11 --frame-subsampling-factor=3 model.raw.txt ark:input.ark ark,t:predicted.ark 2>.error
fail=0
diff output.ark predicted.ark || fail=1;
rm predicted.ark
echo "fail=$fail" && [ $fail -gt 0 ] && exit 1;
