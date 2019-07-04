#!/bin/bash
nnet3-compute --frames-per-chunk=31 --frame-subsampling-factor=2 model.raw.txt ark:input.ark ark,t:predicted.ark 2>.error
fail=0
diff output.ark predicted.ark || fail=1;
rm predicted.ark
echo "fail=$fail" && [ $fail -gt 0 ] && exit 1;
