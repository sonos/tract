// vim: ft=arm

// serves as a canary build file to figure out which flag combination will accept half precision fmla 

.text
.align 4

// .cpu generic+fp+simd+fp16
.global foo
foo:
    fmla        v16.8h, v0.8h, v8.h[0]
    ret

