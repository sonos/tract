run: mobilenet
	LD_LIBRARY_PATH=. ./mobilenet ../../../inputs/mobilenetv2-1.0/mobilenetv2-1.0.onnx

clean:
	rm -f mobilenet libtract.a tract.h

mobilenet: tract.h libtract.a mobilenet.c
	cc -g mobilenet.c -o mobilenet -L. -ltract -lm -lpthread

libtract.so:
	cargo build -p tract-ffi --release
	cp ../../target/release/libtract.a .

tract.h:
	cd ../ffi ; cbindgen -l c > tract.h
	cp ../ffi/tract.h .
