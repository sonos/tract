run: mobilenet mobilenet_v2_1.0.onnx.nnef.tgz
	LD_LIBRARY_PATH=. ./mobilenet

clean:
	rm -f mobilenet libtract.so tract.h

mobilenet: tract.h libtract.so mobilenet.c
	cc mobilenet.c -o mobilenet -L. -ltract

libtract.so:
	cargo build -p tract-ffi --release
	cp ../../target/release/libtract.so .

tract.h:
	cd ../ffi ; cbindgen -l c > tract.h
	cp ../ffi/tract.h .

mobilenet_v2_1.0.onnx.nnef.tgz:
	wget -q https://sfo2.digitaloceanspaces.com/nnef-public/mobilenet_v2_1.0.onnx.nnef.tgz
