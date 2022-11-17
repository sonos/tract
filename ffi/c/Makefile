run: mobilenet mobilenet_v2_1.0.onnx.nnef.tgz
	LD_LIBRARY_PATH=. ./mobilenet

clean:
	rm -f mobilenet libtract.so tract.h dlpack.h

mobilenet: tract.h dlpack.h libtract.so mobilenet.c
	cc mobilenet.c -o mobilenet -L. -ltract

libtract.so:
	cargo build -p tract-ffi --release
	cp ../../target/release/libtract.so .

tract.h:
	cd .. ; cbindgen -l c > c/tract.h

dlpack.h:
	wget https://raw.githubusercontent.com/dmlc/dlpack/main/include/dlpack/dlpack.h

mobilenet_v2_1.0.onnx.nnef.tgz:
	wget https://sfo2.digitaloceanspaces.com/nnef-public/mobilenet_v2_1.0.onnx.nnef.tgz
