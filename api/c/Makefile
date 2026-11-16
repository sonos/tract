run: mobilenet mobilenet_v2_1.0.onnx.nnef.tgz hey_snips hey_snips.nnef.tar hey-snips-tts.wav
	LD_LIBRARY_PATH=. ./mobilenet
	LD_LIBRARY_PATH=. ./hey_snips

clean:
	rm -f mobilenet hey_snips libtract.so tract.h

mobilenet: tract.h libtract.so mobilenet.c
	cc mobilenet.c -o mobilenet -L. -ltract

hey_snips: tract.h libtract.so hey_snips.c
	cc hey_snips.c -o hey_snips -L. -ltract

libtract.so:
	cargo build -p tract-ffi --profile opt-no-lto
	cp ../../target/opt-no-lto/libtract.so .

tract.h:
	cd ../ffi ; cbindgen -l c > tract.h
	cp ../ffi/tract.h .

mobilenet_v2_1.0.onnx.nnef.tgz:
	wget -q https://s3.amazonaws.com/tract-ci-builds/tests/mobilenet_v2_1.0.onnx.nnef.tgz

hey_snips.nnef.tar:
	wget -q https://tract-test-assets.tract.rs/hey-snips/v1/hey_snips.nnef.tar

hey-snips-tts.wav:
	wget -q https://tract-test-assets.tract.rs/e2e/hey-snips-tts.wav
