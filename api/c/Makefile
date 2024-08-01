all: tract.h libtract.a mobilenet

mobilenet: tract.h libtract.a mobilenet.c
	cc -g mobilenet.c -o mobilenet -L. -I. -ltract -lm -lpthread -ldl

libtract.a:
	cargo build -p tract-ffi --release
	cp ../../target/release/libtract.a .

tract.h:
	cd ../ffi ; cbindgen -l c > tract.h
	cp ../ffi/tract.h .

clean:
	rm -f mobilenet libtract.a tract.h