FROM debian:stretch

COPY sources.list /etc/apt/sources.list
RUN apt-get -y update ; apt-get -y upgrade ; apt-get install -y curl wget build-essential # gcc-arm-linux-gnueabihf binutils-arm-linux-gnueabihf
# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# ENV PATH $PATH:/root/.cargo/bin
# RUN rustup target install armv7-unknown-linux-gnueabihf
