#!/bin/sh

set -ex

ANDROID_SDK_VERSION=4333796

which java || (
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:webupd8team/java
    apt-get update
    echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections
    echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections
    apt-get install -y oracle-java8-installer ca-certificates-java python
)

ANDROID_SDK=$HOME/cached/android-sdk
if [ ! -d "$ANDROID_SDK" ]
then
    mkdir -p $ANDROID_SDK
    cd $ANDROID_SDK
    curl -s -o android-sdk.zip \
      "https://dl.google.com/android/repository/sdk-tools-linux-${ANDROID_SDK_VERSION}.zip"
    unzip -q android-sdk.zip
    rm android-sdk.zip
fi

export JAVA_OPTS='-XX:+IgnoreUnrecognizedVMOptions'
yes | $ANDROID_SDK/tools/bin/sdkmanager --licenses > /dev/null

$ANDROID_SDK/tools/bin/sdkmanager \
    "build-tools;28.0.3" "platform-tools" "platforms;android-28" "tools" "ndk-bundle" \
    > /dev/null
