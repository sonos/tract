#!/bin/sh

set -ex

which java || sudo apt install -y default-jdk

ANDROID_SDK=$HOME/cached/android-sdk
if [ ! -d "$ANDROID_SDK" ]
then
    mkdir -p $ANDROID_SDK
    cd $ANDROID_SDK

      # ANDROID_SDK_VERSION=4333796
      # "https://dl.google.com/android/repository/sdk-tools-linux-${ANDROID_SDK_VERSION}.zip"

    curl -s -o android-sdk.zip \
       https://dl.google.com/android/repository/commandlinetools-linux-8092744_latest.zip
    unzip -q android-sdk.zip
    rm android-sdk.zip
fi

yes | $ANDROID_SDK/cmdline-tools/bin/sdkmanager --sdk_root=$ANDROID_SDK --licenses > /dev/null

$ANDROID_SDK/cmdline-tools/bin/sdkmanager --sdk_root=$ANDROID_SDK \
    "build-tools;30.0.0" "platform-tools" "platforms;android-31" "tools" "ndk-bundle" \
    > /dev/null
