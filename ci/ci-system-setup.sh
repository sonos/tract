#!/bin/sh
set -e

[ -d $ROOT/ci ] || exit 1 "\$ROOT not set correctly '$ROOT'"

sh $ROOT/ci/cpu-features.sh

export RUSTUP_TOOLCHAIN
PATH=$PATH:$HOME/.cargo/bin

# Defined unconditionally (not gated on the once-per-job /tmp/ci-setup-done
# marker below) so every script that sources this file gets apt_retry, even
# when ci-system-setup.sh's own one-time setup already ran in a parent shell.
if [ `uname` != "Darwin" -a "$RUNNER_ENVIRONMENT" != "self-hosted" ]
then
    if [ `whoami` != "root" ]
    then
        SUDO=sudo
    fi
    apt_retry() {
        tries=0
        while [ $tries -lt 3 ]
        do
            timeout 150 $SUDO "$@" && return 0
            tries=$((tries + 1))
            # If the timeout killed apt mid-install, dpkg stays locked in an
            # "interrupted" state and every retry fails immediately.
            $SUDO dpkg --configure -a 2>/dev/null || true
            # azure.archive.ubuntu.com is a known flaky mirror; after the
            # first failure, drop it so the rest of the retries go
            # straight to the canonical mirror instead of stalling again.
            $SUDO sed -i '/azure\.archive\.ubuntu\.com/d' /etc/apt/apt-mirrors.txt 2>/dev/null || true
            $SUDO sed -i 's/azure\.archive\.ubuntu\.com/archive.ubuntu.com/g' /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list 2>/dev/null || true
            sleep 5
        done
        return 1
    }
fi

if [ -n "$CI" -a ! -e /tmp/ci-setup-done -a -z "$TRACT_PREBUILT_CI" ]
then
    if [ `uname` = "Darwin" ]
    then
        python3 --version
        # Self-hosted Macs carry these already; brew's index fetch is a network
        # round trip per job that fails the job when it fails.
        if [ "$RUNNER_ENVIRONMENT" != "self-hosted" ]
        then
            brew install coreutils numpy python-setuptools jq
        fi
        PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH"
        export PYTHON_BIN_PATH=python3
    else
        if [ "$RUNNER_ENVIRONMENT" != "self-hosted" ]
        then
            apt_retry apt-get update
            # apt_retry apt-get upgrade -y
            apt_retry apt-get install -y llvm python3 python3-numpy jq wget curl build-essential sudo clang
        fi
    fi

    which rustup || curl --retry 5 --retry-delay 5 -sSf https://sh.rustup.rs | sh -s -- -y
    rustup update
    rustup toolchain add $RUSTUP_TOOLCHAIN
    [ -n "$GITHUB_PATH" ] && echo $HOME/.cargo/bin >> $GITHUB_PATH

    touch /tmp/ci-setup-done
fi

export TRACT_MODELS_URL=${TRACT_MODELS_URL:-https://tract-test-assets.tract.rs}

if  [ -n "$LARGE_MODELS" ]
then
    export CACHE_FILE=$ROOT/ci/cache_file.sh
    export MODELS=$HOME/.cache/tract-test-assets
    export CACHEDIR=$MODELS
    mkdir -p $MODELS
elif [ -n "$CI" ]
then
    MODELS=$TRACT_MODELS_URL
    CACHE_FILE=true
else
    CACHE_FILE=$ROOT/ci/cache_file.sh
    MODELS=${MODELS:-$HOME/.cache/tract-test-assets}
    export CACHEDIR=${CACHEDIR:-$MODELS}
    mkdir -p $MODELS
fi

if [ -z "$TRACT_RUN" ]
then
    TRACT_RUN="cargo run -p tract-cli $CARGO_EXTRA --profile opt-no-lto --no-default-features --features transformers,pulse,metal --"
    export TRACT_RUN
fi

TRACT_RUNTIMES="-O"
if [ "$(uname)" = "Darwin" ] && (system_profiler SPDisplaysDataType | grep -i "Metal")
then 
    TRACT_RUNTIMES="$TRACT_RUNTIMES --metal"
fi

if which nvidia-smi
then
    TRACT_RUNTIMES="$TRACT_RUNTIMES --cuda"
fi

# Only over the NVIDIA Vulkan driver: elsewhere wgpu would settle for a CPU implementation.
if vulkaninfo --summary 2>/dev/null | grep -q "driverName *= NVIDIA" \
    && $TRACT_RUN list-runtimes 2>/dev/null | grep -qw wgpu
then
    TRACT_RUNTIMES="$TRACT_RUNTIMES --runtime=wgpu"
fi

export TRACT_RUNTIMES
echo $TRACT_RUNTIMES
