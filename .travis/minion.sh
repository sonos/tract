#!/bin/bash

set -ex
. $HOME/.minionrc

exec 200>$LOCKFILE || exit 1
flock -n 200 || { echo "WARN: flock() failed." >&2; exit 0; }


mkdir -p $WORKDIR/taskdone/
for task in `aws s3 ls $S3PATH_TASKS/$PLATFORM/ | awk '{ print $4; }'`
do
    cd $HOME
    task_name="${task%.tgz}"
    if [ -e $WORKDIR/taskdone/$task_name ]
    then
        continue
    fi
    echo considering task $task
    rm -rf $WORKDIR/current
    mkdir -p $WORKDIR/current
    cd $WORKDIR/current
    aws s3 cp s3://$S3PATH_TASKS/$PLATFORM/$task .
    tar zxf $task
    . $task_name/vars
    cd $task_name
    (
        ./entrypoint.sh 2> stderr.log > stdout.log || true
    )
    gzip stderr.log
    gzip stdout.log
    aws s3 cp stderr.log.gz s3://$S3PATH_RESULTS/$MINION_ID/$task_name/stderr.log.gz
    aws s3 cp stdout.log.gz s3://$S3PATH_RESULTS/$MINION_ID/$task_name/stdout.log.gz
    touch $WORKDIR/taskdone/$task_name
    cat metrics | sed "s/^/$GRAPHITE_PREFIX.$PLATFORM.$MINION_ID.$TRAVIS_BRANCH_SANE./;s/$/ $TIMESTAMP/" | tr '-' '_' | nc -q 5 $GRAPHITE_HOST $GRAPHITE_PORT
done

sleep 1

echo "DONE"
