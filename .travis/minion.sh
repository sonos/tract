#!/bin/sh

set -ex

. $HOME/.minionrc

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
    cd $task_name
    (
        . ./vars
        ./entrypoint.sh 2> stderr.log > stdout.log || true
    )
    gzip stderr.log
    gzip stdout.log
    aws s3 cp stderr.log.gz s3://$S3PATH_RESULTS/$MINION_ID/$task_name/stderr.log.gz
    aws s3 cp stdout.log.gz s3://$S3PATH_RESULTS/$MINION_ID/$task_name/stdout.log.gz
    touch $WORKDIR/taskdone/$task_name
    cat metrics | sed "s/^/$GRAPHITE_PREFIX./" | tr '-' '_' | nc $GRAPHITE_HOST $GRAPHITE_PORT
done
