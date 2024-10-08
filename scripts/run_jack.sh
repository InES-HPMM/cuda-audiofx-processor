#!/bin/sh

SOUNDCARD_ID="hw:CODEC"
SAMPLE_RATE=48000
ISOLATED_CPUS="4-7"
RT_PRIO="99"
PERIODS=2

./jfloorit.sh

killall -9 jackd
sleep 2
taskset -c $ISOLATED_CPUS /usr/bin/jackd -t2000 -P $RT_PRIO -dalsa -r $SAMPLE_RATE -n $PERIODS -Xnone -D -C $SOUNDCARD_ID -P $SOUNDCARD_ID &
