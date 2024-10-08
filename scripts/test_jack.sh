# use "cat /proc/asound/cards" to list all soundcards and their IDs
SOUNDCARD_ID="hw:USB"
# SOUNDCARD_ID="hw:CODEC"
INPUT_PORT="system:capture_1"
OUTPUT_PORT_1="system:playback_1"
OUTPUT_PORT_2="system:playback_2"
BUFFER_SIZE=16
SAMPLE_RATE=48000
ISOLATED_CPUS="4-7"
RT_PRIO="99"
PERIODS=2


killall -9 jackd
sleep 2
./jfloorit.sh
/usr/bin/jackd -t2000 -P $RT_PRIO -R -dalsa -r $SAMPLE_RATE -p $BUFFER_SIZE -n $PERIODS -D -C $SOUNDCARD_ID -P $SOUNDCARD_ID &
sleep 2
jack_connect $INPUT_PORT $OUTPUT_PORT_1
# jack_connect $INPUT_PORT $OUTPUT_PORT_2



