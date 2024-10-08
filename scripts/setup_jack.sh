sudo apt install jackd2 libjack-jack2-dev
sudo sysctl -w kernel.sched_rt_runtime_us=-1 # disable real-time scheduling group check. needs to be run after every restart TODO:find permanent solution