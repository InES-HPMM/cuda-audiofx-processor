uplink="enp0s31f6"
downlink="enx0050b606c212"
dl_host_ip="192.168.123.100/24"

ip link set up dev $downlink 
ip addr add $dl_host_ip dev $downlink
iptables -t nat -A POSTROUTING -o $uplink -j MASQUERADE
iptables -A FORWARD -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
iptables -A FORWARD -i $downlink  -o $uplink -j ACCEPT

