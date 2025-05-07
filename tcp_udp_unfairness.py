#!/usr/bin/python3

from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import OVSKernelSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI
import time

# This script sets up a Mininet topology with two TCP and one UDP flows.
# In detail this mininet topology is rebuild from Sally Floyd's and Kevin Fall's famous paper "Promoting the Use of End-to-End Congestion Control in the Internet" (1999).
# The topology is as follows:
""" 
     10 Mbps, 2ms              1.5 Mbps, 3ms                10 Mbps, 10ms
   +-------------+           +------------+              +-------------+
   |     s1      |-----------|     R1     |--------------|     R2      |-----------|     s3      |
   +-------------+           +------------+              +-------------+           +-------------+
        |                          |                                              
        | 10 Mbps, 3ms             |                                              
        |                          |                                              
   +-------------+                [Routing/Forwarding]             0.5 Kbps, 5ms (X)
   |     s2      |---------------------------------------------------------------|     s4      |
   +-------------+                                                                +-------------+
 """

class SimulationTopo(Topo):
    def build(self):
        # Hosts
        s1 = self.addHost('s1', ip='10.0.0.1/24')
        s2 = self.addHost('s2', ip='10.0.0.2/24')
        s3 = self.addHost('s3', ip='10.0.0.3/24')
        s4 = self.addHost('s4', ip='10.0.0.4/24')

        # Routers (as switches)
        r1 = self.addSwitch('r1', cls=OVSKernelSwitch)
        r2 = self.addSwitch('r2', cls=OVSKernelSwitch)

        # Links
        self.addLink(s1, r1, bw=10, delay='2ms')
        self.addLink(s2, r1, bw=10, delay='3ms')
        self.addLink(r1, r2, bw=1.5, delay='3ms')
        self.addLink(r2, s3, bw=10, delay='10ms')
        self.addLink(r2, s4, bw=0.5, delay='5ms', max_queue_size=5000) 

def runExperiment():
    topo = SimulationTopo()
    net = Mininet(topo=topo, link=TCLink)
    net.start()

    s1, s2, s4 = net.get('s1', 's2', 's4')
    server_ip = s4.IP()

    # Start iperf server on s4
    s4.cmd(f'iperf -s -D > /tmp/iperf_server.log 2>&1')
    time.sleep(1)

    # Ping for monitoring response times and unfairness between TCP (icmp) and UDP flow
    s1.cmd(f'echo "Start pinging: $(date)" > /tmp/ping_results.log')
    s1.cmd(f'ping -i 1 -w 65 {server_ip} >> /tmp/ping_results.log 2>&1 &')
    time.sleep(2)

    test_duration = 120 
    tcp_duration = test_duration
    udp_start_delay = 5
    udp_duration = test_duration - udp_start_delay

    # s1 -> Runs tcp client
    s1.cmd(f'iperf -c {server_ip} -i 5 -t {tcp_duration} -J > /tmp/iperf_tcp.json &')

    # a2 -> Runs udp client with short delay
    time.sleep(udp_start_delay)
    s2.cmd(f'iperf -c {server_ip} -u -b 15M -l 100 -i 5 -t {udp_duration} -J > /tmp/iperf_udp.json &')

    time.sleep(tcp_duration + 5)

    # kill all
    s1.cmd('killall ping 2>/dev/null')
    s4.cmd('killall iperf 2>/dev/null')
    time.sleep(1)

    net.stop()
    info("\n*** Results saved in /tmp/\n")

if __name__ == '__main__':
    setLogLevel('info')
    runExperiment()
