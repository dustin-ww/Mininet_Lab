#!/usr/bin/python3

from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import OVSKernelSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI
import time
import argparse
import os

from visualize import visualize

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
    def __init__(self, queue_size=100, **opts):
        self.queue_size = queue_size
        super(SimulationTopo, self).__init__(**opts)

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
        self.addLink(r2, s4, bw=0.5, delay='5ms', max_queue_size=self.queue_size)

def runSingleExperiment(queue_size=100, test_duration=30, iteration=1, result_base_path='./results/tcp_udp_fairness'):
    # Params for experiment
    tcp_duration = test_duration
    udp_start_delay = 5
    udp_duration = test_duration - udp_start_delay

    # Create a directory for this specific iteration
    result_path = f'{result_base_path}/queue_{queue_size}_duration_{test_duration}/iteration_{iteration}'
    os.makedirs(result_path, exist_ok=True) 
    
    # Create topology with specified queue size
    topo = SimulationTopo(queue_size=queue_size)
    net = Mininet(topo=topo, link=TCLink)
    net.start()
    
    s1, s2, s4 = net.get('s1', 's2', 's4')
    server_ip = s4.IP()
    
    # Start iperf3 servers on s4
    # TCP server on port 5201
    s4.cmd(f'iperf3 -s -p 5201 -J > {result_path}/iperf3_tcp_server.log 2>&1 &')
    # Start udp server on port 5202
    s4.cmd(f'iperf3 -s -p 5202 -J > {result_path}/iperf3_udp_server.log 2>&1 &')
    time.sleep(1)
    
    # Ping for monitoring response times and unfairness between TCP (icmp) and UDP flow
    s1.cmd(f'echo "Start pinging: $(date)" > {result_path}/ping_start.log 2>&1')
    s1.cmd(f'ping -i 1 -w {test_duration + 5} {server_ip} > {result_path}/ping_result.log 2>&1 &')
    time.sleep(2)
    
    # s1 -> Runs tcp client
    s1.cmd(f'iperf3 -c {server_ip} -p 5201 -i 5 -t {tcp_duration} -J > {result_path}/iperf3_tcp.json &')
    
    # s2 -> Runs udp client with short delay
    time.sleep(udp_start_delay)
    s2.cmd(f'iperf3 -c {server_ip} -p 5202 -u -b 15M --length 100 -i 5 -t {udp_duration} -J > {result_path}/iperf3_udp.json &')
    
    time.sleep(tcp_duration + 5)
    
    # kill all processes
    s1.cmd('killall ping 2>/dev/null')
    s4.cmd('killall iperf3 2>/dev/null')
    time.sleep(1)
    
    net.stop()
    
    info(f"\n*** Iteration {iteration} completed with parameters:\n")
    info(f"    - Queue size: {queue_size} packets\n")
    info(f"    - Test duration: {test_duration} seconds\n")
    info(f"    - Results saved in {result_path}\n")
    
    return result_path

def runExperiment(queue_size=100, test_duration=30, iterations=1):
    # Create base directory for this experiment configuration
    result_base_path = './results/tcp_udp_fairness'
    os.makedirs(result_base_path, exist_ok=True)
    
    iteration_results = []
    
    info(f"\n*** Starting experiment with {iterations} iterations\n")
    info(f"    - Queue size: {queue_size} packets\n")
    info(f"    - Test duration: {test_duration} seconds per iteration\n")
    
    # Run the experiment for the specified number of iterations
    for i in range(1, iterations + 1):
        info(f"\n*** Running iteration {i}/{iterations}\n")
        result_path = runSingleExperiment(
            queue_size=queue_size, 
            test_duration=test_duration, 
            iteration=i,
            result_base_path=result_base_path
        )
        iteration_results.append(result_path)
        
        # Optional: add a delay between iterations if needed
        if i < iterations:
            info(f"\n*** Waiting 5 seconds before next iteration...\n")
            time.sleep(5)
    
    # Create a summary directory for the entire experiment
    summary_dir = f'{result_base_path}/queue_{queue_size}_duration_{test_duration}_summary'
    os.makedirs(summary_dir, exist_ok=True)
    
    # Write summary information
    with open(f'{summary_dir}/experiment_summary.txt', 'w') as f:
        f.write(f"Experiment Summary\n")
        f.write(f"=================\n")
        f.write(f"Queue Size: {queue_size} packets\n")
        f.write(f"Test Duration: {test_duration} seconds\n")
        f.write(f"Total Iterations: {iterations}\n\n")
        f.write(f"Iteration Directories:\n")
        for i, path in enumerate(iteration_results, 1):
            f.write(f"  Iteration {i}: {path}\n")
    
    info(f"\n*** Full experiment completed with {iterations} iterations\n")
    info(f"    - Queue size: {queue_size} packets\n")
    info(f"    - Test duration: {test_duration} seconds per iteration\n")
    info(f"    - Summary saved in {summary_dir}\n")
    
    # TODO: Add visualization of results
    visualize(iteration_results[-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run network simulation with configurable parameters')
    parser.add_argument('--queue-size', '-q', type=int, default=100,
                        help='Queue size in packets (default: 100)')
    parser.add_argument('--duration', '-d', type=int, default=120,
                        help='Test duration in seconds (default: 120)')
    parser.add_argument('--iterations', '-i', type=int, default=1,
                        help='Number of times to run the experiment (default: 1)')
    args = parser.parse_args()
    
    setLogLevel('info')
    runExperiment(
        queue_size=args.queue_size, 
        test_duration=args.duration,
        iterations=args.iterations
    )