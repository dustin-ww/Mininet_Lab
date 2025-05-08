#!/usr/bin/env python3
import polars as pl
import json
import matplotlib.pyplot as plt
import numpy as np
import os

MSS_BYTES = 1460  # TCP Maximum Segment Size

def parse_ping_log(file_path):
    import numpy as np
    with open(file_path, 'r') as f:
        lines = f.readlines()

    seq_to_latency = {}
    for line in lines[1:]:
        if 'icmp_seq=' in line and 'time=' in line:
            try:
                parts = line.strip().split()
                seq_part = next((p for p in parts if p.startswith('icmp_seq=')), None)
                time_part = next((p for p in parts if p.startswith('time=')), None)
                if not seq_part or not time_part:
                    continue
                seq = int(seq_part.split('=')[1])
                latency = float(time_part.split('=')[1].replace('ms', ''))
                seq_to_latency[seq] = latency
            except (ValueError, IndexError, AttributeError) as e:
                print(f"Error while parsing: {line.strip()} -- {e}")
                continue

    max_seq = max(seq_to_latency.keys()) if seq_to_latency else 0
    exp_times = list(range(1, max_seq + 1))
    latencies = [seq_to_latency.get(seq, np.nan) for seq in exp_times]

    return pl.DataFrame({'experiment_time': exp_times, 'latency_ms': latencies})

def parse_iperf_json(file_path, offset, is_udp=False):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error while file read {file_path}: {e}")
        return pl.DataFrame({'experiment_time': [], 'throughput_mbps': [], 'drop_rate_pct': [], 'retransmits': []})

    times, throughputs, drop_rates, retransmits = [], [], [], []

    for interval in data.get('intervals', []):
        sum_data = interval.get('sum', {})
        duration = sum_data.get('end', 0) - sum_data.get('start', 0)
        if duration <= 0:
            continue
        t = offset + sum_data['end']
        tp = (sum_data.get('bytes', 0) * 8) / (duration * 1e6)
        times.append(t)
        throughputs.append(tp)
        rtx = sum_data.get('retransmits', 0) if not is_udp else 0
        retransmits.append(rtx)

        if is_udp:
            lost, pkts = 0, 0
            for st in interval.get('streams', []):
                udp = st.get('udp', {})
                lost += udp.get('lost_packets', 0)
                pkts += udp.get('packets', 0)
            drop_pct = (lost / (lost + pkts) * 100) if (lost + pkts) > 0 else 0
        else:
            packets = sum_data.get('bytes', 0) / MSS_BYTES if sum_data.get('bytes', 0) > 0 else 0
            drop_pct = (rtx / packets * 100) if packets > 0 else 0
        drop_rates.append(drop_pct)

    if 'end' in data and 'sum' in data['end']:
        end = data['end']['sum']
        t = offset + end.get('end', 0)
        tp = end.get('bits_per_second', 0) / 1e6
        times.append(t)
        throughputs.append(tp)
        if is_udp:
            drop_rates.append(end.get('lost_percent', 0))
            retransmits.append(0)
        else:
            rtx = end.get('retransmits', 0)
            retransmits.append(rtx)
            packets = end.get('bytes', 0) / MSS_BYTES if end.get('bytes', 0) > 0 else 0
            drop_rates.append((rtx / packets * 100) if packets > 0 else 0)

    return pl.DataFrame({
        'experiment_time': times,
        'throughput_mbps': throughputs,
        'drop_rate_pct': drop_rates,
        'retransmits': retransmits
    })

# -------------------------
# Main Function
# -------------------------
def visualize(path, name='network_metrics_extended'):
    ping_log_path = os.path.join(path, 'ping_result.log')
    tcp_json_path = os.path.join(path, 'iperf3_tcp.json')
    udp_json_path = os.path.join(path, 'iperf3_udp.json')
    output_path = os.path.join(path, f'{name}.png')

    print(f"Verarbeite Dateien in: {path}")
    ping_df = parse_ping_log(ping_log_path)
    tcp_df = parse_iperf_json(tcp_json_path, offset=3, is_udp=False)
    udp_df = parse_iperf_json(udp_json_path, offset=8, is_udp=True)

    print(f"Pings: {len(ping_df)}, TCP: {len(tcp_df)}, UDP: {len(udp_df)}")

    plt.figure(figsize=(12, 10))

    # 1) Latency
    plt.subplot(4, 1, 1)
    if len(ping_df) > 0:
        valid_pings = ping_df.filter(~pl.col("latency_ms").is_nan())
        plt.plot(valid_pings['experiment_time'], valid_pings['latency_ms'], label='Ping Latenz', color='blue')
        high_latency = ping_df.filter(pl.col("latency_ms") > 2000)
        if len(high_latency) > 0:
            plt.scatter(high_latency["experiment_time"], high_latency["latency_ms"], color='red', label='Extrem hohe Latenz')
        lost_packets = ping_df.filter(pl.col("latency_ms").is_nan())
        if len(lost_packets) > 0:
            plt.scatter(lost_packets['experiment_time'], [plt.gca().get_ylim()[1] * 0.95] * len(lost_packets), marker='x', color='red', label='Paketverlust')
    plt.ylabel('Latenz (ms)')
    plt.title('Netzwerkmetriken')
    plt.grid(True)
    plt.legend()

    # 2) TCP/UDP Throughput
    plt.subplot(4, 1, 2)
    if len(tcp_df) > 0:
        plt.plot(tcp_df['experiment_time'], tcp_df['throughput_mbps'], label='TCP Durchsatz')
    if len(udp_df) > 0:
        plt.plot(udp_df['experiment_time'], udp_df['throughput_mbps'], label='UDP Durchsatz')
    plt.ylabel('Durchsatz (Mbps)')
    plt.grid(True)
    plt.legend()

    # 3) TCP Retransmits and drop-rate
    plt.subplot(4, 1, 3)
    if len(tcp_df) > 0:
        plt.plot(tcp_df['experiment_time'], tcp_df['retransmits'], label='TCP Retransmits')
        plt_twin = plt.twinx()
        plt_twin.plot(tcp_df['experiment_time'], tcp_df['drop_rate_pct'], label='TCP Drop-Rate (%)', color='red', linestyle='--')
        plt_twin.set_ylabel('Drop-Rate (%)', color='red')
        plt_twin.tick_params(axis='y', labelcolor='red')
    plt.ylabel('Retransmits')
    plt.grid(True)
    plt.legend()

    # 4) UDP drop-rate
    plt.subplot(4, 1, 4)
    if len(udp_df) > 0:
        plt.plot(udp_df['experiment_time'], udp_df['drop_rate_pct'], label='UDP Drop-Rate (%)')
    plt.ylabel('Drop-Rate (%)')
    plt.xlabel('Experimentdauer (s)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Diagram saved in: {output_path}")
    plt.show()

# CLI Usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize network metrics from iperf3 and ping logs')
    parser.add_argument('--path', '-p', type=str, default='/tmp', help='Path to the directory containing the log files')
    parser.add_argument('--name', '-n', type=str, default='network_metrics_extended', help='Name for the output image file')
    args = parser.parse_args()
    visualize(args.path, args.name)
