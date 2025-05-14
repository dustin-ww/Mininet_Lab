#!/usr/bin/env python3
import polars as pl
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import glob

MSS_BYTES = 1460  # TCP Maximum Segment Size

def parse_ping_log(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Warning: File not found - {file_path}")
        return pl.DataFrame({'experiment_time': [], 'latency_ms': []})

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
                print(f"Fehler beim Parsen: {line.strip()} -- {e}")
                continue

    max_seq = max(seq_to_latency.keys()) if seq_to_latency else 0
    exp_times = list(range(1, max_seq + 1))
    latencies = [seq_to_latency.get(seq, np.nan) for seq in exp_times]

    return pl.DataFrame({'experiment_time': exp_times, 'latency_ms': latencies})

def parse_iperf_json(file_path, offset, is_udp=False, is_server=False):
    """Parse iperf3 JSON output with enhanced metrics for TCP/UDP analysis"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error while reading the file {file_path}: {e}")
        return pl.DataFrame({
            'experiment_time': [], 
            'throughput_mbps': [], 
            'drop_rate_pct': [], 
            'retransmits': [],
            'jitter_ms': [], 
            'received_bytes': []
        })

    # Initialize data collection lists
    times = []
    throughputs = []
    drop_rates = []
    retransmits = []
    jitters = []
    received_bytes = []
    window_sizes = []
    rtts = []
    cwnd_sizes = []
    inter_arrival_times = []
    last_time = None

    # Process intervals
    for interval in data.get('intervals', []):
        sum_data = interval.get('sum', {})
        duration = sum_data.get('end', 0) - sum_data.get('start', 0)
        if duration <= 0:
            continue

        # Basic metrics
        t = float(offset + sum_data['end'])
        tp = float((sum_data.get('bytes', 0) * 8) / (duration * 1e6))
        times.append(t)
        throughputs.append(tp)
        
        # Calculate inter-arrival time
        current_time = float(offset + sum_data['end'])
        if last_time is not None:
            inter_arrival_times.append(current_time - last_time)
        last_time = current_time

        # Protocol specific processing
        if is_udp:
            # UDP specific metrics
            lost, pkts = 0, 0
            for st in interval.get('streams', []):
                udp = st.get('udp', {})
                lost += int(udp.get('lost_packets', 0))
                received = int(udp.get('packets', 0))
                pkts += (lost + received)
            drop_pct = float(lost / pkts * 100) if pkts > 0 else 0.0
            retransmits.append(0)  # UDP doesn't have retransmissions
        else:
            # TCP specific metrics
            rtx = int(sum_data.get('retransmits', 0))
            retransmits.append(rtx)
            packets = int(sum_data.get('bytes', 0) / MSS_BYTES) if sum_data.get('bytes', 0) > 0 else 0
            drop_pct = float(rtx / packets * 100) if packets > 0 else 0.0
            
            # Extract TCP specific info
            tcp_info = interval.get('tcp', {})
            window_sizes.append(float(tcp_info.get('window_size', 0)))
            rtts.append(float(tcp_info.get('rtt', 0)))
            cwnd_sizes.append(float(tcp_info.get('cwnd', 0)))
        
        drop_rates.append(drop_pct)

        # Server specific metrics
        if is_server:
            if is_udp:
                jitter_val = float(0)
                for st in interval.get('streams', []):
                    udp = st.get('udp', {})
                    jitter_val = float(udp.get('jitter_ms', 0))
                jitters.append(jitter_val)
            else:
                jitters.append(float(0))
            received_bytes.append(int(sum_data.get('bytes', 0)))

    # Process end summary
    if 'end' in data and 'sum' in data['end']:
        end = data['end']['sum']
        t = float(offset + end.get('end', 0))
        tp = float(end.get('bits_per_second', 0) / 1e6)
        times.append(t)
        throughputs.append(tp)
        
        if is_server:
            if is_udp:
                jitter_val = float(0)
                if 'streams' in data['end']:
                    for st in data['end']['streams']:
                        udp = st.get('udp', {})
                        jitter_val = float(udp.get('jitter_ms', 0))
                jitters.append(jitter_val)
            else:
                jitters.append(float(0))
            received_bytes.append(int(end.get('bytes', 0)))
        
        if is_udp:
            drop_rates.append(float(end.get('lost_percent', 0)))
            retransmits.append(0)
        else:
            rtx = int(end.get('retransmits', 0))
            retransmits.append(rtx)
            packets = int(end.get('bytes', 0) / MSS_BYTES) if end.get('bytes', 0) > 0 else 0
            drop_pct = float(rtx / packets * 100) if packets > 0 else 0.0
            drop_rates.append(drop_pct)

    # Add placeholder for first inter-arrival time
    if inter_arrival_times:
        inter_arrival_times.insert(0, inter_arrival_times[0])

    # Ensure all series have the same length
    series_lengths = {
        'times': len(times),
        'throughputs': len(throughputs),
        'drop_rates': len(drop_rates),
        'retransmits': len(retransmits),
        'inter_arrival_times': len(inter_arrival_times),
        'window_sizes': len(window_sizes) if not is_udp else 0,
        'rtts': len(rtts) if not is_udp else 0,
        'cwnd_sizes': len(cwnd_sizes) if not is_udp else 0,
        'jitters': len(jitters) if is_server else 0,
        'received_bytes': len(received_bytes) if is_server else 0
    }
    max_length = max(series_lengths.values())

    # Truncate or pad each series to match the max length
    def adjust_length(series, target_length, pad_value=None):
        if len(series) > target_length:
            return series[:target_length]
        return series + [pad_value] * (target_length - len(series))

    times = adjust_length(times, max_length)
    throughputs = adjust_length(throughputs, max_length)
    drop_rates = adjust_length(drop_rates, max_length)
    retransmits = adjust_length(retransmits, max_length)
    inter_arrival_times = adjust_length(inter_arrival_times, max_length)
    if not is_udp:
        window_sizes = adjust_length(window_sizes, max_length)
        rtts = adjust_length(rtts, max_length)
        cwnd_sizes = adjust_length(cwnd_sizes, max_length)
    if is_server:
        jitters = adjust_length(jitters, max_length)
        received_bytes = adjust_length(received_bytes, max_length)

    # Create DataFrame dictionary with proper types
    df_dict = {
        'experiment_time': pl.Series(times, dtype=pl.Float64),
        'throughput_mbps': pl.Series(throughputs, dtype=pl.Float64),
        'drop_rate_pct': pl.Series(drop_rates, dtype=pl.Float64),
        'retransmits': pl.Series(retransmits, dtype=pl.Int64),
        'inter_arrival_time': pl.Series(inter_arrival_times, dtype=pl.Float64)
    }

    # Add protocol specific columns
    if not is_udp:
        df_dict.update({
            'window_size': pl.Series(window_sizes, dtype=pl.Float64),
            'rtt': pl.Series(rtts, dtype=pl.Float64),
            'cwnd': pl.Series(cwnd_sizes, dtype=pl.Float64)
        })
    
    # Add server specific columns
    if is_server:
        df_dict['jitter_ms'] = pl.Series(jitters, dtype=pl.Float64)
        df_dict['received_bytes'] = pl.Series(received_bytes, dtype=pl.Int64)
    
    # Return the DataFrame
    return pl.DataFrame(df_dict)

def align_dataframes(df_list, time_column='experiment_time'):
    """Standardize experiment times across iterations to allow averaging"""
    if not df_list:
        return []
    
    # Find the full time range across all dataframes
    all_times = set()
    for df in df_list:
        if not df.is_empty():
            all_times.update(df[time_column].to_list())
    
    if not all_times:
        return []
        
    # Sort times to ensure consistent ordering
    all_times = sorted(all_times)
    
    # Reindex each dataframe to include all time points
    aligned_dfs = []
    for df in df_list:
        if df.is_empty():
            # Create an empty dataframe with the standard time index
            aligned_df = pl.DataFrame({
                time_column: all_times,
                **{col: [None] * len(all_times) for col in df.columns if col != time_column}
            })
        else:
            # Create a dictionary to map times to row values
            df_dict = {t: row for t, row in zip(df[time_column], df.rows())}
            
            # Create new rows with the standard time index
            new_rows = []
            for t in all_times:
                if t in df_dict:
                    new_rows.append(df_dict[t])
                else:
                    new_row = [t] + [None] * (len(df.columns) - 1)
                    new_rows.append(new_row)
            
            # Create a new dataframe with aligned time indices
            aligned_df = pl.DataFrame(
                new_rows,
                schema=[time_column] + [col for col in df.columns if col != time_column]
            )
        
        aligned_dfs.append(aligned_df)
    
    return aligned_dfs

def calculate_fairness_metrics(tcp_throughput, udp_throughput):
    """Calculate Jain's Fairness Index and bandwidth share"""
    # Align lengths of tcp_throughput and udp_throughput
    min_length = min(len(tcp_throughput), len(udp_throughput))
    tcp_throughput = tcp_throughput[:min_length]
    udp_throughput = udp_throughput[:min_length]

    total_throughput = tcp_throughput + udp_throughput
    tcp_share = tcp_throughput / total_throughput if total_throughput.any() else 0
    udp_share = udp_throughput / total_throughput if total_throughput.any() else 0
    
    # Jain's Fairness Index
    sum_squares = tcp_throughput**2 + udp_throughput**2
    square_sums = (tcp_throughput + udp_throughput)**2
    jain_index = square_sums / (2 * sum_squares) if sum_squares.any() else 0
    
    return tcp_share, udp_share, jain_index

def calculate_statistics(df_list, value_columns):
    """Calculate mean and standard deviation for each time point across iterations"""
    if not df_list or all(df.is_empty() for df in df_list):
        return None
    
    aligned_dfs = align_dataframes(df_list)
    if not aligned_dfs:
        return None
    
    # Extract time points from the first dataframe
    time_points = aligned_dfs[0]['experiment_time'].to_list() if not aligned_dfs[0].is_empty() else []
    
    # Calculate statistics for each value column
    stats = {'experiment_time': time_points}
    
    for col in value_columns:
        means = []
        stds = []
        for t_idx, t in enumerate(time_points):
            values = [df[col][t_idx] for df in aligned_dfs if not df.is_empty() and t_idx < len(df)]
            values = [v for v in values if v is not None and not np.isnan(v)]
            
            if values:
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        
        stats[f'{col}_mean'] = means
        stats[f'{col}_std'] = stds
    
    return pl.DataFrame(stats)

def collect_iteration_data(base_path):
    """Collect data from all iterations in the base path"""
    iteration_dirs = sorted(glob.glob(os.path.join(base_path, 'iteration_*')))
    
    if not iteration_dirs:
        print(f"No iteration directories found in {base_path}")
        # If no iteration directories found, treat the base path as a single iteration
        # Check if the required files exist directly in the base path
        if all(os.path.exists(os.path.join(base_path, f)) for f in ['ping_result.log', 'iperf3_tcp.json', 'iperf3_udp.json']):
            print(f"Found data files directly in {base_path}, treating as single iteration")
            ping_df = parse_ping_log(os.path.join(base_path, 'ping_result.log'))
            tcp_df = parse_iperf_json(os.path.join(base_path, 'iperf3_tcp.json'), offset=3, is_udp=False)
            udp_df = parse_iperf_json(os.path.join(base_path, 'iperf3_udp.json'), offset=8, is_udp=True)
            
            # Add server side data if available
            tcp_server_df = parse_iperf_json(os.path.join(base_path, 'iperf3_tcp_server.log'), offset=3, is_udp=False, is_server=True)
            udp_server_df = parse_iperf_json(os.path.join(base_path, 'iperf3_udp_server.log'), offset=8, is_udp=True, is_server=True)
            
            return [ping_df], [tcp_df], [udp_df], [tcp_server_df], [udp_server_df], 1
        else:
            print(f"No data files found in {base_path}")
            return [], [], [], [], [], 0
    
    ping_dfs = []
    tcp_dfs = []
    udp_dfs = []
    tcp_server_dfs = []
    udp_server_dfs = []
    
    for iteration_dir in iteration_dirs:
        ping_log = os.path.join(iteration_dir, 'ping_result.log')
        tcp_json = os.path.join(iteration_dir, 'iperf3_tcp.json')
        udp_json = os.path.join(iteration_dir, 'iperf3_udp.json')
        tcp_server_json = os.path.join(iteration_dir, 'iperf3_tcp_server.log')
        udp_server_json = os.path.join(iteration_dir, 'iperf3_udp_server.log')
        
        ping_df = parse_ping_log(ping_log)
        tcp_df = parse_iperf_json(tcp_json, offset=3, is_udp=False)
        udp_df = parse_iperf_json(udp_json, offset=8, is_udp=True)
        tcp_server_df = parse_iperf_json(tcp_server_json, offset=3, is_udp=False, is_server=True)
        udp_server_df = parse_iperf_json(udp_server_json, offset=8, is_udp=True, is_server=True)
        
        ping_dfs.append(ping_df)
        tcp_dfs.append(tcp_df)
        udp_dfs.append(udp_df)
        tcp_server_dfs.append(tcp_server_df)
        udp_server_dfs.append(udp_server_df)
    
    return ping_dfs, tcp_dfs, udp_dfs, tcp_server_dfs, udp_server_dfs, len(iteration_dirs)

def visualize_iterations(base_path, name='network_metrics_iterations'):
    print(f"Collecting data from iterations in: {base_path}")
    ping_dfs, tcp_dfs, udp_dfs, tcp_server_dfs, udp_server_dfs, num_iterations = collect_iteration_data(base_path)
    
    if num_iterations == 0:
        print("No data found to visualize.")
        return
    
    print(f"Found {num_iterations} iterations")
    
    # Calculate statistics across iterations
    ping_stats = calculate_statistics(ping_dfs, ['latency_ms'])
    tcp_stats = calculate_statistics(tcp_dfs, ['throughput_mbps', 'drop_rate_pct', 'retransmits'])
    udp_stats = calculate_statistics(udp_dfs, ['throughput_mbps', 'drop_rate_pct'])
    
    # Calculate server statistics if available
    tcp_server_stats = calculate_statistics(tcp_server_dfs, ['throughput_mbps', 'drop_rate_pct', 'retransmits', 'received_bytes'])
    udp_server_stats = calculate_statistics(udp_server_dfs, ['throughput_mbps', 'drop_rate_pct', 'jitter_ms', 'received_bytes'])
    
    has_server_data = tcp_server_stats is not None and not tcp_server_stats.is_empty() or \
                     udp_server_stats is not None and not udp_server_stats.is_empty()
    
    # Update plot height based on available data
    n_plots = 6 if has_server_data else 4
    plt.figure(figsize=(12, 3 * n_plots))
    
    # 1) Latency
    plt.subplot(n_plots, 1, 1)
    if ping_stats is not None and not ping_stats.is_empty():
        plt.plot(ping_stats['experiment_time'], ping_stats['latency_ms_mean'], 
                 label='Mean Ping Latency', color='blue')
        plt.fill_between(ping_stats['experiment_time'], 
                         np.array(ping_stats['latency_ms_mean']) - np.array(ping_stats['latency_ms_std']),
                         np.array(ping_stats['latency_ms_mean']) + np.array(ping_stats['latency_ms_std']),
                         alpha=0.3, color='blue')
    plt.ylabel('Latency (ms)')
    plt.title(f'Network Metrics Over {num_iterations} Iterations')
    plt.grid(True)
    plt.legend()
    
    # 2) Throughput (Client)
    plt.subplot(n_plots, 1, 2)
    if tcp_stats is not None and not tcp_stats.is_empty():
        plt.plot(tcp_stats['experiment_time'], tcp_stats['throughput_mbps_mean'], 
                 label='Mean TCP Throughput (Client)', color='green')
        plt.fill_between(tcp_stats['experiment_time'], 
                         np.array(tcp_stats['throughput_mbps_mean']) - np.array(tcp_stats['throughput_mbps_std']),
                         np.array(tcp_stats['throughput_mbps_mean']) + np.array(tcp_stats['throughput_mbps_std']),
                         alpha=0.3, color='green')
    
    if udp_stats is not None and not udp_stats.is_empty():
        plt.plot(udp_stats['experiment_time'], udp_stats['throughput_mbps_mean'], 
                 label='Mean UDP Throughput (Client)', color='orange')
        plt.fill_between(udp_stats['experiment_time'], 
                         np.array(udp_stats['throughput_mbps_mean']) - np.array(udp_stats['throughput_mbps_std']),
                         np.array(udp_stats['throughput_mbps_mean']) + np.array(udp_stats['throughput_mbps_std']),
                         alpha=0.3, color='orange')
    plt.ylabel('Throughput (Mbps)')
    plt.grid(True)
    plt.legend()
    
    # 3) TCP Retransmissions + Drop-Rate (Client)
    plt.subplot(n_plots, 1, 3)
    if tcp_stats is not None and not tcp_stats.is_empty():
        plt.plot(tcp_stats['experiment_time'], tcp_stats['retransmits_mean'], 
                 label='Mean TCP Retransmits (Client)', color='blue')
        plt.fill_between(tcp_stats['experiment_time'], 
                         np.array(tcp_stats['retransmits_mean']) - np.array(tcp_stats['retransmits_std']),
                         np.array(tcp_stats['retransmits_mean']) + np.array(tcp_stats['retransmits_std']),
                         alpha=0.3, color='blue')
        
        plt_twin = plt.twinx()
        plt_twin.plot(tcp_stats['experiment_time'], tcp_stats['drop_rate_pct_mean'], 
                      label='Mean TCP Drop Rate % (Client)', color='red', linestyle='--')
        plt_twin.fill_between(tcp_stats['experiment_time'], 
                             np.array(tcp_stats['drop_rate_pct_mean']) - np.array(tcp_stats['drop_rate_pct_std']),
                             np.array(tcp_stats['drop_rate_pct_mean']) + np.array(tcp_stats['drop_rate_pct_std']),
                             alpha=0.2, color='red')
        plt_twin.set_ylabel('Drop Rate (%)', color='red')
        plt_twin.tick_params(axis='y', labelcolor='red')
    plt.ylabel('Retransmits')
    plt.grid(True)
    plt.legend()
    
    # 4) UDP Drop-Rate (Client)
    plt.subplot(n_plots, 1, 4)
    if udp_stats is not None and not udp_stats.is_empty():
        plt.plot(udp_stats['experiment_time'], udp_stats['drop_rate_pct_mean'], 
                 label='Mean UDP Drop Rate % (Client)', color='purple')
        plt.fill_between(udp_stats['experiment_time'], 
                         np.array(udp_stats['drop_rate_pct_mean']) - np.array(udp_stats['drop_rate_pct_std']),
                         np.array(udp_stats['drop_rate_pct_mean']) + np.array(udp_stats['drop_rate_pct_std']),
                         alpha=0.3, color='purple')
    plt.ylabel('Drop Rate (%)')
    plt.grid(True)
    plt.legend()

    # 5) Server Throughput Comparison
    if has_server_data:
        plt.subplot(n_plots, 1, 5)
        if tcp_server_stats is not None and not tcp_server_stats.is_empty():
            plt.plot(tcp_server_stats['experiment_time'], tcp_server_stats['throughput_mbps_mean'], 
                     label='Mean TCP Throughput (Server)', color='darkgreen')
            plt.fill_between(tcp_server_stats['experiment_time'], 
                             np.array(tcp_server_stats['throughput_mbps_mean']) - np.array(tcp_server_stats['throughput_mbps_std']),
                             np.array(tcp_server_stats['throughput_mbps_mean']) + np.array(tcp_server_stats['throughput_mbps_std']),
                             alpha=0.3, color='darkgreen')
        
        if udp_server_stats is not None and not udp_server_stats.is_empty():
            plt.plot(udp_server_stats['experiment_time'], udp_server_stats['throughput_mbps_mean'], 
                     label='Mean UDP Throughput (Server)', color='darkorange')
            plt.fill_between(udp_server_stats['experiment_time'], 
                             np.array(udp_server_stats['throughput_mbps_mean']) - np.array(udp_server_stats['throughput_mbps_std']),
                             np.array(udp_server_stats['throughput_mbps_mean']) + np.array(udp_server_stats['throughput_mbps_std']),
                             alpha=0.3, color='darkorange')
        plt.ylabel('Throughput (Mbps)')
        plt.grid(True)
        plt.legend()

    # 6) Fairness Index
    plt.subplot(n_plots, 1, 6)
    if tcp_stats is not None and not tcp_stats.is_empty() and udp_stats is not None and not udp_stats.is_empty():
        tcp_throughput = np.array(tcp_stats['throughput_mbps_mean'])
        udp_throughput = np.array(udp_stats['throughput_mbps_mean'])
        _, _, jain_index = calculate_fairness_metrics(tcp_throughput, udp_throughput)
        
        # Ensure dimensions match
        experiment_time = tcp_stats['experiment_time']
        if len(experiment_time) != len(jain_index):
            min_length = min(len(experiment_time), len(jain_index))
            experiment_time = experiment_time[:min_length]
            jain_index = jain_index[:min_length]
            print("Warning: Mismatched dimensions for Jain's Fairness Index plot. Data truncated.")

        plt.plot(experiment_time, jain_index, label="Jain's Fairness Index", color='purple')
    plt.ylabel('Fairness Index')
    plt.xlabel('Experiment Time (s)')
    plt.grid(True)
    plt.legend()

    # Save the summary plot
    plt.tight_layout()
    summary_dir = os.path.join(os.path.dirname(base_path), f"{os.path.basename(base_path)}_summary")
    os.makedirs(summary_dir, exist_ok=True)
    output_path = os.path.join(summary_dir, f"{name}.png")
    plt.savefig(output_path)
    print(f"Summary visualization saved in: {output_path}")

    # Generate individual plots for each iteration
    for i, (ping_df, tcp_df, udp_df, tcp_server_df, udp_server_df) in enumerate(
        zip(ping_dfs, tcp_dfs, udp_dfs, tcp_server_dfs, udp_server_dfs), 1):
        
        plt.figure(figsize=(12, 18))
        has_indiv_server_data = not tcp_server_df.is_empty() or not udp_server_df.is_empty()
        n_indiv_plots = 6 if has_indiv_server_data else 4
        
        # 1) Latency
        plt.subplot(n_indiv_plots, 1, 1)
        if not ping_df.is_empty():
            # Plot all latency values (including NaN which will create gaps)
            plt.plot(ping_df['experiment_time'], ping_df['latency_ms'], 
                    label='Ping Latency', color='blue')
            
            # Calculate y-axis limits from valid values only
            valid_latencies = ping_df.filter(~pl.col("latency_ms").is_nan())
            if not valid_latencies.is_empty():
                max_latency = max(valid_latencies['latency_ms'].max() * 1.1, 100)
                plt.ylim(0, max_latency)
            
            # Plot high latency points
            high_latency = ping_df.filter(pl.col("latency_ms") > 2000)
            if not high_latency.is_empty():
                plt.scatter(high_latency["experiment_time"], high_latency["latency_ms"], 
                        color='red', label='High Latency')
            
            # Plot packet loss points at 95% of plot height
            lost_packets = ping_df.filter(pl.col("latency_ms").is_nan())
            if not lost_packets.is_empty():
                ylim = plt.gca().get_ylim()
                plt.scatter(lost_packets['experiment_time'], 
                        [ylim[1] * 0.95] * len(lost_packets), 
                        marker='x', color='red', label='Packet Loss')

            plt.ylabel('Latency (ms)')
            plt.title(f'Network Metrics (Iteration {i})')
            plt.grid(True)
            plt.legend()
        if not ping_df.is_empty():
            plt.legend()

        # 2) Client Throughput
        plt.subplot(n_indiv_plots, 1, 2)
        if not tcp_df.is_empty():
            plt.plot(tcp_df['experiment_time'], tcp_df['throughput_mbps'], label='TCP Throughput (Client)', color='green')
        if not udp_df.is_empty():
            plt.plot(udp_df['experiment_time'], udp_df['throughput_mbps'], label='UDP Throughput (Client)', color='orange')
        plt.ylabel('Throughput (Mbps)')
        plt.grid(True)
        plt.legend()

        # 3) TCP Retransmissions + Drop-Rate
        plt.subplot(n_indiv_plots, 1, 3)
        if not tcp_df.is_empty():
            plt.plot(tcp_df['experiment_time'], tcp_df['retransmits'], label='TCP Retransmits (Client)', color='blue')
            plt_twin = plt.twinx()
            plt_twin.plot(tcp_df['experiment_time'], tcp_df['drop_rate_pct'], label='TCP Drop Rate % (Client)', color='red', linestyle='--')
            plt_twin.set_ylabel('Drop Rate (%)', color='red')
            plt_twin.tick_params(axis='y', labelcolor='red')
        plt.ylabel('Retransmits')
        plt.grid(True)
        plt.legend()

        # 4) UDP Drop-Rate
        plt.subplot(n_indiv_plots, 1, 4)
        if not udp_df.is_empty():
            plt.plot(udp_df['experiment_time'], udp_df['drop_rate_pct'], label='UDP Drop Rate % (Client)', color='purple')
        plt.ylabel('Drop Rate (%)')
        plt.grid(True)
        plt.legend()

        # 5) Server Throughput Comparison (if available)
        if has_indiv_server_data:
            plt.subplot(n_indiv_plots, 1, 5)
            if not tcp_server_df.is_empty():
                plt.plot(tcp_server_df['experiment_time'], tcp_server_df['throughput_mbps'], label='TCP Throughput (Server)', color='darkgreen')
            if not udp_server_df.is_empty():
                plt.plot(udp_server_df['experiment_time'], udp_server_df['throughput_mbps'], label='UDP Throughput (Server)', color='darkorange')
            plt.ylabel('Throughput (Mbps)')
            plt.grid(True)
            plt.legend()

        # 6) Fairness Index (if both TCP and UDP data are available)
        if not tcp_df.is_empty() and not udp_df.is_empty():
            plt.subplot(n_indiv_plots, 1, 6)
            tcp_throughput = np.array(tcp_df['throughput_mbps'])
            udp_throughput = np.array(udp_df['throughput_mbps'])
            _, _, jain_index = calculate_fairness_metrics(tcp_throughput, udp_throughput)
            experiment_time = tcp_df['experiment_time'][:len(jain_index)]
            plt.plot(experiment_time, jain_index, label="Jain's Fairness Index", color='purple')
            plt.ylabel('Fairness Index')
            plt.xlabel('Experiment Time (s)')
            plt.grid(True)
            plt.legend()

        # Save individual iteration plot
        iteration_output_path = os.path.join(summary_dir, f"{name}_iteration_{i}.png")
        plt.tight_layout()
        plt.savefig(iteration_output_path)
        plt.close()
        print(f"Iteration {i} visualization saved in: {iteration_output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize network metrics from experiment data.')
    parser.add_argument('--path', help='Path to the experiment directory')
    parser.add_argument('--name', default='network_metrics', help='Base name for output files')
    
    args = parser.parse_args()
    
    visualize_iterations(args.path, args.name)

if __name__ == "__main__":
    main()