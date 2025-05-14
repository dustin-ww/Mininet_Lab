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

    times, throughputs, drop_rates, retransmits = [], [], [], []
    jitters, received_bytes = [], []

    for interval in data.get('intervals', []):
        sum_data = interval.get('sum', {})
        duration = sum_data.get('end', 0) - sum_data.get('start', 0)
        if duration <= 0:
            continue
        t = float(offset + sum_data['end'])
        tp = float((sum_data.get('bytes', 0) * 8) / (duration * 1e6))
        times.append(t)
        throughputs.append(tp)
        rtx = int(sum_data.get('retransmits', 0)) if not is_udp else 0
        retransmits.append(rtx)
        
        # Add server-specific data points
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

        if is_udp:
            lost, pkts = 0, 0
            for st in interval.get('streams', []):
                udp = st.get('udp', {})
                lost += int(udp.get('lost_packets', 0))
                received = int(udp.get('packets', 0))
                pkts += (lost + received)
            drop_pct = float(lost / pkts * 100) if pkts > 0 else 0.0
        else:
            packets = int(sum_data.get('bytes', 0) / MSS_BYTES) if sum_data.get('bytes', 0) > 0 else 0
            drop_pct = float(rtx / packets * 100) if packets > 0 else 0.0
        drop_rates.append(drop_pct)

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

    df_dict = {
        'experiment_time': pl.Series(times, dtype=pl.Float64),
        'throughput_mbps': pl.Series(throughputs, dtype=pl.Float64),
        'drop_rate_pct': pl.Series(drop_rates, dtype=pl.Float64),
        'retransmits': pl.Series(retransmits, dtype=pl.Int64)
    }
    
    # Add server-specific columns if this is server data
    if is_server:
        df_dict['jitter_ms'] = pl.Series(jitters, dtype=pl.Float64)
        df_dict['received_bytes'] = pl.Series(received_bytes, dtype=pl.Int64)
        
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
    
    # Additional server plots if data is available
    if has_server_data:
        # 5) Server Throughput Comparison
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
        
        # 6) UDP Jitter (Server) and Received Bytes Comparison
        plt.subplot(n_plots, 1, 6)
        
        if udp_server_stats is not None and not udp_server_stats.is_empty() and 'jitter_ms_mean' in udp_server_stats.columns:
            plt.plot(udp_server_stats['experiment_time'], udp_server_stats['jitter_ms_mean'], 
                    label='Mean UDP Jitter (Server)', color='blue')
            plt.fill_between(udp_server_stats['experiment_time'], 
                            np.array(udp_server_stats['jitter_ms_mean']) - np.array(udp_server_stats['jitter_ms_std']),
                            np.array(udp_server_stats['jitter_ms_mean']) + np.array(udp_server_stats['jitter_ms_std']),
                            alpha=0.3, color='blue')
            plt.ylabel('Jitter (ms)')
            
            # Add received bytes comparison if available (right axis)
            if 'received_bytes_mean' in udp_server_stats.columns and 'received_bytes_mean' in tcp_server_stats.columns:
                plt_twin = plt.twinx()
                plt_twin.plot(tcp_server_stats['experiment_time'], 
                             np.array(tcp_server_stats['received_bytes_mean']) / 1024, 
                             label='TCP Received (KB)', color='green', linestyle='-.')
                plt_twin.plot(udp_server_stats['experiment_time'], 
                             np.array(udp_server_stats['received_bytes_mean']) / 1024, 
                             label='UDP Received (KB)', color='orange', linestyle='-.')
                plt_twin.set_ylabel('Received (KB)', color='black')
                plt_twin.legend(loc='upper right')
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.xlabel('Experiment Time (s)')
    else:
        # If no server data, add x-axis label to the 4th plot
        plt.subplot(n_plots, 1, 4)
        plt.xlabel('Experiment Time (s)')
    
    plt.tight_layout()
    
    # Create summary directory if it doesn't exist
    parent_dir = os.path.dirname(base_path)
    experiment_name = os.path.basename(base_path)
    summary_dir = os.path.join(parent_dir, f"{experiment_name}_summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(summary_dir, f"{name}.png")
    plt.savefig(output_path)
    print(f"Summary visualization saved in: {output_path}")
    
    # Save statistics to CSV
    if ping_stats is not None and not ping_stats.is_empty():
        ping_stats.write_csv(os.path.join(summary_dir, "ping_statistics.csv"))
    if tcp_stats is not None and not tcp_stats.is_empty():
        tcp_stats.write_csv(os.path.join(summary_dir, "tcp_statistics.csv"))
    if udp_stats is not None and not udp_stats.is_empty():
        udp_stats.write_csv(os.path.join(summary_dir, "udp_statistics.csv"))
    
    # Save server statistics if available
    if tcp_server_stats is not None and not tcp_server_stats.is_empty():
        tcp_server_stats.write_csv(os.path.join(summary_dir, "tcp_server_statistics.csv"))
    if udp_server_stats is not None and not udp_server_stats.is_empty():
        udp_server_stats.write_csv(os.path.join(summary_dir, "udp_server_statistics.csv"))
    
    # Also create individual iteration plots
    if num_iterations > 1:
        for i, (ping_df, tcp_df, udp_df, tcp_server_df, udp_server_df) in enumerate(
            zip(ping_dfs, tcp_dfs, udp_dfs, tcp_server_dfs, udp_server_dfs), 1):
            
            has_indiv_server_data = not tcp_server_df.is_empty() or not udp_server_df.is_empty()
            n_indiv_plots = 6 if has_indiv_server_data else 4
            
            plt.figure(figsize=(12, 3 * n_indiv_plots))
            
            # 1) Latency
            plt.subplot(n_indiv_plots, 1, 1)
            if not ping_df.is_empty():
                valid_pings = ping_df.filter(~pl.col("latency_ms").is_nan())
                plt.plot(valid_pings['experiment_time'], valid_pings['latency_ms'], label=f'Ping Latency', color='blue')
                high_latency = ping_df.filter(pl.col("latency_ms") > 2000)
                if not high_latency.is_empty():
                    plt.scatter(high_latency["experiment_time"], high_latency["latency_ms"], color='red', label='High Latency')
                lost_packets = ping_df.filter(pl.col("latency_ms").is_nan())
                if not lost_packets.is_empty():
                    plt.scatter(lost_packets['experiment_time'], [plt.gca().get_ylim()[1] * 0.95] * len(lost_packets), marker='x', color='red', label='Packet Loss')
            plt.ylabel('Latency (ms)')
            plt.title(f'Network Metrics (Iteration {i})')
            plt.grid(True)
            plt.legend()
            
            # 2) Client Throughput
            plt.subplot(n_indiv_plots, 1, 2)
            if not tcp_df.is_empty():
                plt.plot(tcp_df['experiment_time'], tcp_df['throughput_mbps'], label='TCP Throughput (Client)')
            if not udp_df.is_empty():
                plt.plot(udp_df['experiment_time'], udp_df['throughput_mbps'], label='UDP Throughput (Client)')
            plt.ylabel('Throughput (Mbps)')
            plt.grid(True)
            plt.legend()
            
            # 3) TCP Client Retransmissions + Drop-Rate
            plt.subplot(n_indiv_plots, 1, 3)
            if not tcp_df.is_empty():
                plt.plot(tcp_df['experiment_time'], tcp_df['retransmits'], label='TCP Retransmits (Client)')
                plt_twin = plt.twinx()
                plt_twin.plot(tcp_df['experiment_time'], tcp_df['drop_rate_pct'], label='TCP Drop Rate % (Client)', color='red', linestyle='--')
                plt_twin.set_ylabel('Drop Rate (%)', color='red')
                plt_twin.tick_params(axis='y', labelcolor='red')
            plt.ylabel('Retransmits')
            plt.grid(True)
            plt.legend()
            
            # 4) UDP Client Drop-Rate
            plt.subplot(n_indiv_plots, 1, 4)
            if not udp_df.is_empty():
                plt.plot(udp_df['experiment_time'], udp_df['drop_rate_pct'], label='UDP Drop Rate % (Client)')
            plt.ylabel('Drop Rate (%)')
            plt.grid(True)
            plt.legend()
            
            # Add server plots if data is available
            if has_indiv_server_data:
                # 5) Server Throughput Comparison
                plt.subplot(n_indiv_plots, 1, 5)
                if not tcp_server_df.is_empty():
                    plt.plot(tcp_server_df['experiment_time'], tcp_server_df['throughput_mbps'], 
                            label='TCP Throughput (Server)', color='darkgreen')
                
                if not udp_server_df.is_empty():
                    plt.plot(udp_server_df['experiment_time'], udp_server_df['throughput_mbps'], 
                            label='UDP Throughput (Server)', color='darkorange')
                plt.ylabel('Throughput (Mbps)')
                plt.grid(True)
                plt.legend()
                
                # 6) Server UDP Jitter and Received Bytes
                               # 6) Server UDP Jitter and Received Bytes
                plt.subplot(n_indiv_plots, 1, 6)
                
                if not udp_server_df.is_empty() and 'jitter_ms' in udp_server_df.columns:
                    plt.plot(udp_server_df['experiment_time'], udp_server_df['jitter_ms'], 
                            label='UDP Jitter (Server)', color='blue')
                    plt.ylabel('Jitter (ms)')
                    
                    # Add received bytes comparison on right axis
                    if 'received_bytes' in udp_server_df.columns and 'received_bytes' in tcp_server_df.columns:
                        plt_twin = plt.twinx()
                        plt_twin.plot(tcp_server_df['experiment_time'], 
                                    np.array(tcp_server_df['received_bytes']) / 1024, 
                                    label='TCP Received (KB)', color='green', linestyle='-.')
                        plt_twin.plot(udp_server_df['experiment_time'], 
                                    np.array(udp_server_df['received_bytes']) / 1024, 
                                    label='UDP Received (KB)', color='orange', linestyle='-.')
                        plt_twin.set_ylabel('Received (KB)', color='black')
                        plt_twin.legend(loc='upper right')
                plt.grid(True)
                plt.legend(loc='upper left')
                plt.xlabel('Experiment Time (s)')
            else:
                # If no server data, add x-axis label to the 4th plot
                plt.subplot(n_indiv_plots, 1, 4)
                plt.xlabel('Experiment Time (s)')
            
            plt.tight_layout()
            
            # Save individual iteration plot
            output_path = os.path.join(summary_dir, f"{name}_iteration_{i}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Iteration {i} visualization saved in: {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize network metrics from experiment data.')
    parser.add_argument('--path', help='Path to the experiment directory')
    parser.add_argument('--name', default='network_metrics', help='Base name for output files')
    
    args = parser.parse_args()
    
    visualize_iterations(args.path, args.name)

if __name__ == "__main__":
    main()