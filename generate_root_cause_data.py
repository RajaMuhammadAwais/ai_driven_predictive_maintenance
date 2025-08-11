
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_root_cause_data(num_records=1000, start_date="2024-01-01"):
    """Generates synthetic data for root cause analysis."""
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    timestamps = [start_datetime + timedelta(minutes=i) for i in range(num_records)]

    # System metrics (features that might indicate a root cause)
    cpu_load = np.random.normal(loc=0.5, scale=0.1, size=num_records)
    memory_usage = np.random.normal(loc=0.5, scale=0.1, size=num_records)
    disk_io = np.random.normal(loc=0.5, scale=0.1, size=num_records)
    network_latency = np.random.normal(loc=0.5, scale=0.1, size=num_records)
    service_response_time = np.random.normal(loc=0.5, scale=0.1, size=num_records)

    # Potential root causes
    root_causes = [
        "software_bug",
        "hardware_failure",
        "network_issue",
        "resource_exhaustion",
        "configuration_error",
        "normal_operation" # For non-failure events
    ]
    failure_types = [
        "server_crash",
        "disk_failure",
        "service_downtime",
        "performance_degradation"
    ]

    # Initialize columns
    failure_indicator = np.zeros(num_records, dtype=int)
    failure_type = [None] * num_records
    root_cause = ["normal_operation"] * num_records

    # Introduce failure events and associated root causes
    num_failures = num_records // 20 # Approximately 5% failures
    failure_indices = np.random.choice(num_records, num_failures, replace=False)

    for idx in failure_indices:
        failure_indicator[idx] = 1
        current_failure_type = np.random.choice(failure_types)
        failure_type[idx] = current_failure_type

        # Assign root cause based on failure type and introduce symptoms
        if current_failure_type == "server_crash":
            root_cause[idx] = np.random.choice(["hardware_failure", "software_bug", "resource_exhaustion"], p=[0.4, 0.3, 0.3])
            if root_cause[idx] == "hardware_failure":
                cpu_load[idx] = np.random.uniform(0.8, 1.0)
                memory_usage[idx] = np.random.uniform(0.8, 1.0)
            elif root_cause[idx] == "software_bug":
                cpu_load[idx] = np.random.uniform(0.7, 0.9)
                memory_usage[idx] = np.random.uniform(0.7, 0.9)
            elif root_cause[idx] == "resource_exhaustion":
                cpu_load[idx] = np.random.uniform(0.9, 1.0)
                memory_usage[idx] = np.random.uniform(0.9, 1.0)

        elif current_failure_type == "disk_failure":
            root_cause[idx] = np.random.choice(["hardware_failure", "resource_exhaustion"])
            disk_io[idx] = np.random.uniform(0.8, 1.0)

        elif current_failure_type == "service_downtime":
            root_cause[idx] = np.random.choice(["network_issue", "configuration_error", "software_bug"])
            if root_cause[idx] == "network_issue":
                network_latency[idx] = np.random.uniform(0.8, 1.0)
            elif root_cause[idx] == "configuration_error":
                service_response_time[idx] = np.random.uniform(0.8, 1.0)

        elif current_failure_type == "performance_degradation":
            root_cause[idx] = np.random.choice(["resource_exhaustion", "software_bug", "network_issue"])
            cpu_load[idx] = np.random.uniform(0.6, 0.8)
            memory_usage[idx] = np.random.uniform(0.6, 0.8)
            service_response_time[idx] = np.random.uniform(0.7, 0.9)

    # Ensure values are within reasonable bounds (0-1 for percentages, etc.)
    cpu_load = np.clip(cpu_load, 0, 1)
    memory_usage = np.clip(memory_usage, 0, 1)
    disk_io = np.clip(disk_io, 0, 1)
    network_latency = np.clip(network_latency, 0, 1)
    service_response_time = np.clip(service_response_time, 0, 1)

    data = pd.DataFrame({
        "timestamp": timestamps,
        "cpu_load": cpu_load,
        "memory_usage": memory_usage,
        "disk_io": disk_io,
        "network_latency": network_latency,
        "service_response_time": service_response_time,
        "failure_indicator": failure_indicator,
        "failure_type": failure_type,
        "root_cause": root_cause
    })

    return data

if __name__ == "__main__":
    synthetic_data = generate_root_cause_data(num_records=5000)
    output_path = "/home/ubuntu/synthetic_root_cause_data.csv"
    synthetic_data.to_csv(output_path, index=False)
    print(f"Synthetic root cause data generated and saved to {output_path}")
    print(synthetic_data.head())
    print(synthetic_data.tail())
    print(f"Number of failures in data: {synthetic_data[synthetic_data['failure_indicator'] == 1].shape[0]}")
    print("Root cause distribution:")
    print(synthetic_data["root_cause"].value_counts())


