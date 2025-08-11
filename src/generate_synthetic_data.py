
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(num_records=1000, start_date="2024-01-01"):
    """Generates synthetic data for predictive maintenance."""
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    timestamps = [start_datetime + timedelta(minutes=i) for i in range(num_records)]

    # Simulate normal operating conditions with some fluctuations
    cpu_load = np.random.normal(loc=0.4, scale=0.1, size=num_records)
    memory_usage = np.random.normal(loc=0.6, scale=0.15, size=num_records)
    disk_activity = np.random.normal(loc=0.3, scale=0.05, size=num_records)

    # Introduce anomalies/pre-failure indicators and failures
    failure_indicator = np.zeros(num_records, dtype=int)
    failure_type = [None] * num_records

    # Simulate a few failure events
    num_failures = num_records // 100 # Approximately 1% failures
    failure_indices = np.random.choice(num_records, num_failures, replace=False)

    for idx in failure_indices:
        # Introduce pre-failure symptoms (e.g., increased load) before actual failure
        pre_failure_window = np.random.randint(5, 20) # 5 to 20 minutes before failure
        if idx - pre_failure_window >= 0:
            for i in range(pre_failure_window):
                # Gradually increase metrics before failure
                cpu_load[idx - pre_failure_window + i] += np.random.uniform(0.1, 0.3)
                memory_usage[idx - pre_failure_window + i] += np.random.uniform(0.1, 0.4)
                disk_activity[idx - pre_failure_window + i] += np.random.uniform(0.05, 0.15)

        # Mark the failure point
        failure_indicator[idx] = 1
        failure_types = ["server_crash", "disk_failure", "service_downtime"]
        failure_type[idx] = np.random.choice(failure_types)

        # Simulate post-failure recovery or continued high load
        post_failure_window = np.random.randint(1, 5) # 1 to 5 minutes after failure
        for i in range(1, post_failure_window + 1):
            if idx + i < num_records:
                cpu_load[idx + i] = np.random.normal(loc=0.8, scale=0.1)
                memory_usage[idx + i] = np.random.normal(loc=0.9, scale=0.1)
                disk_activity[idx + i] = np.random.normal(loc=0.7, scale=0.1)

    # Ensure values are within reasonable bounds (0-1 for percentages, etc.)
    cpu_load = np.clip(cpu_load, 0, 1)
    memory_usage = np.clip(memory_usage, 0, 1)
    disk_activity = np.clip(disk_activity, 0, 1)

    data = pd.DataFrame({
        "timestamp": timestamps,
        "cpu_load": cpu_load,
        "memory_usage": memory_usage,
        "disk_activity": disk_activity,
        "failure_indicator": failure_indicator,
        "failure_type": failure_type
    })

    return data

if __name__ == "__main__":
    synthetic_data = generate_synthetic_data(num_records=5000)
    output_path = "/home/ubuntu/synthetic_predictive_maintenance_data.csv"
    synthetic_data.to_csv(output_path, index=False)
    print(f"Synthetic data generated and saved to {output_path}")
    print(synthetic_data.head())
    print(synthetic_data.tail())
    print(f"Number of failures in data: {synthetic_data[synthetic_data['failure_indicator'] == 1].shape[0]}")


