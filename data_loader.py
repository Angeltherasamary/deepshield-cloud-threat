import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_telemetry_dataset(n_samples=6000, timesteps=10, n_features=20):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 0: Benign (40%), 1: DDoS (20%), 2: PortScan (20%), 3: BruteForce (20%)
    y = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.40, 0.20, 0.20, 0.20])
    
    # Ambient baseline network telemetry with realistic variance
    X = np.random.normal(loc=0.0, scale=0.85, size=(n_samples, timesteps, n_features))
    
    for i in range(n_samples):
        # Ambient random network jitter on all telemetry channels
        jitter = np.random.normal(0, 0.25, size=(timesteps, n_features))
        X[i] += jitter
        
        if y[i] == 1:  # DDoS: packet volume surge + temporal ramp
            ramp = np.linspace(0.8, 2.2, timesteps)[:, None] * np.random.uniform(0.7, 1.2)
            X[i, :, :5] += ramp + np.random.normal(0.4, 0.3, size=(timesteps, 5))
            
        elif y[i] == 2:  # PortScan: short bursts in packet frequency & small lengths
            scan_steps = np.random.choice(timesteps, size=np.random.randint(4, 8), replace=False)
            X[i, scan_steps, 5:10] -= np.random.uniform(1.0, 1.8, size=(len(scan_steps), 5))
            
        elif y[i] == 3:  # BruteForce: intermittent spikes on authentication/flag features
            spike_mask = np.random.binomial(1, 0.45, size=(timesteps, 5))
            X[i, :, 15:] += spike_mask * np.random.normal(1.4, 0.5, size=(timesteps, 5))

    return X, y

def export_eda_visualizations(X, y):
    sns.set_theme(style="darkgrid")
    class_names = ['Benign', 'DDoS Flood', 'PortScan', 'BruteForce']
    
    # 1. Class Distribution
    plt.figure(figsize=(7, 4))
    counts = pd.Series(y).value_counts().sort_index()
    sns.barplot(x=[class_names[i] for i in counts.index], y=counts.values, palette="crest")
    plt.title("Telemetry Threat Class Distribution")
    plt.ylabel("Sample Count")
    plt.tight_layout()
    plt.savefig("viz_class_distribution.png", dpi=300)
    plt.close()

    # 2. Feature Correlation Heatmap (Top 10 features aggregated across time)
    plt.figure(figsize=(8, 6))
    X_mean = X.mean(axis=1)[:, :10]
    df_corr = pd.DataFrame(X_mean, columns=[f"Feature_{i+1}" for i in range(10)]).corr()
    sns.heatmap(df_corr, cmap="vlag", annot=True, fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Matrix (Top 10 Telemetry Features)")
    plt.tight_layout()
    plt.savefig("viz_feature_correlation.png", dpi=300)
    plt.close()

    # 3. Temporal Trace Comparison
    plt.figure(figsize=(9, 4))
    for cls_idx, label in enumerate(class_names):
        sample_idx = np.where(y == cls_idx)[0][0]
        plt.plot(range(1, 11), X[sample_idx, :, 0], marker='o', label=label)
    plt.title("Temporal Trace Across 10 Time-Steps (Feature 1)")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("viz_temporal_trace.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating telemetry dataset and EDA visualizations...")
    X, y = generate_telemetry_dataset()
    export_eda_visualizations(X, y)
    print("Generated viz_class_distribution.png, viz_feature_correlation.png, and viz_temporal_trace.png.")