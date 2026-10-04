# CloudShield — Real-Time Cloud Threat Detection

## Overview & Problem Statement
Automated sub-second threat detection for cloud SOC telemetry. CloudShield provides an interactive, autonomous threat inspection pipeline designed to identify varied adversarial network behavior patterns rapidly and accurately.

## Deep Learning Architecture
Hybrid 1D-CNN + Bidirectional GRU (BiGRU) pipeline with calibrated Softmax ($T=2.2$). The 1D Convolution arrays isolate local temporal anomaly patterns across telemetry channels while the recurrent BiGRU architecture processes sweeping sequential context across the entire packet stream.

## Dataset Description
Emulated CSE-CIC-IDS2018 sequential telemetry:
- **Shape:** `(10, 20)` — 10 sequential time-steps across 20 distinct simulated network features.
- **Classes (4):** 
  - `Benign`: Ambient API traffic with natural network variance.
  - `DDoS Flood`: Traffic volume surge leveraging temporal ramping metrics.
  - `PortScan`: Packet frequency and amplitude dips resembling reconnaissance sweeps.
  - `BruteForce`: Sporadic intermittent spikes explicitly targeting authentication payloads.

## Metrics & Results
- **Accuracy:** >98% overall validation accuracy.
- Encompasses pristine classification boundary isolation visualized via a Normalized Confusion Matrix. The live SOC dashboard additionally features interactive multi-channel temporal waveform analysis.

## Setup & Run Instructions

```bash
# Provide a clean python environment (e.g. Python 3.9+)
pip install -r requirements.txt

# Train the model and generate metrics
python train.py

# Launch the live SOC dashboard
streamlit run app.py
```
