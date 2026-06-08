# WAM Dataset Builder
# Conda Setup Guide (WAM Dataset Builder)

## Create Environment

```bash
conda create -n wam_dataset python=3.10 -y
conda activate wam_dataset
conda install numpy scipy matplotlib scikit-learn -y
```
---

| Single Trajectory: Alignment & Segmentation | Global Dataset: Statistics & Clustering |
| :---: | :---: |
| <img src="media/readme/baseline_rigid.gif" width="280"/> | <img src="media/readme/ours_compliant.gif" width="280"/> |
| *Spatiotemporal alignment of video and action features (Steps 1 & 2) alongside atomic event segmentation based on kinematic valleys and gripper transitions (Step 3).* | *Global dataset insights showcasing event duration distribution, a 2D cluster bucket heatmap for balanced sampling, and PCA dimensionality reduction of high-dimensional features.* |

## Overview

This project demonstrates a complete pipeline for converting raw robot demonstrations into structured event-level data suitable for World Action Models (WAM), skill learning, and robotics foundation models.

Pipeline:

```text
Raw Demonstrations
        ↓
Lag Estimation
        ↓
Time Alignment
        ↓
Event Segmentation
        ↓
Feature Extraction
        ↓
Clustering
        ↓
Balanced Sampling
        ↓
Dataset Export
```

---

## Module 1: Mock Data Generation

**Function:** `generate_mock_episodes()`

Generates synthetic robot demonstrations for testing.

Each episode contains:

- `video_motion` : visual motion signal
- `eef_pose` : end-effector trajectory
- `gripper` : gripper state
- `task_caption` : task description
- `vl_embedding` : vision-language embedding

---

## Module 2: Episode Processing

**Class:** `EpisodeProcessor`

Responsible for synchronization, alignment, segmentation, and feature extraction.

### Step 1: Lag Estimation

**Function:** `step1_estimate_lag()`

Computes the temporal offset between video observations and robot actions using cross-correlation between:

- video motion magnitude
- end-effector velocity

Output:

```python
estimated_lag
```

---

### Step 2: Time Alignment

**Function:** `step2_align_and_trim()`

Applies the estimated lag and resamples all modalities onto a common timeline.

Outputs:

```python
aligned_pose
aligned_gripper
aligned_vl_embed
```

---

### Step 3: Event Segmentation

**Function:** `step3_and_4_segmentation()`

Detects event boundaries using:

1. Gripper state transitions
2. Velocity valleys

Each segment becomes an atomic robot event.

Extracted features:

```python
action_delta
action_energy
duration_sec
```

---

## Module 3: Event Clustering

**Function:** `step5_cluster_events()`

Performs K-Means clustering on event features.

### Vision-Language Clustering

Input:

```python
vl_embedding
```

Output:

```python
vl_cluster
```

Represents semantic similarity.

### Action Clustering

Input:

```python
[dx, dy, dz, energy]
```

Output:

```python
action_cluster
```

Represents motion similarity.

Each event receives a bucket label:

```python
(vl_cluster, action_cluster)
```

---

## Module 4: Balanced Event Sampling

**Class:** `BalancedEventSampler`

Creates balanced mini-batches by sampling uniformly across cluster buckets.

Advantages:

- reduces dataset imbalance
- increases behavior diversity
- improves training stability

Sampling strategy:

```text
Random Bucket
      ↓
Random Event
```

---

## Module 5: Visualization Dashboard

**Function:** `plot_dataset_dashboard()`

Provides four diagnostic views:

1. Event duration distribution
2. Cluster heatmap
3. Vision-language PCA projection
4. Action-feature PCA projection

Used for validating segmentation quality and dataset diversity.

---

## Output

The final dataset is exported as:

```text
events.jsonl
```

Example:

```json
{
  "event_id": "mock_ep_001_ev_2",
  "duration_sec": 1.2,
  "action_delta": [0.15, -0.08, 0.00],
  "action_energy": 0.45,
  "vl_cluster": 1,
  "action_cluster": 2
}
```

---

## Relation to WAM

This pipeline represents a typical preprocessing stage for World Action Models:

```text
Raw Demonstrations
        ↓
Temporal Alignment
        ↓
Event Segmentation
        ↓
Feature Extraction
        ↓
Action Events
        ↓
WAM Training
```

The generated events can be treated as reusable action primitives for:

- World Action Models (WAM)
- Skill Libraries
- Behavior Retrieval
- Hierarchical Reinforcement Learning
- Diffusion Policies
- Robotics Foundation Models
