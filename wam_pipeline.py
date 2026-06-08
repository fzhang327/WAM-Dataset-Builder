import os
import json
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import medfilt, find_peaks
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Suppress Windows MKL multithreading warning
os.environ["OMP_NUM_THREADS"] = "1"


# ==========================================
# 1. Dummy Data Generator
# ==========================================
def generate_mock_episodes(num_episodes=15):
    """
    Mathematical Principle:
    This function uses a Gaussian Distribution to generate a simulated velocity profile.
    Gaussian function: f(t) = exp(- (t - \mu)^2 / (2 * \sigma^2))
    It overlays two Gaussian peaks to simulate two primary robotic movements
    (e.g., reaching out to grasp -> retracting to place).
    """
    episodes = []
    fps_video, fps_action = 20, 50
    duration_sec = 10.0

    T_video, T_action = int(duration_sec * fps_video), int(duration_sec * fps_action)
    t_video = np.linspace(0, duration_sec, T_video)
    t_action = np.linspace(0, duration_sec, T_action)

    for i in range(num_episodes):
        peak1, peak2 = np.random.normal(3.0, 0.5), np.random.normal(7.0, 0.5)

        def true_velocity(t):
            return np.exp(-((t - peak1) ** 2) / 0.5) + np.exp(-((t - peak2) ** 2) / 0.5)

        video_motion = np.clip(true_velocity(t_video) + np.random.normal(0, 0.05, T_video), 0, None)
        artificial_lag = np.random.uniform(0.05, 0.25)
        shifted_t_action = t_action - artificial_lag
        velocity_profile = true_velocity(shifted_t_action)

        eef_pose = np.zeros((T_action, 7))
        dir_x, dir_y = np.random.choice([-1, 1]), np.random.choice([-1, 1])

        # Mathematical Principle: Displacement is the integral of velocity over time.
        # In discrete time, we use Cumulative Sum: S(T) = sum_{t=0}^{T} v(t) * dt
        eef_pose[:, 0] = np.cumsum(velocity_profile) * (dir_x / fps_action)
        eef_pose[:, 1] = np.cumsum(velocity_profile) * (dir_y / fps_action)
        eef_pose[:, 3:7] = [0, 0, 0, 1]
        eef_pose += np.random.normal(0, 0.001, (T_action, 7))

        gripper = np.ones(T_action)
        grasp_idx = np.searchsorted(shifted_t_action, peak1 + 1.0)
        release_idx = np.searchsorted(shifted_t_action, peak2 + 1.0)
        gripper[grasp_idx:release_idx] = 0.0

        base_embed = np.ones(128) * (1 if i % 2 == 0 else -1)
        vl_embedding = base_embed + np.random.normal(0, 0.5, (T_video, 128))

        episodes.append({
            "episode_id": f"mock_ep_{i:03d}", "fps": fps_video,
            "video_motion": video_motion.astype(np.float32),
            "action_timestamps": t_action.astype(np.float64),
            "eef_pose": eef_pose.astype(np.float32),
            "gripper": gripper.astype(np.float32),
            "task_caption": f"pick and place object {i}",
            "vl_embedding": vl_embedding.astype(np.float32),
            "_ground_truth_lag": artificial_lag
        })
    return episodes


# ==========================================
# 2. Core Data Processor (Alignment & Segmentation)
# ==========================================
class EpisodeProcessor:
    def __init__(self, episode_data):
        self.ep = episode_data
        self.dt_video = 1.0 / self.ep['fps']
        self.t_video = np.arange(len(self.ep['video_motion'])) * self.dt_video
        self.t_action = self.ep['action_timestamps']
        self.quality_flags = []

    def process_all(self, show_debug_plot=False):
        print(f"\n--- Processing trajectory: {self.ep['episode_id']} ---")
        self.step1_estimate_lag()
        self.step2_align_and_trim()
        events = self.step3_and_4_segmentation()

        if show_debug_plot:
            self.plot_debug_alignment()

        return events

    def step1_estimate_lag(self):
        """
        [Math & Algorithms - Cross-Correlation for Delay Estimation]
        1. L2 Norm (Euclidean Distance):
           Calculate the instantaneous speed of the robotic arm using 3D coordinate changes.
           v = sqrt(dx^2 + dy^2 + dz^2) / dt

        2. Z-Score Normalization:
           Standardize both signals (video motion and physical speed) to eliminate scale differences.
           Z = (X - \mu) / \sigma  (Subtract mean, divide by standard deviation)

        3. Discrete Cross-Correlation:
           Measures the similarity of two series as a function of the displacement of one relative to the other.
           (f * g)[n] = \sum f[m] * g[m + n]
           The lag 'n' where this function peaks is the estimated time delay between the two signals.
        """
        eef_pos = self.ep['eef_pose'][:, :3]
        dt_action = np.diff(self.t_action, prepend=self.t_action[0] - (self.t_action[1] - self.t_action[0]))
        dt_action[dt_action <= 0] = 1e-6

        # Calculate speed using L2 Norm
        self.raw_eef_speed = np.linalg.norm(np.diff(eef_pos, axis=0, prepend=eef_pos[0:1]), axis=1) / dt_action

        # 1D Linear Interpolation: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
        interp_func = interp1d(self.t_action, self.raw_eef_speed, bounds_error=False, fill_value=0.0)
        eef_speed_20hz = interp_func(self.t_video)

        # Z-Score Normalization
        v_norm = (self.ep['video_motion'] - np.mean(self.ep['video_motion'])) / (np.std(self.ep['video_motion']) + 1e-8)
        a_norm = (eef_speed_20hz - np.mean(eef_speed_20hz)) / (np.std(eef_speed_20hz) + 1e-8)

        # Cross-correlation to find the temporal peak
        correlation = signal.correlate(v_norm, a_norm, mode='full')
        lags = signal.correlation_lags(len(v_norm), len(a_norm), mode='full')
        self.estimated_lag = lags[np.argmax(correlation)] * self.dt_video

        print(
            f"  -> [Step 1] Cross-correlation lag: {self.estimated_lag:.3f} sec (GT: {self.ep['_ground_truth_lag']:.3f} sec)")

    def step2_align_and_trim(self):
        """
        [Math & Algorithms - Coordinate Transformation & Interpolation]
        t_action_new = t_action_old + estimated_lag
        Uses interpolation functions to resample misaligned or irregularly spaced time series
        onto a uniform grid (20Hz to match the video).
        """
        corrected_t_action = self.t_action + self.estimated_lag
        valid_mask = (self.t_video >= np.min(corrected_t_action)) & (self.t_video <= np.max(corrected_t_action))
        self.aligned_t = self.t_video[valid_mask]

        self.aligned_pose = interp1d(corrected_t_action, self.ep['eef_pose'], axis=0, bounds_error=False,
                                     fill_value="extrapolate")(self.aligned_t)
        self.aligned_gripper = interp1d(corrected_t_action, self.ep['gripper'], kind='nearest', bounds_error=False,
                                        fill_value="extrapolate")(self.aligned_t)
        self.aligned_vl_embed = self.ep['vl_embedding'][valid_mask]
        print(f"  -> [Step 2] Timeline alignment completed. Unified length: {len(self.aligned_t)} frames (20Hz)")

    def step3_and_4_segmentation(self):
        """
        [Math & Algorithms - Signal Filtering & Integration]
        1. Median Filter:
           A non-linear filter that replaces the center value with the median of the sliding window.
           Excellent for removing salt-and-pepper noise (e.g., jittering from gripper sensors).

        2. Local Extrema Detection:
           Finds valleys by looking for changes in the sign of the first derivative.
           Here, we find local minima of velocity `v` by finding the peaks of `-v`.
           Velocity valleys usually indicate the end of one action and the start of another.

        3. Action Energy Integration:
           Energy = \int_{t0}^{t1} v(t) dt \approx \sum_{i} v_i * \Delta t
           Represents the total magnitude of movement or trajectory length over the action's lifespan.
        """
        # Debounce using Median Filter
        self.clean_gripper = medfilt(self.aligned_gripper, kernel_size=5)

        # Absolute difference for boundary detection: diff = |g[i] - g[i-1]|
        gripper_diff = np.abs(np.diff(self.clean_gripper, prepend=self.clean_gripper[0]))
        gripper_boundaries = np.where(gripper_diff > 0.5)[0]

        pos = self.aligned_pose[:, :3]
        self.aligned_vel = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[0:1]), axis=1) / self.dt_video

        # Smooth velocity again to make peak-finding more robust
        self.smooth_vel = medfilt(self.aligned_vel, kernel_size=11)
        valleys, _ = find_peaks(-self.smooth_vel, distance=20, prominence=0.01)

        self.all_boundaries = np.unique(np.concatenate(([0], gripper_boundaries, valleys, [len(self.aligned_t) - 1])))
        self.all_boundaries.sort()

        events = []
        start_idx = self.all_boundaries[0]
        for end_idx in self.all_boundaries[1:]:
            if end_idx - start_idx < 10: continue

            action_delta = (pos[end_idx] - pos[start_idx]).tolist()
            # Discrete integration to calculate total motion energy (trajectory length)
            action_energy = float(np.sum(self.aligned_vel[start_idx:end_idx] * self.dt_video))
            # Mean Pooling for high-dimensional visual features
            vl_feature = np.mean(self.aligned_vl_embed[start_idx:end_idx], axis=0)

            events.append({
                "episode_id": self.ep["episode_id"],
                "event_id": f"{self.ep['episode_id']}_ev_{len(events)}",
                "start_frame": int(start_idx), "end_frame": int(end_idx),
                "duration_sec": round(float((end_idx - start_idx) * self.dt_video), 3),
                "caption_level": "action", "caption": self.ep["task_caption"],
                "action_delta": [round(x, 4) for x in action_delta],
                "action_energy": round(action_energy, 4),
                "gripper_transitions": int(np.sum(gripper_diff[start_idx:end_idx])),
                "quality_flags": self.quality_flags.copy(),
                "_vl_feature_raw": vl_feature, "_act_feature_raw": action_delta + [action_energy]
            })
            start_idx = end_idx

        print(f"  -> [Step 3&4] Semantic segmentation completed, extracted {len(events)} atomic Events.")
        return events

    def plot_debug_alignment(self):
        print(f"  -> [Plot Validation] Displaying {self.ep['episode_id']} alignment and segmentation debug plot...")
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Debug Plot for {self.ep['episode_id']}: Alignment & Segmentation", fontsize=14,
                     fontweight='bold')

        ax1 = axes[0]
        ax1.plot(self.aligned_t, self.ep['video_motion'][:len(self.aligned_t)], label='Video Motion', color='blue',
                 alpha=0.6)
        interp_raw = interp1d(self.t_action, self.raw_eef_speed, bounds_error=False, fill_value=0.0)
        raw_speed_20hz = interp_raw(self.aligned_t)
        ax1.plot(self.aligned_t, raw_speed_20hz, label='Raw EEF Speed (Misaligned)', color='red', linestyle='--',
                 alpha=0.5)
        ax1.plot(self.aligned_t, self.aligned_vel, label='Aligned EEF Speed', color='green')
        ax1.set_title("Step 1&2: Video vs Action Synchronization")
        ax1.legend(loc="upper right")

        ax2 = axes[1]
        ax2.plot(self.aligned_t, self.aligned_vel, color='lightgray', label='Raw Aligned Speed')
        ax2.plot(self.aligned_t, self.smooth_vel, color='green', label='Smoothed Speed')
        ax2.set_title("Step 3: Kinematic Valleys Detection")
        ax2.legend(loc="upper right")

        ax3 = axes[2]
        ax3.plot(self.aligned_t, self.clean_gripper, color='darkred', linewidth=2, label='Debounced Gripper')
        ax3.set_yticks([0, 1]);
        ax3.set_yticklabels(['Closed (0)', 'Open (1)'])
        ax3.set_title("Step 3: Event Boundaries")

        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
        for i in range(len(self.all_boundaries) - 1):
            s, e = self.all_boundaries[i], self.all_boundaries[i + 1]
            if e - s < 10: continue
            t_s, t_e = self.aligned_t[s], self.aligned_t[e]
            for ax in [ax2, ax3]:
                ax.axvspan(t_s, t_e, color=colors[i % len(colors)], alpha=0.3)
                ax.axvline(t_e, color='black', linestyle=':', alpha=0.5)
            ax3.text((t_s + t_e) / 2, 0.5, f"Ev {i}", ha='center', va='center')

        plt.tight_layout()
        plt.show()


# ==========================================
# 3. Clustering and Sampling
# ==========================================
def step5_cluster_events(events, n_vl_clusters=2, n_act_clusters=3):
    """
    [Math & Algorithms - Machine Learning Clustering]
    K-Means Clustering:
    The objective is to minimize the Sum of Squared Errors (SSE) from all data points to their respective cluster centroids.
    Objective Function: J = \sum_{j=1}^{k} \sum_{i=1}^{n} ||x_i^{(j)} - c_j||^2
    Where x_i is the feature vector and c_j is the centroid. It converges iteratively.
    """
    print(f"\n[Step 5] Performing clustering on {len(events)} events using K-Means feature clustering...")
    vl_features = np.array([e['_vl_feature_raw'] for e in events])
    act_features = np.array([e['_act_feature_raw'] for e in events])

    kmeans_vl = KMeans(n_clusters=n_vl_clusters, n_init=10, random_state=42).fit(vl_features)
    kmeans_act = KMeans(n_clusters=n_act_clusters, n_init=10, random_state=42).fit(act_features)

    for i, e in enumerate(events):
        e['vl_cluster'] = int(kmeans_vl.labels_[i])
        e['action_cluster'] = int(kmeans_act.labels_[i])
    print(
        f"  -> Clustering completed: Vision-language semantics divided into {n_vl_clusters} clusters, physical trajectories divided into {n_act_clusters} clusters.")
    return events


class BalancedEventSampler:
    def __init__(self, events):
        self.valid_events = [e for e in events if not e.get('quality_flags')]
        self.buckets = collections.defaultdict(list)
        for e in self.valid_events:
            self.buckets[(e['vl_cluster'], e['action_cluster'])].append(e)
        self.bucket_keys = list(self.buckets.keys())
        print(
            f"\n[Step 6] Sampler initialization: {len(self.valid_events)} valid events assigned to {len(self.bucket_keys)} 2D buckets.")

    def sample(self, batch_size):
        # Uniform Distribution Sampling to prevent long-tail imbalance:
        # 1. Randomly pick a bucket (probability 1/N buckets)
        # 2. Randomly pick an event inside that bucket (probability 1/M items)
        return [random.choice(self.buckets[random.choice(self.bucket_keys)]) for _ in range(batch_size)]


# ==========================================
# 4. Visualization Module (Dashboard)
# ==========================================
def plot_dataset_dashboard(events):
    """
    [Math & Algorithms - Dimensionality Reduction]
    Principal Component Analysis (PCA):
    An orthogonal linear transformation. By performing Eigen Decomposition on the covariance matrix,
    it projects high-dimensional data (e.g., 128-D visual embeddings) onto the orthogonal axes (Principal Components)
    that maximize the variance.
    Here, n_components=2 maps the data into a 2D plane for scatter plot visualization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("WAM Dataset Builder: Global Statistics & Clustering", fontsize=18, fontweight='bold')

    ax1 = axes[0, 0]
    ax1.hist([e['duration_sec'] for e in events], bins=15, color='teal', edgecolor='black', alpha=0.7)
    ax1.set_title("1. Event Duration Distribution", fontsize=12)
    ax1.set_xlabel("Duration (Seconds)")

    ax2 = axes[0, 1]
    vl_clusters, act_clusters = [e['vl_cluster'] for e in events], [e['action_cluster'] for e in events]
    heatmap = np.zeros((max(vl_clusters) + 1, max(act_clusters) + 1))
    for v, a in zip(vl_clusters, act_clusters): heatmap[v, a] += 1
    cax = ax2.imshow(heatmap, cmap='YlOrRd', origin='lower')
    fig.colorbar(cax, ax=ax2, label='Number of Events')
    ax2.set_title("2. 2D Bucket Heatmap (VL vs Action)", fontsize=12)
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            ax2.text(j, i, int(heatmap[i, j]), ha="center", va="center",
                     color="black" if heatmap[i, j] < heatmap.max() / 2 else "white")

    ax3 = axes[1, 0]
    # Reduce dimensions using PCA
    pca_vl = PCA(n_components=2).fit_transform(np.array([e['_vl_feature_raw'] for e in events]))
    s1 = ax3.scatter(pca_vl[:, 0], pca_vl[:, 1], c=vl_clusters, cmap='tab10', s=100, alpha=0.8, edgecolors='k')
    ax3.set_title("3. Vision-Language Embeddings (PCA 2D)", fontsize=12)
    fig.colorbar(s1, ax=ax3, label="VL Cluster ID")

    ax4 = axes[1, 1]
    # Reduce dimensions using PCA
    pca_act = PCA(n_components=2).fit_transform(np.array([e['_act_feature_raw'] for e in events]))
    s2 = ax4.scatter(pca_act[:, 0], pca_act[:, 1], c=act_clusters, cmap='Dark2', s=100, alpha=0.8, edgecolors='k')
    ax4.set_title("4. Action Trajectory Features (PCA 2D)", fontsize=12)
    fig.colorbar(s2, ax=ax4, label="Action Cluster ID")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ==========================================
# 5. Execution Entry
# ==========================================
def main():
    print("🚀 [Step 0] Generating simulated physical trajectories (15 episodes)...")
    episodes = generate_mock_episodes(15)

    all_events = []
    for i, ep in enumerate(episodes):
        processor = EpisodeProcessor(ep)
        should_plot = True if i == 0 else False
        events = processor.process_all(show_debug_plot=should_plot)
        all_events.extend(events)

    all_events = step5_cluster_events(all_events, n_vl_clusters=2, n_act_clusters=4)
    sampler = BalancedEventSampler(all_events)

    print("\n[Step 6 Test] Sample a batch (Size=4):")
    for b in sampler.sample(4):
        print(
            f"  [-] {b['event_id']} (Duration: {b['duration_sec']}s) | Bucket: (VL:{b['vl_cluster']}, Act:{b['action_cluster']})")

    print("\n[Step 7] Plotting global statistics and clustering dashboard (see popup window)...")
    plot_dataset_dashboard(all_events)

    # Clean up raw features before saving
    for e in all_events:
        e.pop('_vl_feature_raw', None)
        e.pop('_act_feature_raw', None)

    output_file = "events.jsonl"
    with open(output_file, "w") as f:
        for e in all_events: f.write(json.dumps(e) + "\n")

    print("\n==========================================")
    print(f"✅ Processing complete! Generated {len(all_events)} events saved to: {os.path.abspath(output_file)}")
    print("==========================================")


if __name__ == "__main__":
    main()
