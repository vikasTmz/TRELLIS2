import numpy as np
import sys
import random
from collections import defaultdict, deque, Counter


class VertexSegmenter:
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.vertices = []
        self.raw_normals = []

        # Mappings
        self.vertex_normal_accum = defaultdict(lambda: np.zeros(3))
        self.vertex_counts = defaultdict(int)
        self.adj_list = defaultdict(set)

        self.faces_indices = []
        self.final_normals = None
        self.labels = None

    def load_and_process(self):
        """Parses OBJ, builds adjacency, computes canonical normals."""
        print(f"Loading {self.obj_path}...")

        with open(self.obj_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                vals = line.split()
                if not vals:
                    continue

                if vals[0] == "v":
                    self.vertices.append([float(x) for x in vals[1:4]])
                elif vals[0] == "vn":
                    self.raw_normals.append([float(x) for x in vals[1:4]])
                elif vals[0] == "f":
                    v_idxs = []
                    vn_idxs = []
                    for w in vals[1:]:
                        parts = w.split("/")
                        v_idx = int(parts[0]) - 1
                        v_idxs.append(v_idx)
                        if len(parts) >= 3 and parts[2]:
                            vn_idx = int(parts[2]) - 1
                            vn_idxs.append(vn_idx)

                    self.faces_indices.append(v_idxs)

                    # Build Adjacency & Accumulate Normals
                    num_v = len(v_idxs)
                    for i in range(num_v):
                        curr_v = v_idxs[i]
                        next_v = v_idxs[(i + 1) % num_v]
                        self.adj_list[curr_v].add(next_v)
                        self.adj_list[next_v].add(curr_v)

                        if len(vn_idxs) == num_v:
                            norm_vec = np.array(self.raw_normals[vn_idxs[i]])
                            self.vertex_normal_accum[curr_v] += norm_vec
                            self.vertex_counts[curr_v] += 1

        # Canonical normals
        num_vertices = len(self.vertices)
        self.final_normals = np.zeros((num_vertices, 3))

        for i in range(num_vertices):
            if self.vertex_counts[i] > 0:
                avg_n = self.vertex_normal_accum[i] / self.vertex_counts[i]
                norm_len = np.linalg.norm(avg_n)
                if norm_len > 1e-6:
                    self.final_normals[i] = avg_n / norm_len
            else:
                self.final_normals[i] = np.array([0, 1, 0])

    def segment(self, angle_threshold_degrees=15):
        """Standard Region Growing."""
        print(f"Segmenting (Threshold: {angle_threshold_degrees}°)...")
        threshold_rad = np.radians(angle_threshold_degrees)
        min_cos = np.cos(threshold_rad)

        num_vertices = len(self.vertices)
        self.labels = -1 * np.ones(num_vertices, dtype=int)
        current_label = 0

        for i in range(num_vertices):
            if self.labels[i] != -1:
                continue

            q = deque([i])
            self.labels[i] = current_label

            while q:
                curr_idx = q.popleft()
                curr_n = self.final_normals[curr_idx]

                for neighbor_idx in self.adj_list[curr_idx]:
                    if self.labels[neighbor_idx] == -1:
                        neighbor_n = self.final_normals[neighbor_idx]
                        dot = np.clip(np.dot(curr_n, neighbor_n), -1.0, 1.0)

                        if dot >= min_cos:
                            self.labels[neighbor_idx] = current_label
                            q.append(neighbor_idx)
            current_label += 1

        print(f"Initial segmentation: {current_label} segments.")
        return current_label

    def smooth_labels(self, iterations=1):
        """
        Re-assigns labels based on the majority label of neighbors.
        This fixes boundary artifacts where edge vertices get unique labels.
        """
        print(f"Smoothing labels ({iterations} iterations)...")

        for it in range(iterations):
            # Create a copy so we don't cascade changes within the same iteration
            new_labels = self.labels.copy()
            changes = 0

            for i in range(len(self.vertices)):
                # Get labels of all neighbors
                neighbor_indices = list(self.adj_list[i])
                if not neighbor_indices:
                    continue

                neighbor_labels = [self.labels[n] for n in neighbor_indices]

                # If vertex has neighbors, find the most frequent label
                if neighbor_labels:
                    # Counter finds the most common elements
                    # most_common(1) returns [(label, count)]
                    most_freq_label, count = Counter(neighbor_labels).most_common(1)[0]

                    # Only update if the current label is different
                    if new_labels[i] != most_freq_label:
                        new_labels[i] = most_freq_label
                        changes += 1

            self.labels = new_labels
            print(f"  Iteration {it+1}: updated {changes} vertices.")

            if changes == 0:
                break

    def export_weighted_sampled_ply(self, ply_path, txt_path, weight, num_samples=1000):
        """
        Samples vertices based on label rarity. Vertices in small segments
        have higher weight than vertices in large segments.
        """
        print(f"Performing label-weighted sampling ({num_samples} points)...")

        # 1. Calculate frequency of each label
        label_counts = Counter(self.labels)

        # 2. Compute weights for every vertex (inverse of label count)
        # Weight = 1 / FrequencyOfThatLabel
        weights = np.array(
            [
                (1.0 / label_counts[lbl] ** weight) if label_counts[lbl] > 100 else 0.0
                for lbl in self.labels
            ]
        )

        # 3. Normalize weights to sum to 1
        weights /= weights.sum()

        # 4. Perform weighted random sampling without replacement
        # If total vertices < num_samples, take all vertices
        n_to_pick = min(num_samples, len(self.vertices))
        sampled_indices = np.random.choice(
            len(self.vertices), size=n_to_pick, replace=False, p=weights
        )

        print(f"Exporting weighted point cloud to {ply_path}...")
        self._write_point_cloud(ply_path, txt_path, sampled_indices)

    def _write_point_cloud(self, ply_path, txt_path, indices):
        """Helper to write point cloud data to PLY."""
        num_segments = self.labels.max() + 1
        np.random.seed(42)
        seg_colors = np.random.randint(0, 255, size=(num_segments, 3))

        with open(ply_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(indices)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for idx in indices:
                v = self.vertices[idx]
                c = seg_colors[self.labels[idx]]
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

        # print(f"Exporting Labels to {txt_path}...")
        # with open(txt_path, "w") as f:
        #     for idx in indices:
        #         v = self.vertices[idx]
        #         label = self.labels[idx]
        #         f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {label:d}\n")

    def export_ply(self, ply_path):
        print(f"Exporting PLY to {ply_path}...")

        # Generate random colors for segments
        num_segments = self.labels.max() + 1
        np.random.seed(42)
        seg_colors = np.random.randint(0, 255, size=(num_segments, 3))

        with open(ply_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self.vertices)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            # f.write(f"element face {len(self.faces_indices)}\n")
            # f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            for i, v in enumerate(self.vertices):
                lbl = self.labels[i]
                c = seg_colors[lbl] if lbl != -1 else [0, 0, 0]
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

            # for v_idxs in self.faces_indices:
            #     s = " ".join(map(str, v_idxs))
            #     f.write(f"3 {s}\n")

    def export_labels_txt(self, txt_path):
        print(f"Exporting Labels to {txt_path}...")
        with open(txt_path, "w") as f:
            f.write("vertex_index segment_label\n")
            for idx, label in enumerate(self.labels):
                f.write(f"{idx} {label}\n")


# --- Execution ---
if __name__ == "__main__":
    import glob

    TXT_ROOT = "/home/vthamizharas/Documents/point2cad/assets/segments_thingi10k_gt"
    PLY_ROOT = "output/segments_thingi10k_gt"

    ext = "obj"
    inputs = glob.glob(f"assets/thingi10k_gt_shapes/engine*.{ext}")

    angle_threshold_degrees = 2
    smooth_iter = 3

    weight = 0.2
    num_samples = int(60e4)

    for input_file in inputs:
        print(f"Processing {input_file}...")
        filename = input_file.split("/")[-1].replace(f".{ext}", "")

        seg = VertexSegmenter(input_file)
        seg.load_and_process()

        # 30 degrees is a generous curve tolerance.
        # Use smaller (e.g., 5-10) for stricter planar detection.
        seg.segment(angle_threshold_degrees=angle_threshold_degrees)

        # 2. Smooth Boundaries
        # This looks at neighbors and adopts the most frequent label.
        # Usually 1 iteration is enough to fix single-pixel edge artifacts.
        seg.smooth_labels(iterations=smooth_iter)

        seg.export_ply(
            f"{PLY_ROOT}/{filename}-gemini_angle={str(angle_threshold_degrees)}.ply"
        )

        # seg.export_weighted_sampled_ply(
        #     f"{PLY_ROOT}/{filename}-geminiFPS_angle={str(angle_threshold_degrees)}.ply",
        #     f"{TXT_ROOT}/{filename}-gemini_angle={str(angle_threshold_degrees)}_FPS{weight}.xyzc",
        #     weight,
        #     num_samples=num_samples,
        # )

        # seg.export_labels_txt()
        print("All operations finished.")
