import numpy as np
import trimesh
from collections import deque, defaultdict


class NormalBaseRegionGrowing:
    """Class for performing normal-based region growing segmentation on a mesh."""

    def __init__(
        self,
    ):
        pass

    def normalize_rows(self, v):
        """Normalize each row vector to unit length; rows with near-zero norm stay zero."""
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n < 1e-12] = 1.0
        return v / n

    def compute_face_normals_from_vertex_normals(
        self, mesh: trimesh.Trimesh
    ) -> np.ndarray:
        """
        Average the per-vertex normals of each face and renormalize.
        Returns (F, 3) array of unit face normals.
        """
        # Ensure vertex normals exist
        if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
            mesh.rezero()  # ensure data consistency
            mesh.vertex_normals  # triggers computation inside trimesh

        vn = mesh.vertex_normals  # (V, 3)
        faces = mesh.faces  # (F, 3)
        fn = vn[faces].mean(axis=1)  # (F, 3)
        fn = self.normalize_rows(fn)
        return fn

    def build_face_adjacency_list(self, mesh: trimesh.Trimesh):
        """
        Build adjacency list for each face from trimesh.face_adjacency.
        Returns a list of lists: neighbors[i] = list of adjacent face indices to i.
        """
        F = len(mesh.faces)
        neighbors = [[] for _ in range(F)]
        adj = mesh.face_adjacency  # (E, 2)
        if adj is None or len(adj) == 0:
            return neighbors
        for a, b in adj:
            neighbors[a].append(b)
            neighbors[b].append(a)
        return neighbors

    def region_grow_by_normal_similarity(
        self,
        face_normals: np.ndarray,
        neighbors,
        angle_deg: float = 15.0,
        use_abs_dot: bool = False,
    ) -> np.ndarray:
        """
        Cluster faces by BFS using a cosine/angle threshold between adjacent face normals.
        Returns face_labels: (F,) with consecutive non-negative ints.
        """
        F = face_normals.shape[0]
        cos_thr = np.cos(np.deg2rad(angle_deg))
        labels = -np.ones(F, dtype=np.int32)

        def similar(i, j):
            d = float(np.dot(face_normals[i], face_normals[j]))
            if use_abs_dot:
                d = abs(d)
            return d >= cos_thr

        cur_label = 0
        for seed in range(F):
            if labels[seed] != -1:
                continue
            # start a new region
            labels[seed] = cur_label
            q = deque([seed])
            while q:
                f = q.popleft()
                for g in neighbors[f]:
                    if labels[g] == -1 and similar(f, g):
                        labels[g] = cur_label
                        q.append(g)
            cur_label += 1

        return labels

    def reassign_small_segments(
        self,
        face_labels: np.ndarray,
        mesh: trimesh.Trimesh,
        face_normals: np.ndarray,
        min_faces: int = 0,
    ) -> np.ndarray:
        """
        Optional: reassign segments with < min_faces to the most similar neighboring segment.
        Similarity is max dot between the small segment normal mean and neighboring segment means.
        """
        if min_faces <= 0:
            return face_labels

        # Build segment -> faces mapping
        seg2faces = defaultdict(list)
        for f, lab in enumerate(face_labels):
            seg2faces[int(lab)].append(f)

        # Precompute mean normal per segment
        seg_ids = sorted(seg2faces.keys())
        seg_mean = {}
        for sid in seg_ids:
            m = self.normalize_rows(
                face_normals[seg2faces[sid]].mean(axis=0, keepdims=True)
            )[0]
            seg_mean[sid] = m

        # Face adjacency to find neighboring segments
        neighbors = self.build_face_adjacency_list(mesh)

        fl = face_labels.copy()
        for sid in seg_ids:
            faces = seg2faces[sid]
            if len(faces) >= min_faces:
                continue

            # Collect neighboring segment IDs
            neighbor_sids = set()
            for f in faces:
                for g in neighbors[f]:
                    if fl[g] != sid:
                        neighbor_sids.add(int(fl[g]))
            if not neighbor_sids:
                continue

            # Choose the neighbor whose mean normal is most similar
            best_dst, best_sid = -1.0, None
            for nsid in neighbor_sids:
                d = float(np.dot(seg_mean[sid], seg_mean[nsid]))
                if d > best_dst:
                    best_dst, best_sid = d, nsid
            if best_sid is None:
                continue

            # Reassign
            for f in faces:
                fl[f] = best_sid

        # Relabel to consecutive integers (optional)
        unique = np.unique(fl)
        remap = {old: i for i, old in enumerate(unique)}
        return np.vectorize(remap.get)(fl).astype(np.int32)

    def face_labels_to_vertex_colors(
        self, mesh: trimesh.Trimesh, face_labels: np.ndarray, seed: int = 42
    ) -> np.ndarray:
        """
        Convert face labels to vertex colors by voting the most frequent
        face label incident to each vertex. Returns (V, 4) uint8 RGBA colors.
        """
        V = len(mesh.vertices)
        faces = mesh.faces
        # incident faces per vertex
        vert2faces = [[] for _ in range(V)]
        for f, (a, b, c) in enumerate(faces):
            vert2faces[a].append(f)
            vert2faces[b].append(f)
            vert2faces[c].append(f)

        # vertex labels by majority vote
        vlabels = np.zeros(V, dtype=np.int32)
        for v in range(V):
            labs = face_labels[vert2faces[v]]
            if len(labs) == 0:
                vlabels[v] = -1
            else:
                # mode
                counts = np.bincount(labs)
                vlabels[v] = np.argmax(counts)

        # color map
        rng = np.random.default_rng(seed)
        num_labels = int(np.max(vlabels)) + 1 if len(vlabels) else 0
        palette = (rng.random((num_labels, 3)) * 255).astype(np.uint8)

        colors = np.zeros((V, 4), dtype=np.uint8)
        for v in range(V):
            if vlabels[v] >= 0:
                colors[v, :3] = palette[vlabels[v]]
                colors[v, 3] = 255
            else:
                colors[v] = np.array([0, 0, 0, 255], dtype=np.uint8)
        return colors, vlabels

    def segment_mesh_by_normal(
        self,
        mesh: trimesh.Trimesh,
        angle_deg: float = 15.0,
        use_abs_dot: bool = False,
        min_faces: int = 0,
        recolor_mesh: bool = True,
    ):
        """
        Perform normal-similarity face clustering and (optionally) assign vertex colors.

        Parameters
        ----------
        mesh : trimesh.Trimesh
        angle_deg : float
            Angle threshold between adjacent face normals (degrees).
        use_abs_dot : bool
            If True, uses |dot| so flipped normals across a crease still match.
        min_faces : int
            If >0, reassign segments smaller than this to their best neighboring segment.
        recolor_mesh : bool
            If True, writes vertex colors into mesh.visual.vertex_colors.

        Returns
        -------
        face_labels : (F,) int array
        mesh : trimesh.Trimesh (possibly with vertex colors applied)
        """
        face_normals = self.compute_face_normals_from_vertex_normals(mesh)
        neighbors = self.build_face_adjacency_list(mesh)
        face_labels = self.region_grow_by_normal_similarity(
            face_normals, neighbors, angle_deg=angle_deg, use_abs_dot=use_abs_dot
        )
        if min_faces > 0:
            face_labels = self.reassign_small_segments(
                face_labels, mesh, face_normals, min_faces
            )

        if recolor_mesh:
            vcolors, vlabels = self.face_labels_to_vertex_colors(mesh, face_labels)
            mesh.visual.vertex_colors = vcolors

        points_labels = np.column_stack((mesh.vertices, vlabels))

        return face_labels, mesh, points_labels


def load_glb_as_mesh(path: str) -> trimesh.Trimesh:
    obj = trimesh.load(path, force="scene")  # GLB usually returns a Scene
    if isinstance(obj, trimesh.Trimesh):
        mesh = obj.copy()
    else:
        scene: trimesh.Scene = obj
        geoms = []
        for name, geom in scene.geometry.items():
            g = geom.copy()
            # Apply the node's world transform to the geometry
            # (get returns transform from node -> world)
            T = scene.graph.get(name)[0] if hasattr(scene.graph, "get") else np.eye(4)
            g.apply_transform(T)
            geoms.append(g)
        if not geoms:
            raise ValueError("GLB scene contains no geometry.")
        mesh = trimesh.util.concatenate(geoms)

    # ---- sanitize without remove_degenerate_faces ----
    mesh.remove_infinite_values()

    # # drop degenerate triangles
    # mask = getattr(mesh, "nondegenerate_faces", None)
    # if mask is not None:
    #     mesh.update_faces(mask)

    # drop duplicate faces if available
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()

    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)

    # Ensure vertex normals exist and are unit
    _ = mesh.vertex_normals  # triggers computation if absent
    return mesh


def strip_colors_and_uv(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Remove vertex/face colors and texture coordinates from a Trimesh.
    - Converts TextureVisuals -> ColorVisuals (drops UV, images/material maps)
    - Clears any per-vertex / per-face colors
    """
    # If visuals are textured, convert to plain color visuals (no UV, no texture)
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        mesh.visual = trimesh.visual.ColorVisuals(mesh)

    # Now mesh.visual is ColorVisuals; clear colors if present
    if hasattr(mesh.visual, "vertex_colors"):
        # Set to None/empty to avoid exporting color properties
        mesh.visual.vertex_colors = None
    if hasattr(mesh.visual, "face_colors"):
        mesh.visual.face_colors = None

    # Some loaders stash UVs on the visuals object; ensure it’s gone
    if hasattr(mesh.visual, "uv"):
        try:
            mesh.visual.uv = None
        except Exception:
            # On some versions it's a property without setter; just replace visuals
            mesh.visual = trimesh.visual.ColorVisuals(mesh)

    # Also clear material reference if any (prevents reattaching textures on export)
    if hasattr(mesh.visual, "material"):
        mesh.visual.material = None

    return mesh


# -------- Example usage --------
if __name__ == "__main__":
    import glob
    from trellis2.renderers import EnvMap
    import cv2
    import torch
    from trellis2.utils import render_utils
    import imageio
    import os
    import random

    import offline_render

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    # 1. Setup Environment Map
    envmap = EnvMap(
        torch.tensor(
            cv2.cvtColor(
                cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED),
                cv2.COLOR_BGR2RGB,
            ),
            dtype=torch.float32,
            device="cuda",
        )
    )

    # Load a mesh (must be triangles)
    ext = "obj"  # change to "obj" if needed
    objects = glob.glob(f"assets/thingi10k_gt_shapes/pcb*.{ext}")
    random.shuffle(objects)

    # Parameters:
    # - angle 15° groups fairly flat patches; increase to merge more aggressively
    # - use_abs_dot=True if meshes have inconsistent normal orientations
    # - min_faces=20 to eliminate tiny patches
    angle_deg = 1
    min_faces = 100
    use_abs_dot = False

    for object in objects:
        print(f"Processing {object}...")
        filename = object.split("/")[-1].replace(f".{ext}", "")

        if ext == "glb":
            mesh = load_glb_as_mesh(object)
            mesh = strip_colors_and_uv(mesh)
        else:
            mesh = trimesh.load(object, force="mesh")

        segmentation_algo = NormalBaseRegionGrowing()

        face_labels, colored_mesh, points_labels = (
            segmentation_algo.segment_mesh_by_normal(
                mesh,
                angle_deg=angle_deg,
                use_abs_dot=use_abs_dot,
                min_faces=min_faces,
                recolor_mesh=True,
            )
        )

        offline_render.offscreen_render_fixedviews(
            colored_mesh,
            f"output/segments_thingi10k_gt/{filename}_angle={str(angle_deg)}_minfaces={str(min_faces)}.jpg",
        )

        # imageio.mimsave(
        #     f"output/segments_thingi10k_gt/{filename}_angle={str(angle_deg)}_minfaces={str(min_faces)}.mp4",
        #     offline_render.offscreen_render_turntable(
        # colored_mesh,
        # None,
        # ),
        #     fps=15,
        # )

        # Save colored result (vertex colors)
        colored_mesh.export(
            f"output/segments_thingi10k_gt/{filename}_angle={str(angle_deg)}_minfaces={str(min_faces)}.ply",
            encoding="ascii",
        )

        # # Save points + labels
        # np.savetxt(
        #     f"/home/vthamizharas/Documents/point2cad/assets/segments_thingi10k_gt/{filename}_angle={str(angle_deg)}_minfaces={str(min_faces)}.xyzc",
        #     points_labels,
        #     fmt="%.6f %.6f %.6f %.0f",
        # )
