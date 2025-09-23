import bpy
import bmesh
import mathutils
import random
import os
import itertools
import bisect
import math
from collections import namedtuple
import sys
import argparse

# Default values
TARGET_POINTS = 500_000
INCLUDE_COLORS = True

argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]  # everything after '--'
else:
    argv = []

parser = argparse.ArgumentParser(description="Blender script parameters")
parser.add_argument("--points", type=int, default=TARGET_POINTS, help="Target total points")
parser.add_argument("--colors", type=int, choices=[0,1], default=int(INCLUDE_COLORS), help="Include colors (1=True, 0=False)")

args = parser.parse_args(argv)

TARGET_POINTS = args.points
INCLUDE_COLORS = bool(args.colors)

OBJECT_SIZE_EXPONENT = 1.3
FACE_SIZE_EXPONENT = 0.8
BASE_DENSITY_FALLBACK = 10.0
# ---------------------------------------------------

Sample = namedtuple("Sample", ["co", "color"])

def sample_face_points_with_color(face, world_matrix, density, color_layer, material_color):
    verts = [loop.vert.co.copy() for loop in face.loops]
    loops = [loop for loop in face.loops]
    tris = []
    loop_tris = []
    if len(verts) == 3:
        tris = [tuple(verts)]
        loop_tris = [tuple(loops)]
    else:
        for i in range(1, len(verts) - 1):
            tris.append((verts[0], verts[i], verts[i+1]))
            loop_tris.append((loops[0], loops[i], loops[i+1]))

    # triangle areas
    areas = []
    for v0, v1, v2 in tris:
        areas.append(((v1 - v0).cross(v2 - v0)).length * 0.5)
    total_area = sum(areas)
    if total_area <= 0.0 or density <= 0.0:
        return []

    exact_n = total_area * density
    n_int = int(exact_n)
    remainder = exact_n - n_int
    num_samples = n_int + 1 if random.random() < remainder else n_int

    cum_areas = list(itertools.accumulate(areas))
    sampled = []
    for _ in range(num_samples):
        r = random.random() * total_area
        idx = bisect.bisect_left(cum_areas, r)
        v0, v1, v2 = tris[idx]
        l0, l1, l2 = loop_tris[idx]

        u = random.random()
        v = random.random()
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        w = 1.0 - u - v
        p_local = v0 * u + v1 * v + v2 * w
        p_world = world_matrix @ p_local

        col = None
        if color_layer is not None:
            c0 = l0[color_layer]
            c1 = l1[color_layer]
            c2 = l2[color_layer]
            def to_rgb(c):
                try:
                    return (float(c[0]), float(c[1]), float(c[2]))
                except Exception:
                    # fallback if weird structure
                    return (1.0, 1.0, 1.0)
            cr0 = to_rgb(c0)
            cr1 = to_rgb(c1)
            cr2 = to_rgb(c2)
            col = (cr0[0]*u + cr1[0]*v + cr2[0]*w,
                   cr0[1]*u + cr1[1]*v + cr2[1]*w,
                   cr0[2]*u + cr1[2]*v + cr2[2]*w)
        elif material_color is not None:
            col = material_color
        else:
            col = None

        sampled.append(Sample(co=p_world, color=col))

    return sampled


def estimate_scene_terms(object_size_exponent, face_size_exponent):
    denom = 0.0
    obj_meta = []
    depsgraph = bpy.context.evaluated_depsgraph_get()

    for obj in bpy.data.objects:
        if obj.type != 'MESH' and obj.type != 'CURVE':
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        try:
            mesh_eval = obj_eval.to_mesh()
        except Exception:
            # skip objects that can't be converted
            continue

        # compute face areas if present
        if len(mesh_eval.polygons) > 0:
            face_areas = [poly.area for poly in mesh_eval.polygons]
            surface_area = sum(face_areas)
            sum_face_powers = sum((a ** face_size_exponent) for a in face_areas)
        else:
            edge_lengths = [ (mesh_eval.vertices[e.vertices[0]].co - mesh_eval.vertices[e.vertices[1]].co).length
                            for e in mesh_eval.edges ]
            pseudo_area_scale = 0.001
            face_areas = [l * pseudo_area_scale for l in edge_lengths]
            surface_area = sum(face_areas)
            sum_face_powers = sum((a ** face_size_exponent) for a in face_areas) if face_areas else 0.0

        term = 0.0
        if surface_area > 0.0 and sum_face_powers > 0.0:
            term = (surface_area ** (object_size_exponent - 1.0)) * sum_face_powers
        else:
            term = 0.0

        denom += term
        obj_meta.append({
            "name": obj.name,
            "surface_area": surface_area,
            "sum_face_powers": sum_face_powers,
            "term": term
        })

        obj_eval.to_mesh_clear()

    return denom, obj_meta


def compute_base_density(target_points, object_size_exponent, face_size_exponent, fallback=BASE_DENSITY_FALLBACK):
    denom, meta = estimate_scene_terms(object_size_exponent, face_size_exponent)
    if denom <= 0.0:
        print("Warning: computed denominator <= 0. Using fallback base density:", fallback)
        return fallback
    base_density = float(target_points) / denom
    # clamp to reasonable values
    base_density = max(base_density, 1e-6)
    base_density = min(base_density, 1e9)
    print(f"Computed base_density={base_density:.6f} to reach ~{target_points} points (denom={denom:.6e})")
    return base_density


def get_object_material_color(obj):
    if not obj.material_slots:
        return None
    mat = obj.material_slots[0].material
    if mat is None:
        return None
    if mat.use_nodes and mat.node_tree:
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                try:
                    col = node.inputs['Base Color'].default_value
                except Exception:
                    try:
                        col = node.inputs[0].default_value
                    except Exception:
                        col = None
                if col:
                    try:
                        return (float(col[0]), float(col[1]), float(col[2]))
                    except Exception:
                        pass
    try:
        c = mat.diffuse_color
        return (float(c[0]), float(c[1]), float(c[2]))
    except Exception:
        return None


def sample_mesh_surface_with_color(obj, base_density, object_size_exponent, face_size_exponent, include_colors):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh_eval = obj_eval.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(mesh_eval)
    bm.faces.ensure_lookup_table()
    wm = obj.matrix_world

    surface_area = sum(f.calc_area() for f in bm.faces) if bm.faces else 0.0
    multiplier = surface_area ** (object_size_exponent - 1.0) if surface_area > 0.0 else 1.0
    effective_density = base_density * multiplier
    effective_density = max(effective_density, base_density * 0.1)
    effective_density = min(effective_density, base_density * 1e6)

    color_layer = None
    if include_colors:
        color_layer = bm.loops.layers.color.active

    mat_color = get_object_material_color(obj) if include_colors else None

    samples = []
    for face in bm.faces:
        face_area = face.calc_area()
        face_multiplier = (face_area ** (face_size_exponent - 1.0)) if face_area > 0.0 else 1.0
        face_density = effective_density * face_multiplier
        if face_density <= 0.0:
            continue
        pts = sample_face_points_with_color(face, wm, density=face_density, color_layer=color_layer, material_color=mat_color)
        samples.extend(pts)

    if not bm.faces and len(mesh_eval.edges) > 0:
        total_len = sum((mesh_eval.vertices[e.vertices[0]].co - mesh_eval.vertices[e.vertices[1]].co).length for e in mesh_eval.edges)
        if total_len > 0:
            pseudo_area_scale = 0.001
            expected_points = base_density * ( (total_len * pseudo_area_scale) ** (object_size_exponent - 1.0) ) * ((total_len * pseudo_area_scale) ** face_size_exponent)
            n_int = int(expected_points)
            if random.random() < (expected_points - n_int):
                n_int += 1
            edge_lengths = [ (mesh_eval.vertices[e.vertices[0]].co - mesh_eval.vertices[e.vertices[1]].co).length for e in mesh_eval.edges ]
            cum_lengths = list(itertools.accumulate(edge_lengths))
            for _ in range(n_int):
                r = random.random() * cum_lengths[-1]
                idx = bisect.bisect_left(cum_lengths, r)
                e = mesh_eval.edges[idx]
                v0 = mesh_eval.vertices[e.vertices[0]].co
                v1 = mesh_eval.vertices[e.vertices[1]].co
                t = random.random()
                p_local = v0 * (1 - t) + v1 * t
                p_world = wm @ p_local
                col = mat_color if include_colors else None
                samples.append(Sample(co=p_world, color=col))

    obj_eval.to_mesh_clear()
    bm.free()
    return samples


def normalize_points_and_colors(samples):
    if not samples:
        return [], 1.0, mathutils.Vector((0.0, 0.0, 0.0))
    centroid = mathutils.Vector((0.0, 0.0, 0.0))
    for s in samples:
        centroid += s.co
    centroid /= len(samples)
    max_d = max((s.co - centroid).length for s in samples)
    scale = 1.0 / max_d if max_d > 0.0 else 1.0
    normalized = [Sample(co=(s.co - centroid) * scale, color=s.color) for s in samples]
    return normalized, scale, centroid


def write_ply(filepath, samples, include_colors):
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(samples)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if include_colors:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        for s in samples:
            x, y, z = s.co.x, s.co.y, s.co.z
            if include_colors and s.color is not None:
                r = int(max(0, min(255, round(s.color[0] * 255.0))))
                g = int(max(0, min(255, round(s.color[1] * 255.0))))
                b = int(max(0, min(255, round(s.color[2] * 255.0))))
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            elif include_colors:
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {255} {255} {255}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def create_pointcloud_object(name, samples, include_colors):
    mesh_data = bpy.data.meshes.new(name + "_Mesh")
    coords = [(s.co.x, s.co.y, s.co.z) for s in samples]
    mesh_data.from_pydata(coords, [], [])
    mesh_data.update()

    if include_colors:
        try:
            ca = mesh_data.color_attributes.new(name="Col", type='FLOAT_COLOR', domain='POINT')
            color_data = [ (s.color if s.color is not None else (1.0,1.0,1.0)) for s in samples ]
            for i, c in enumerate(color_data):
                r,g,b = c
                try:
                    ca.data[i].color = (r, g, b, 1.0)
                except Exception:
                    try:
                        ca.data[i].color = (r, g, b)
                    except Exception:
                        pass
        except Exception as e:
            print("Note: couldn't create per-point color attribute:", e)

    pc_obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(pc_obj)
    return pc_obj


def main(target_points=TARGET_POINTS,
         include_colors=INCLUDE_COLORS,
         object_size_exponent=OBJECT_SIZE_EXPONENT,
         face_size_exponent=FACE_SIZE_EXPONENT):
    base_density = compute_base_density(target_points, object_size_exponent, face_size_exponent)

    all_samples = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH' or obj.type == 'CURVE':
            samples = sample_mesh_surface_with_color(
                obj,
                base_density=base_density,
                object_size_exponent=object_size_exponent,
                face_size_exponent=face_size_exponent,
                include_colors=include_colors
            )
            all_samples.extend(samples)

    if not all_samples:
        print("Error: No points generated. Scene needs Mesh or Curve objects.")
        return

    # Normalize
    normalized_samples, scale_factor, centroid = normalize_points_and_colors(all_samples)
    print(f"Total samples collected: {len(normalized_samples)} (target was {target_points})")
    # Save PLY
    if bpy.data.is_saved:
        base_dir = os.path.dirname(bpy.data.filepath)
        blend_name = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
        if INCLUDE_COLORS:
            ply_name = f"{blend_name}_{TARGET_POINTS}_colored_pointcloud.ply"
        else:
            ply_name = f"{blend_name}_{TARGET_POINTS}_pointcloud.ply"
    else:
        base_dir = bpy.app.tempdir
        ply_name = "untitled_pointcloud.ply"

    ply_path = os.path.join(base_dir, ply_name)
    write_ply(ply_path, normalized_samples, include_colors=include_colors)
    print(f"Exported .ply ({len(normalized_samples)} points) to:\n  {ply_path}")

    pc_obj = create_pointcloud_object("PointCloud_All", normalized_samples, include_colors=include_colors)
    bpy.ops.object.select_all(action='DESELECT')
    pc_obj.select_set(True)
    bpy.context.view_layer.objects.active = pc_obj


if __name__ == "__main__":
    main()
    filepath = bpy.data.filepath
    if filepath:
        bpy.ops.wm.save_as_mainfile(filepath=filepath, check_existing=False)
        print("Blend file saved successfully.")
    else:
        print("Warning: .blend file not saved; running in temp mode.")
