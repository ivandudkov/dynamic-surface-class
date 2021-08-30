import trimesh
import numpy as np

# test on a sphere primitive
mesh = trimesh.creation.icosphere()
axis = trimesh.creation.axis(axis_radius=0.01,axis_length=10)
# create some rays
ray_origins = np.array([[0, 0, -3],
                        [0, 0.5, -3]])
ray_directions = np.array([[0, 0, 1],
                           [0, 0, 1]])

# check out the docstring for intersects_location queries
print(mesh.ray.intersects_location.__doc__)

# run the mesh- ray query
locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions)

print('The rays hit the mesh at coordinates:\n', locations)

print('The rays with index: {} hit the triangles stored at mesh.faces[{}]'.format(index_ray, index_tri))

# stack rays into line segments for visualization as Path3D
ray_visualize = trimesh.load_path(np.hstack((ray_origins,
                                             ray_origins + ray_directions*5.0)).reshape(-1, 2, 3))

# unmerge so viewer doesn't smooth
mesh.unmerge_vertices()
# make mesh white- ish
mesh.visual.face_colors = [255,255,255,255]
mesh.visual.face_colors[index_tri] = [255, 0, 0, 255]

# create a visualization scene with rays, hits, and mesh
scene = trimesh.Scene([mesh,
                       ray_visualize, axis])

# show the visualization
scene.show()
