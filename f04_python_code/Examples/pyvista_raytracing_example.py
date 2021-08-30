import pyvista as pv

# Source: https://docs.pyvista.org/examples/99-advanced/ray-trace.html

# Create source to ray trace
sphere = pv.Sphere(radius=0.85)

# Define line segment
start = [0, 0, 0]
stop = [0.25, 1, 0.5]

# Perform ray trace
points, ind = sphere.ray_trace(start, stop)

# Create geometry to represent ray trace
ray = pv.Line(start, stop)
intersection = pv.PolyData(points)

# Render the result
p = pv.Plotter()
p.add_mesh(sphere,
           show_edges=True, opacity=0.5, color="w",
           lighting=False, label="Test Mesh")
p.add_mesh(ray, color="blue", line_width=5, label="Ray Segment")
p.add_mesh(intersection, color="maroon",
           point_size=25, label="Intersection Points")
p.add_legend()
p.show()


# https://github.com/pyvista/pyvista-support/issues/173
import numpy as np
import pyvista as pv

sphere = pv.Sphere(radius=0.85)

def ray_trace(start, stop):
    """Pass two same sized arrays of the start and stop coordinates for all rays"""
    assert start.shape == stop.shape
    assert start.ndim == 2
    # Launch this for loop in parallel if needed
    zeroth_cellids = []
    for i in range(len(start)):
        _, ids = sphere.ray_trace(start[i], stop[i])
        if len(ids) < 1:
            v = None
        else:
            v = ids[0]
        zeroth_cellids.append(v)
    return np.array(zeroth_cellids)

start = np.array([[0,0,0], [0,0,0]])
stop = np.array([[0.25, 1, 0.5], [0.5, 1, 0.25]])
cell_ids = ray_trace(start, stop)

cells = sphere.extract_cells(cell_ids)

p = pv.Plotter(notebook=0)
p.add_mesh(sphere,
           show_edges=True, opacity=0.5, color="w",
           lighting=False, label="Test Mesh")
p.add_arrows(start, stop, mag=1, color=True, opacity=0.5, )
p.add_mesh(cells, color="maroon",
           label="Intersection Cells")
p.add_legend()
p.show()