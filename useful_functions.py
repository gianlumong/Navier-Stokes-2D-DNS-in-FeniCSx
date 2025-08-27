#Useful functions

from dolfinx import plot 
import pyvista
import numpy as np




#-------Plot the mesh to make sure everything is ok-----------#
def plt_mesh(domain, save_as_png: bool =False):
    tdim=domain.topology.dim # topological dimension of the domain (will be 2, since we have a 2D domain)
    fdim=tdim-1 # dimension of the facets (for 2D mesh the facets are segments, so 1D)
    print('tdim= ',tdim)

    domain.topology.create_connectivity(tdim, tdim)
    topology, cell_types, geometry = plot.vtk_mesh(domain, tdim) #vtk_mesh returns all useful info to build an unstructured grid (to be used in pyvista)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    plotter = pyvista.Plotter(off_screen=save_as_png,window_size=(15000,10000)) #creates the 2D and 3D rendering window
    plotter.add_mesh(grid, show_edges=True) #adds the mesh to this window (and shows the edges of the elements)
    plotter.view_xy()

    if save_as_png:
        name_png='mia_mesh.png'
        plotter.screenshot(name_png)
        print(f"Mesh saved as {name_png}")
    else:
        plotter.show()  #to show the plot on screen



import pyvista as pv
from pyvista import themes
from dolfinx import plot

import pyvista as pv
import numpy as np

def plot_multiple_dof_sets(domain, V, dof_sets, title="DoF sets", save_path="dof_sets_plot.png"):
    """
    Plot multiple sets of DoFs simultaneously in the space V.
    
    Parameters:
    - domain: dolfinx.mesh.Mesh
    - V: dolfinx.fem.FunctionSpace (vector space)
    - dof_sets: list of tuples (dofs, label, color)
    - title: plot title
    - save_path: path to save the plot
    """

    dof_coords = V.tabulate_dof_coordinates()
    dof_coords = dof_coords.reshape((-1, domain.geometry.dim))  # In case of vector spaces

    print("shape dof_coords:", dof_coords.shape)

    plotter = pv.Plotter()
    plotter.add_text(title, font_size=12)

    for dofs, label, color in dof_sets:
        coords = dof_coords[dofs]
        print(f"{label}: {len(dofs)} DoFs found")
        if len(coords) == 0:
            print(f" No DoFs found for '{label}'")
            continue
        
        # Convert to 3D for PyVista
        coords_3d = np.zeros((coords.shape[0], 3))
        coords_3d[:, :domain.geometry.dim] = coords

        points = pv.PolyData(coords_3d)
        plotter.add_mesh(points, color=color, point_size=10.0,
                         render_points_as_spheres=True, label=label)

    plotter.show_grid()
    plotter.view_xy()
    plotter.show(screenshot=save_path)







#Identify length (L) and height (H) of the domain
def L_and_H_domain(domain):
    coords = domain.geometry.x
    # Find index of the bottom-left node (min x, min y)
    idx_bottom_left = np.lexsort((coords[:, 1], coords[:, 0]))[0]
    bottom_left = coords[idx_bottom_left]

    # Find index of the top-right node (max x, max y)
    idx_top_right = np.lexsort((-coords[:, 1], -coords[:, 0]))[0]
    top_right = coords[idx_top_right]

    if bottom_left[0]==0 and bottom_left[1]==0:
        L= top_right[0] #domain length
        H=top_right[1] #domain height
        #print('domain length: ',L)
        #print('domain height:',H)
    else:
        RuntimeError('the domain does not have origin at (0,0)')
    return L,H



def L_H_cx_cy_r(domain, facet_tags):
    import numpy as np

    coords = domain.geometry.x

    # Find index of the bottom-left node (min x, min y)
    idx_bottom_left = np.lexsort((coords[:, 1], coords[:, 0]))[0]
    bottom_left = coords[idx_bottom_left]

    # Find index of the top-right node (max x, max y)
    idx_top_right = np.lexsort((-coords[:, 1], -coords[:, 0]))[0]
    top_right = coords[idx_top_right]

    # Check that the domain starts at (0, 0)
    if not (np.isclose(bottom_left[0], 0) and np.isclose(bottom_left[1], 0)):
        raise RuntimeError("The domain does not have origin at (0, 0)")

    # Length and height of the domain
    L = top_right[0]
    H = top_right[1]

    # Extract facets marked as "obstacle" (marker 40)
    obstacle_marker = 40
    obstacle_facets = facet_tags.indices[facet_tags.values == obstacle_marker]

    # Build facet -> vertex connectivity
    domain.topology.create_connectivity(domain.topology.dim - 1, 0)

    # Find all vertices connected to the obstacle facets
    obstacle_vertices = []
    for facet in obstacle_facets:
        verts = domain.topology.connectivity(domain.topology.dim - 1, 0).links(facet)
        obstacle_vertices.extend(verts)

    obstacle_vertices = np.unique(obstacle_vertices)
    obstacle_coords = coords[obstacle_vertices]

    # Use only x, y components
    obstacle_coords_2d = obstacle_coords[:, :2]

    # Compute centroid (center of obstacle)
    c_x = np.mean(obstacle_coords_2d[:, 0])
    c_y = np.mean(obstacle_coords_2d[:, 1])

    # Compute average radius
    r = np.mean(np.linalg.norm(obstacle_coords_2d - np.array([c_x, c_y]), axis=1))

    return L, H, c_x, c_y, r
