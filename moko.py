import dolfinx
import ufl
from mpi4py import MPI
import numpy as np
from basix.ufl import element
from dolfinx.fem.petsc import assemble_vector

# creazione mesh 
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

# creazione spazio vettoriale e funzione (u) di mio interesse 
v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
V = dolfinx.fem.functionspace(mesh, v_cg2)
u = dolfinx.fem.Function(V)
u.interpolate(lambda x: (np.sin(x[0]), np.cos(x[1])))




dt = dolfinx.fem.Constant(mesh, 0.01)

# Compute CFL number locally
unorm = ufl.sqrt(ufl.dot(u, u))  # local norm of the velocity
h = ufl.CellDiameter(mesh)           # local mesh size
cfl = 2*dt*unorm/h                

# creo lo spazio vettoriale della cfl_fun 
DG = dolfinx.fem.functionspace(mesh, ("DG", 0))
cfl_expr = dolfinx.fem.Expression(cfl,DG.element.interpolation_points())
cfl_fun = dolfinx.fem.Function(DG)
cfl_fun.interpolate(cfl_expr)
maxcfl = np.max(cfl_fun.x.array)
print("maxcfl = ", maxcfl)


# v = ufl.TrialFunction(DG)
# cfl_vec = assemble_vector(dolfinx.fem.form(cfl_fun*v/ufl.CellVolume(mesh)*ufl.dx))
# print(cfl_vec.array-cfl_fun.x.array)