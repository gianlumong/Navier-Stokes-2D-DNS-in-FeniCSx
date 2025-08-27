from dolfinx.fem import Constant
from ufl import sqrt, dot 
from ufl import CellDiameter, MaxCellEdgeLength
from dolfinx.fem import Expression, Function, functionspace
import numpy as np 
from basix.ufl import element

def CFLnumber(mesh,u,dt):
    dt=Constant(mesh, dt) # making dt a fem.Constant (i don't know if this is necessary)

    # Compute CFL number locally    
    unorm = sqrt(dot(u,u))
    h = CellDiameter(mesh)
    # print('h:',type(h)) ----> <class 'ufl.geometry.CellDiameter'>
    cfl = 2*dt*unorm/h # Perché 2?????????????????

    # creo lo spazio vettoriale della cfl_fun 
    DG = functionspace(mesh, ("DG", 0))
    cfl_expr = Expression(cfl,DG.element.interpolation_points())
    cfl_fun = Function(DG)
    cfl_fun.interpolate(cfl_expr)
    max_CFL = np.max(cfl_fun.x.array)
    
    
    # print("maxcfl = ", max_CFL)
    
    # potrei anche calcolare il dt massimo che potrei utilizzare per avere un 
    # max_cfl=0.99, quindi comunque <1; ma non me ne faccio nulla per adesso sinceramente 

    if max_CFL>1:
        raise ValueError('Ho un CFL number >1 in qualche punto della mesh!! Quindi non va bene!')
    return max_CFL