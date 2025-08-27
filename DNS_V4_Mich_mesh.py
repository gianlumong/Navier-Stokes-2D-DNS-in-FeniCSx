# non riesco bene a calcolare Cd e Cl perché non riesco a creare la normale (n) al disco con questa mesh!!!!

import gmsh
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element
from dolfinx.fem import (Constant, Function, functionspace,assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, set_bc)
from dolfinx.io import (VTXWriter, gmshio)
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs)

from tqdm.notebook import tqdm
from tqdm.autonotebook import tqdm

from xCFL_number import CFLnumber

assert MPI.COMM_WORLD.size == 1, "This example should only be run with 1 MPI process"

from  useful_functions import *
from dolfinx.io.gmshio import read_from_msh

############################################## DEFINING THE MESH ##############################################
# Oppure potrei leggere la mesh (ed altre info utili) da un file.msh 
path_msh="../mesh_files/michele_mesh/Mesh.msh"
mesh, cell_tags, facet_tags=read_from_msh(path_msh,MPI.COMM_WORLD,gdim=2)  


L, H, c_x, c_y, r= L_H_cx_cy_r(mesh, facet_tags) #individuo dimensioni del dominio 
#plt_mesh(domain,save_as_png=True) #Plotto o salvo in un png la mesh 
############################################## DEFINING THE MOST IMPORTANT VARIABLES ##############################################
dt=0.01  # nel problema del Poiseulle flow avevo: dt=0.000625
t=0

Re=80 # Oppure 150
nu=1/Re

restart=False
if restart:
    last_step_stop=100 # metti uguale al numero di step fatti fino al check-point! (importante solo se restart=True)
    t_restart=round((dt*last_step_stop),2)
    print('WE ARE RESTARTING!!!')
    print('')
    print('t_restart:', t_restart)
 
tol=1e-15 # oppure 1e-8 o addirittura 1e-15?
tot_save= 1 # 5000 # we save every -tot_save- steps

step=0
step_init = 1 #2000 # 2 mila (step a cui si inizia ad calcolare u_mean, p_mean ed f)
#stop_to_restart= 120 # step a cui stoppare il codice e salvafre tutto!! (per poi restartare)
step_final = 100 #100000000 # 100 milioni

############################################## DEFINING THE VARIATIONAL PROBLEM ##############################################
from dolfinx import fem
from dolfinx.fem import Function
from ufl import  TestFunction, TrialFunction

#We want to use a continuous piecewise quadratic elements for the velocity and continuous piecewise linear elements for the pressure.
v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = fem.functionspace(mesh, v_cg2)
Q = fem.functionspace(mesh, s_cg1)


# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

p = TrialFunction(Q)
q = TestFunction(Q)

# Create useful functions
u0 = Function(V) 
u1 = Function(V)
u0.name="u0"
u1.name='u1'

p0=Function(Q)
p1 = Function(Q)
p0.name="p0"
p1.name="p1"


#definisco le funzioni che mi serviranno durante il calcolo di mean flow e forzaggio 
u_mean_old = Function(V)
u_mean = Function(V)
u_fluct = Function(V)
u_mean.name= "u_mean"
u_mean_old.name= "u_mean_old"
u_fluct.name="u_fluct"

f_old = Function(V)
f = Function(V)
f_save = Function(V)
f.name="forcing"
f_old.name="f_old"


p_mean = Function(Q)
p_mean.name = "p_mean"

from dolfinx.fem import Function, dirichletbc, locate_dofs_topological
from ufl import div, dx, inner, lhs, rhs
from dolfinx import default_scalar_type

####################### DEFINING THE BOUNDARIES #######################
mesh_comm = MPI.COMM_WORLD
model_rank = 0
#definisco i vari marker 
inlet_marker, outlet_marker, bottom_marker, top_marker, obstacle_marker = 1, 2, 3, 4, 5
boundaries = [(inlet_marker, lambda x: np.isclose(x[0], 0)), # inflow
              (outlet_marker, lambda x: np.isclose(x[0], L)), # outflow
              (bottom_marker, lambda x: np.isclose(x[1], 0)), # bottom boundary
              (top_marker, lambda x: np.isclose(x[1], H)), # top boundary 
              (obstacle_marker, lambda x: np.isclose(np.sqrt((x[0] - c_x)**2 + (x[1] - c_y)**2),r))] # disk boundary 

from dolfinx.mesh import locate_entities, meshtags
from dolfinx.io import XDMFFile

facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = locate_entities(mesh, fdim, locator) # individuo gli indici delle entities (facets) appartenenti ad un particolare boundary 
    facet_indices.append(facets) # aggiungo gli indici appena trovati a "facet_indices"
    facet_markers.append(np.full_like(facets, marker)) # appendo a facet_marker un array delle stesse dimensioni di quello che conteneva gli 
    # indici appena trovati (ovvero facets) 
    # ma con tutti gli elementi al suo interno uguali al "marker" di riferimento!!! (Quindi, ad esempio, un array di tutti 1)

facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)

sorted_facets = np.argsort(facet_indices) # It returns an array of indices of the same shape as 'facet_indices' that contiene gli indici 
# degli elementi di 'facet_indices' ordinati in maniera crescente 
# Ad esempio, se: facet_indices = np.array([3, 1, 2]) allora: np.argsort(facet_indices) = array([1, 2, 0])


facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]) # Create a MeshTags object 
# facet_indices[sorted_facets] e facet_markers[sorted_facets] saranno 2 array contenenti rispettivamente gli indici degli entities 
# appartenenti ai boundaries (ordinati in maniera crescente)
# ed i rispettivi markers identificativi(1,2,3,4)

mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
with XDMFFile(mesh.comm, "facet_tags_V2.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(facet_tag, mesh.geometry)
############################################## DEFINING THE BOUNDARY CONDITIONS ##############################################


class BoundaryCondition():
    def __init__(self, type, marker, values):
        self._type = type
        if type == "Dirichlet_u":
            facets = facet_tag.find(marker)
            dofs = locate_dofs_topological(V, fdim, facets)
            self._bc = dirichletbc(values, dofs, V)
        elif type == "Dirichlet_u_y":
            facets = facet_tag.find(marker)
            dofs = locate_dofs_topological(V.sub(1), fdim, facets)
            self._bc = dirichletbc(values, dofs, V.sub(1))
        elif type == "Dirichlet_p":
            facets = facet_tag.find(marker)
            dofs = locate_dofs_topological(Q, fdim, facets)
            self._bc = dirichletbc(values, dofs, Q)
        elif type == "Neumann_u":
                self._bc = inner(values, v) * ds(marker)
        elif type == "Neumann_p":
                self._bc = values * q * ds(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc
    @property
    def type(self):
        return self._type

#Dirichelet values 
# velocity 
u_inlet=PETSc.ScalarType((1.0,0.0)) 
u_y_top_bottom=PETSc.ScalarType(0.0) 
u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType) 

# pressure 
p_in=PETSc.ScalarType(8.0) 
p_out=PETSc.ScalarType(0.0) # provare ad usare 1e-6??? In modo tale da non avere una pressione nulla?!

#Define the BC for step 1
boundary_conditions_step1 = [BoundaryCondition("Dirichlet_u", inlet_marker, u_inlet), 
                       BoundaryCondition("Dirichlet_u_y", bottom_marker, u_y_top_bottom), 
                       BoundaryCondition("Dirichlet_u_y", top_marker, u_y_top_bottom), 
                       BoundaryCondition("Dirichlet_u", obstacle_marker, u_nonslip)] 

#Define the BC for step 2
boundary_conditions_step2 = [BoundaryCondition("Dirichlet_p", outlet_marker, p_out)]


####################### DEFINING THE VARIATIONAL FORMULATION #######################
from ufl import inner, nabla_grad, dot, grad, ds, dx, lhs, rhs, div 
from dolfinx.fem import form 
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc


k=fem.Constant(mesh,PETSc.ScalarType(dt)) # time-step
f_zero= fem.Constant(mesh, PETSc.ScalarType((0, 0))) # this should be used only at the very first step!!!

F1=(1/k)*inner(u-u0,v)*dx 
F1+=inner(grad(u0) * u0, v) * dx # inner(dot(u0,nabla_grad(u0)),v)*dx
F1+=nu*inner(grad(u),grad(v))*dx
F1-=inner(f_zero,v)*dx

bcu = []
for condition in boundary_conditions_step1:
    if condition.type == "Dirichlet_u" or condition.type == "Dirichlet_u_y":
        bcu.append(condition.bc)
    elif condition.type == "Neumann_u":
        F1 += condition.bc


#definisco quindi la classica forma: a(u,v)=L(v)
a1=form(lhs(F1))
L1=form(rhs(F1))

A1 = assemble_matrix(a1, bcs=bcu)
A1.assemble()
b1 = create_vector(L1)


#Step 2 
#caluclation of the new pressure (p_n) (by using u*)
a2=inner(grad(p),grad(q))*dx 
L2=-(1/k)*div(u1)*q*dx #non si usa inner o dot (poiché sia div(u1) che q sono degli scalari!)

bcp = []
for condition in boundary_conditions_step2:
    if condition.type == "Dirichlet_p":
        bcp.append(condition.bc)
    elif condition.type == "Neumann_p":
        a2 += condition.bc

a2=form(a2)
L2=form(L2)

A2 = assemble_matrix(a2, bcp)
A2.assemble()
b2 = create_vector(L2)


#Step 3 
#calculation of the new velocity (u_n) (by using u* and p_n)
a3=inner(u,v)*dx
L3=inner(u1,v)*dx - k*inner(grad(p1),v)*dx

a3=form(a3)
L3=form(L3)


A3= assemble_matrix(a3, bcs=bcu) # Applico comunque le BC
# A3= assemble_matrix(a3) 
# Ad A3 (e neanche a b3) non c'é bisogno di applicare le BC poiché queste erano già state applicate nei primi 2 step 
# quindi, implicitamente verranno rispettate anche dal risultato dello step 3  
A3.assemble()
b3= create_vector(L3)


# Solver for step 1
solver1 = PETSc.KSP().create(mesh.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.HYPRE)
pc1.setHYPREType("boomeramg")

# Solver for step 2
solver2 = PETSc.KSP().create(mesh.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.BCGS)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for step 3
solver3 = PETSc.KSP().create(mesh.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)

####################### DEFINING THE VARIABLES FOR THE DRAG AND LIFT #######################
n = -FacetNormal(mesh)  # Normal pointing out of obstacle
dObs = Measure("ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=obstacle_marker)
u_t = inner(as_vector((n[1], -n[0])), u1)
rho = Constant(mesh, PETSc.ScalarType(1))

drag = form(2 / 1 * (nu * rho * inner(grad(u_t), n) * n[1] - p1 * n[0]) * dObs)
lift = form(-2 / 1 * (nu * rho * inner(grad(u_t), n) * n[0] + p1 * n[1]) * dObs)
DRAG=[]
LIFT=[]

####################### VTXWriter to visualize in Paravie and adios4dolfinx to store checkpoints #######################
from pathlib import Path
import adios4dolfinx
from dolfinx.io import VTXWriter

folder = Path("results_V4")
folder_check = Path(folder/"results_check")
folder_para = Path(folder/"results_Para")

check_file_u = Path(folder_check/"velocity_u.bp")
check_file_u_mean = Path(folder_check/"velocity_u_mean.bp")
check_file_f = Path(folder_check/"Reynolds_stress_f.bp")
check_file_p = Path(folder_check/"pressure_p.bp")

para_file_u = Path(folder_para/"velocity_u.bp")
para_file_p = Path(folder_para/"pressure_p.bp")

# Creo effettivamente la directory (se esistono già non verrà segnalato errore)
folder_check.mkdir(exist_ok=True, parents=True) 
folder_para.mkdir(exist_ok=True, parents=True)


if restart==False:
    # scriviamo la mesh all'interno dei file.bp che useremo per ricaricare i valori durante un reset 
    adios4dolfinx.write_mesh(check_file_u, mesh)
    adios4dolfinx.write_mesh(check_file_u_mean, mesh)
    adios4dolfinx.write_mesh(check_file_f, mesh)
    adios4dolfinx.write_mesh(check_file_p, mesh)

    # Apro i file.bp in modalità write
    vtx_u = VTXWriter(mesh.comm, para_file_u, [u1,u_mean,u_fluct,f], engine="BP4")
    vtx_p = VTXWriter(mesh.comm, para_file_p, [p1], engine="BP4") 


if restart:
    in_mesh = adios4dolfinx.read_mesh(check_file_u, MPI.COMM_WORLD)
    W = functionspace(in_mesh, v_cg2)
    U = functionspace(in_mesh, s_cg1)

    u1_in = Function(W)
    u0_in = Function(W)
    u_mean_in = Function(W)
    u_mean_old_in = Function(W)
    f_in = Function(W)
    f_old_in= Function(W)

    p1_in = Function(U)

    adios4dolfinx.read_function(check_file_u, u1_in, time=t_restart, name="u1")
    adios4dolfinx.read_function(check_file_u, u0_in, time=round((t_restart-dt),2), name="u1")
    adios4dolfinx.read_function(check_file_u_mean, u_mean_in, time=t_restart, name="u_mean")
    adios4dolfinx.read_function(check_file_u_mean, u_mean_old_in, time=round((t_restart-dt),2), name="u_mean")
    adios4dolfinx.read_function(check_file_f, f_in, time=t_restart, name="f")
    adios4dolfinx.read_function(check_file_f, f_old_in, time=round((t_restart-dt),2), name="f")

    adios4dolfinx.read_function(check_file_p, p1_in, time=t_restart, name="p1")

    u1.x.array[:]=u1_in.x.array[:]
    u0.x.array[:]=u0_in.x.array[:]
    u_mean.x.array[:]=u_mean_in.x.array[:]
    u_mean_old.x.array[:]=u_mean_old_in.x.array[:]
    f.x.array[:]=f_in.x.array[:]
    f_old.x.array[:]=f_old_in.x.array[:]
    
    p1.x.array[:]=p1_in.x.array[:]

    step=t_restart/dt
    t=t_restart
    print('We are starting from step:', step)


    #creo dei nuovi file.bp (per paraview) che partono direttamente dallo step del restart
    #  (poi dovrò fare un merge di questi file con quelli vecchi per poter visualizzare la soluzione in maniera continuativa
    # tramite: adios2_merge velocity_u.bp velocity_u_new.bp output_merged.bp)
    para_file_u_new = Path(folder_para/f"velocity_u_from_step_{step}_on.bp")
    para_file_p_new = Path(folder_para/f"pressure_p_from_step_{step}_on.bp")
    vtx_u = VTXWriter(mesh.comm, para_file_u_new, [u1,u_mean,u_fluct,f], engine="BP4")
    vtx_p = VTXWriter(mesh.comm, para_file_p_new, [p1], engine="BP4") 
    pass


############################################## SOLVING THE PROBLEM ##############################################
#progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=num_steps)
progress = tqdm(desc="Solving PDE", total=step_final)

from dolfinx.fem import apply_lifting, assemble_scalar
from dolfinx.fem.petsc import LinearProblem

# ricordo che --> step_final=300
#                   tot_save=1
#                   step_init=1
while step <= step_final:  
    progress.update(1)

    # Step 1: Tentative velocity step
    with b1.localForm() as loc_1:
        loc_1.set(0)
    assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u1.x.petsc_vec)
    u1.x.scatter_forward()

    # Step 2: Pressure corrrection step
    with b2.localForm() as loc_2:
        loc_2.set(0)
    assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, p1.x.petsc_vec)
    p1.x.scatter_forward()

    # Step 3: Velocity correction step
    with b3.localForm() as loc_3:
        loc_3.set(0)
    assemble_vector(b3, L3)
    apply_lifting(b3, [a3], [bcu])
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b3, bcu)
    solver3.solve(b3, u1.x.petsc_vec)
    u1.x.scatter_forward()


    if step >= step_init:
        actual_step=step-step_init
        
        u_mean.x.array[:]=(actual_step/(actual_step+1))*u_mean.x.array[:] + 1/(actual_step+1)*u1.x.array[:]
        #p_mean.x.array[:]=(actual_step/(actual_step+1))*p_mean.x.array[:] + 1/(actual_step+1)*p1.x.array[:] # tanto è inutile

        u_fluct.x.array[:]=u1.x.array[:]-u_mean.x.array[:]

        ####### Calcolo del forcing f ########
        # voglio calcolare f, che risulta uguale a: f_prod= -dot(grad(u_fluct), u_fluct)
        expr = dot(grad(u_fluct), u_fluct)
        # Problema variazionale per proiettare l'espressione sul dominio 
        prod = TrialFunction(V)
        v_f = TestFunction(V)
        a_f = inner(prod, v_f) * dx
        L_f = inner(expr, v_f) * dx
        # Risolvi la proiezione L2
        problem_f = LinearProblem(a_f, L_f, bcs=[], petsc_options={"ksp_type": "cg"})
        f_prod = problem_f.solve()  
        f_prod.x.array[:] *= -1

        
        f.x.array[:]=(actual_step/(actual_step+1))*f_old.x.array[:] + 1/(actual_step+1)*f_prod.x.array[:]

        
        #calcolo la differenza (norma) fra le variabili calcolate (mean_flow e forcing) in 2 iterazioni successive
        # mi serve per poter verificare la convergenza  
        err_mean = np.linalg.norm((u_mean_old.x.array[:] - u_mean.x.array[:])) / np.linalg.norm(u_mean_old.x.array[:])
        err_forc = np.linalg.norm((f_old.x.array[:] - f.x.array[:])) / np.linalg.norm(f_old.x.array[:])


        # si potrebbe impostare una tolleranza più tranquilla!! Come ad esempio 1e-5 oppure 1e-8
        if (err_mean < tol and err_forc < tol) or step == step_final:
            #cfl_max = CFLnumber(mesh, u1, dt)
            MPI.COMM_WORLD.Barrier()

            if step == step_final:
                print(f'Last step reached: L2 mean:{err_mean}, L2 forc:{err_forc}')
                #print(f'Last step reached: L2 mean:{err_mean}, L2 forc:{err_forc}, CFLmax: {cfl_max}')
            else:
                print(f'Convergence reached: L2 mean:{err_mean}, L2 forc:{err_forc}')
                #print(f'Convergence reached: L2 mean:{err_mean}, L2 forc:{err_forc}, CFLmax: {cfl_max}')

            # Prima di interromprere il codice poiché siamo andati a convergenza 
            # salvo tutte le variabili di mio interesse 
            adios4dolfinx.write_function(check_file_u, u1, time=t, name="u1")
            adios4dolfinx.write_function(check_file_u_mean, u_mean, time=t, name="u_mean")
            adios4dolfinx.write_function(check_file_f, f, time=t, name="f")
            adios4dolfinx.write_function(check_file_p, p1, time=t, name="p1")

            vtx_u.write(t)
            vtx_p.write(t)

            DRAG.append(assemble_scalar(drag))
            LIFT.append(assemble_scalar(lift))
            # Printo l'evoluzione dei Cd e Cl calcolati 
            print('')
            print(f'''Drag evolution: 
                    First calculation, Drag: {DRAG[0]}
                    Mid-way calculation, Drag: {DRAG[len(DRAG)//2]}
                    Final calculation, Drag: {DRAG[-1]}''')
            
            print(f'''Lift evolution: 
                    First calculation, Lift: {LIFT[0]}
                    Mid-way calculation, Lift: {LIFT[len(LIFT)//2]}
                    Final calculation, Lift: {LIFT[-1]}''')

            MPI.COMM_WORLD.Barrier()
            # break mi farà uscire dal ciclo for originale!! (for i in range(num_steps))
            break

        
        if step % tot_save==0:
            MPI.COMM_WORLD.Barrier()
            #salvo "a prescindere" la soluzione ogni tot steps
            adios4dolfinx.write_function(check_file_u, u1, time=t, name="u1")
            adios4dolfinx.write_function(check_file_u_mean, u_mean, time=t, name="u_mean")
            adios4dolfinx.write_function(check_file_f, f, time=t, name="f")
            adios4dolfinx.write_function(check_file_p, p1, time=t, name="p1")

            vtx_u.write(t)
            vtx_p.write(t)

            DRAG.append(assemble_scalar(drag))
            LIFT.append(assemble_scalar(lift))


    if step==1:
            #salvo i primi valori di Drag e Lift calcolati
            DRAG.append(assemble_scalar(drag))
            LIFT.append(assemble_scalar(lift))

    # Write solutions to file (in realtà non lo dovrei fare sempre!!!)
    # vtx_u.write(t)
    # vtx_p.write(t)


    # Drag and Lift 
    # print(f"drag: {assemble_scalar(drag)}, lift: {assemble_scalar(lift)}")

    ######################## Let's move to the next step!!!!! ########################
    # Update variable with solution form this time step
    u0.x.array[:] = u1.x.array[:] # aggiorno il valore della u0 (che rappresenta u(n-1) nello step 1 dell'IPCS) 
    u_mean_old.x.array[:]=u_mean.x.array[:]

    f_old.x.array[:]=f.x.array[:] # aggiono il valore del forzaggio 
    # p0.x.array[:] = p1.x.array[:] # aggiorno il valore della p0


    t = round((t+dt),2)
    # print(t)
    step+=1
    # # Compute error at current time-step
    # error_L2 = np.sqrt(mesh.comm.allreduce(assemble_scalar(L2_error), op=MPI.SUM))
    # error_max = mesh.comm.allreduce(np.max(u_.x.petsc_vec.array - u_ex.x.petsc_vec.array), op=MPI.MAX)
    # # Print error only every 20th step and at the last step
    # if (i % 20 == 0) or (i == num_steps - 1):
    #     print(f"Time {t:.2f}, L2-error {error_L2:.2e}, Max error {error_max:.2e}")


# Close xmdf file
vtx_u.close()
vtx_p.close()

b1.destroy()
b2.destroy()
b3.destroy()
solver1.destroy()
solver2.destroy()
solver3.destroy()

