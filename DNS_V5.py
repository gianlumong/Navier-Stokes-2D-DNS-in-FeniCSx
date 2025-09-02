import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element
from dolfinx.fem import assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector,create_vector
from dolfinx.io import VTXWriter, gmshio
from tqdm.autonotebook import tqdm

from xCFL_number import CFLnumber


assert MPI.COMM_WORLD.size == 1, "This example should only be run with 1 MPI process"
############################################## DEFINING THE MESH ##############################################
import gmsh
gmsh.initialize()  # Initialize the Gmsh environment! This must be done before any other Gmsh call!!!!!
L = 27.0 # Original L=2.2
H = 10.0 # Original H=0.41
c_x = 9.0 # Original c_x=0.2
c_y = 5.0 # Original c_y=0.2
r = 0.5 # Original r=0.05

gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:   # in order to generate the mesh only on the processor with rank==0
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)


# subtract the disk from the rectangle (to avoid meshing the intern of the disk)
if mesh_comm.rank == model_rank:
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
    gmsh.model.occ.synchronize() # Let's synchroinize the mesh on all processor


#To get GMSH to mesh the fluid, we add a physical volume marker
fluid_marker = 1
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim=gdim) # volumes-->[(2,1)]
    assert (len(volumes) == 1) # will raise a warning if we will have more than 1 2D geometric entities 

    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker) #Definisco quindi il gruppo fisico
    #volumes[0][0] <---è uguale a 2 (ed indica la dimensione dell'entità considerata)
    #[volumes[0][1]] <--è un array con unico elemento 1 (che sarà il tags identificativo dell'entità)

    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid") # Assign a name to the physical group


#defining the markers
inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
inflow, outflow, walls, obstacle = [], [], [], []

if mesh_comm.rank == model_rank:
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
     #boundaries will therefore be a list of tuples (dim, tag) related to each boundary of the volumes
    
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        #boundary[0] --> will be the dimension of the boundary under consideration
        #boundary[1] --> will be the tag of the boundary under consideration
        #getCenterOfMass --> therefore returns the coordinates (x,y,z) of the "center" of the boundary under consideration
        if np.allclose(center_of_mass, [0, H / 2, 0]):
            inflow.append(boundary[1]) #so I assign to "inflow" (which I had previously created) the tag corresponding to the related boundary!
        elif np.allclose(center_of_mass, [L, H / 2, 0]):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(center_of_mass, [L / 2, 0, 0]):
            walls.append(boundary[1])
        else:
            obstacle.append(boundary[1])
    #therefore, thanks to this for loop, I was able to identify the various data of the different boundaries of the 2D domain (volumes)

    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")

    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")

    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")


# Create distance field from obstacle.
res_min = r / 3   # r=0.05 is the radius of the disk
if mesh_comm.rank == model_rank:
    distance_field = gmsh.model.mesh.field.add("Distance") 
    
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
    threshold_field = gmsh.model.mesh.field.add("Threshold")

    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H) # H=0.41
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H) # H=0.41
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

if mesh_comm.rank == model_rank:
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")


############################################## IMPORTING THE MESH ##############################################
mesh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
ft.name = "Facet markers"

from  useful_functions import save_mesh
# save_mesh(mesh) # Plot or save 



############################################## DEFINING THE MOST IMPORTANT VARIABLES ##############################################
dt=0.01  # In the Poiseulle flow case I had: dt=0.000625
t=0

Re=80 # Or 150
nu=1/Re

restart=False
if restart:
    last_step_stop=100 # set it equal to the number of steps done up to the checkpoint! (important only if restart=True)
    t_restart=round((dt*last_step_stop),2)
    print('WE ARE RESTARTING!!!')
    print('')
    print('t_restart:', t_restart)
 
tol=1e-8 # I can use as tol one of those values: 1e-5; 1e-8 or even 1e-15?
tot_save= 5000 # we save every -tot_save- steps

step=0
step_init = 2000 # step at which the calculation of u_mean, p_mean and f starts 
save_first_n_steps = 150 # Indicates the first n initial steps for which we will save the solution
step_final = 100000000 # 100 million

# Create the file.txt that will contain the outputs that would normally be written to the terminal (needed especially
# if I run the code remotely)
terminal_output_name=f'terminal_output_Re{Re}.txt' 
with open(terminal_output_name, "w") as txt:
    txt.write("This is the file that will contain the various outputs of the terminal.\n") #\n serves to go to the next line
    txt.write("Therefore, it will contain all the useful information regarding how the code execution went.\n\n") 
    txt.write(f'''Important variables related to execution: 
            Re: {Re}
            dt: {dt}
            tol: {tol}
            \n\n''')


############################################## DEFINING VECTOR SPACES AND USEFUL FUNCTIONS ##############################################
from ufl import TestFunction, TrialFunction
from dolfinx.fem import Function, functionspace

#We want to use a continuous piecewise quadratic elements for the velocity and continuous piecewise linear elements for the pressure.
v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)


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


# defining the functions that I will use for the calculation of mean flow and forcing
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



############################################## DEFINING THE BOUNDARY CONDITIONS ##############################################
from dolfinx.fem import dirichletbc, locate_dofs_topological
from ufl import div, dx, inner, lhs, rhs

#Dirichelet values:
#velocity
u_inlet=PETSc.ScalarType((1.0,0.0)) 
u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType) 
u_y_top_bottom=PETSc.ScalarType(0.0) 

#pressure 
p_in=PETSc.ScalarType(8.0) 
p_out=PETSc.ScalarType(0)  #we can try to use 1e-6??? In order not to have zero pressure?!

fdim = mesh.topology.dim - 1

# BC for the velocity 
bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(inlet_marker)),V)
bcu_walls = dirichletbc(u_y_top_bottom, locate_dofs_topological(V.sub(1), fdim, ft.find(wall_marker)), V.sub(1))
bcu_obstacle = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V)
bcu = [bcu_inflow, bcu_obstacle, bcu_walls]

# BC for the pressure
bcp_outlet = dirichletbc(p_out, locate_dofs_topological(Q, fdim, ft.find(outlet_marker)), Q)
bcp = [bcp_outlet] 



############################################## DEFINING THE VARIATIONAL FORMULATION ##############################################
from ufl import inner, dot, grad, ds, dx, lhs, rhs, div 
from dolfinx.fem import form 
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.fem import Constant


k=Constant(mesh,PETSc.ScalarType(dt)) # time-step
f_zero= Constant(mesh, PETSc.ScalarType((0, 0))) # this should be used only at the very first step!!!

F1=(1/k)*inner(u-u0,v)*dx 
F1+=inner(grad(u0) * u0, v) * dx # inner(dot(u0,nabla_grad(u0)),v)*dx
F1+=nu*inner(grad(u),grad(v))*dx
F1-=inner(f_zero,v)*dx

# Let's defining the classical form: a(u,v)=L(v)
a1=form(lhs(F1))
L1=form(rhs(F1))

A1 = assemble_matrix(a1, bcs=bcu)
A1.assemble()
b1 = create_vector(L1)


#Step 2 
#caluclation of the new pressure (p_n) (by using u*)
a2=inner(grad(p),grad(q))*dx 
L2=-(1/k)*div(u1)*q*dx #We are not using inner or dot (since both div(u1) and q are scalars!)

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


A3= assemble_matrix(a3, bcs=bcu) # Apply the boundary conditions
# In reality you shuldn't need to apply the boundary conditions to A3 (and b3) since they were already applied in the first 2 steps
# therefore, they will be implicitly respected by the result of step 3
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
from ufl import as_vector
from ufl import FacetNormal, Measure

n = -FacetNormal(mesh)  # Normal pointing out of obstacle
dObs = Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=obstacle_marker)
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

folder = Path(f"results_V5_Re{Re}")
folder_check = Path(folder/"results_check")
folder_para = Path(folder/"results_Para")

check_file_u = Path(folder_check/"velocity_u.bp")
check_file_u_mean = Path(folder_check/"velocity_u_mean.bp")
check_file_f = Path(folder_check/"Reynolds_stress_f.bp")
check_file_p = Path(folder_check/"pressure_p.bp")

para_file_u = Path(folder_para/"velocity_and_forcing.bp")
para_file_p = Path(folder_para/"pressure_p.bp")

# Creating the directory (if it already exists, no error will be raised)
folder_check.mkdir(exist_ok=True, parents=True) 
folder_para.mkdir(exist_ok=True, parents=True)


if restart==False:
    # We write the mesh into the .bp files that we will use to reload the values during a reset
    adios4dolfinx.write_mesh(check_file_u, mesh)
    adios4dolfinx.write_mesh(check_file_u_mean, mesh)
    adios4dolfinx.write_mesh(check_file_f, mesh)
    adios4dolfinx.write_mesh(check_file_p, mesh)

    # We open the .bp files in write mode
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


    # Creating the new .bp files (for Paraview) that start directly from the restart step
    #  (then I will have to merge these files with the old ones to be able to visualize the solution continuously
    # via: adios2_merge velocity_u.bp velocity_u_new.bp output_merged.bp)
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

# Remember that --> step_final=300
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
        #p_mean.x.array[:]=(actual_step/(actual_step+1))*p_mean.x.array[:] + 1/(actual_step+1)*p1.x.array[:] # calculating p is kind of useless

        u_fluct.x.array[:]=u1.x.array[:]-u_mean.x.array[:]

        ####### Calcolating the forcing f ########
        # I want to calculate f, which is equal to: f_prod= -dot(grad(u_fluct), u_fluct)
        expr = dot(grad(u_fluct), u_fluct)
        # Variational problem to project the expression onto the domain
        prod = TrialFunction(V)
        v_f = TestFunction(V)
        a_f = inner(prod, v_f) * dx
        L_f = inner(expr, v_f) * dx
        # Solve the L2 projection
        problem_f = LinearProblem(a_f, L_f, bcs=[], petsc_options={"ksp_type": "cg"})
        f_prod = problem_f.solve()  
        f_prod.x.array[:] *= -1

        
        f.x.array[:]=(actual_step/(actual_step+1))*f_old.x.array[:] + 1/(actual_step+1)*f_prod.x.array[:]

        
        # Calculate the difference (norm) between the calculated variables (mean_flow and forcing) in 2 successive iterations
        # This is needed to verify convergence
        err_mean = np.linalg.norm((u_mean_old.x.array[:] - u_mean.x.array[:])) / np.linalg.norm(u_mean_old.x.array[:])
        err_forc = np.linalg.norm((f_old.x.array[:] - f.x.array[:])) / np.linalg.norm(f_old.x.array[:])


        # you should set a more relaxed tolerance!! Such as 1e-5 or 1e-8
        if (err_mean < tol and err_forc < tol) or step == step_final:
            #cfl_max = CFLnumber(mesh, u1, dt)
            MPI.COMM_WORLD.Barrier()

            if step == step_final:
                print(f'Last step reached: L2 mean:{err_mean}, L2 forc:{err_forc}')
                with open(terminal_output_name, "a") as txt:
                    txt.write(f'Last step reached: L2 mean:{err_mean}, L2 forc:{err_forc} \n\n')
                #print(f'Last step reached: L2 mean:{err_mean}, L2 forc:{err_forc}, CFLmax: {cfl_max}')
            else:
                print(f'Convergence reached: L2 mean:{err_mean}, L2 forc:{err_forc}')
                with open(terminal_output_name, "a") as txt:
                    txt.write(f'Convergence reached: L2 mean:{err_mean}, L2 forc:{err_forc} \n\n')
                #print(f'Convergence reached: L2 mean:{err_mean}, L2 forc:{err_forc}, CFLmax: {cfl_max}')

            # Before stopping the code since we have reached convergence
            # I save all the variables of interest
            adios4dolfinx.write_function(check_file_u, u1, time=t, name="u1")
            adios4dolfinx.write_function(check_file_u_mean, u_mean, time=t, name="u_mean")
            adios4dolfinx.write_function(check_file_f, f, time=t, name="f")
            adios4dolfinx.write_function(check_file_p, p1, time=t, name="p1")

            vtx_u.write(t)
            vtx_p.write(t)

            DRAG.append(assemble_scalar(drag))
            LIFT.append(assemble_scalar(lift))
            # Print the evolution of the calculated Cd and Cl
            print('')
            print(f'''Drag evolution: 
                    First calculation, Drag: {DRAG[0]}
                    Mid-way calculation, Drag: {DRAG[len(DRAG)//2]}
                    Final calculation, Drag: {DRAG[-1]}''')
            
            print(f'''Lift evolution: 
                    First calculation, Lift: {LIFT[0]}
                    Mid-way calculation, Lift: {LIFT[len(LIFT)//2]}
                    Final calculation, Lift: {LIFT[-1]}''')
            
            with open(terminal_output_name, "a") as txt:
                txt.write(f'''Drag evolution: 
                    First calculation, Drag: {DRAG[0]}
                    Mid-way calculation, Drag: {DRAG[len(DRAG)//2]}
                    Final calculation, Drag: {DRAG[-1]} \n''')
                txt.write(f'''Lift evolution: 
                    First calculation, Lift: {LIFT[0]}
                    Mid-way calculation, Lift: {LIFT[len(LIFT)//2]}
                    Final calculation, Lift: {LIFT[-1]}''')

            MPI.COMM_WORLD.Barrier()
            # break will exit the original for loop!! (for i in range(num_steps))
            break

        
        if (step % tot_save==0) or step<=save_first_n_steps:
            MPI.COMM_WORLD.Barrier()
            # Save the solution every tot steps
            adios4dolfinx.write_function(check_file_u, u1, time=t, name="u1")
            adios4dolfinx.write_function(check_file_u_mean, u_mean, time=t, name="u_mean")
            adios4dolfinx.write_function(check_file_f, f, time=t, name="f")
            adios4dolfinx.write_function(check_file_p, p1, time=t, name="p1")

            vtx_u.write(t)
            vtx_p.write(t)

            DRAG.append(assemble_scalar(drag))
            LIFT.append(assemble_scalar(lift))


    if step==1:
            # Save the first calculated values of Drag and Lift
            DRAG.append(assemble_scalar(drag))
            LIFT.append(assemble_scalar(lift))


    ######################## Let's move to the next step!!!!! ########################
    # Update variable with solution form this time step
    u0.x.array[:] = u1.x.array[:] # Update the value of u0 (which represents u(n-1) in the first step of IPCS)
    u_mean_old.x.array[:]=u_mean.x.array[:]

    f_old.x.array[:]=f.x.array[:] # Update the value of the forcing
    # p0.x.array[:] = p1.x.array[:] # Update the value of p0


    t = round((t+dt),2)
    # print(t)
    step+=1


# Close xmdf file
vtx_u.close()
vtx_p.close()

b1.destroy()
b2.destroy()
b3.destroy()
solver1.destroy()
solver2.destroy()
solver3.destroy()

