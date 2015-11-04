# Adapted from the Firedrake-Fluids project.
from dolfin import *
from dolfin_adjoint import *

# Constants
C1 = 1.44
C2 = 1.92
C3 = -0.33
Cnu = 0.09
sigma_k = 1.0
sigma_eps = 1.3
        
class KEpsilon(object):

    def __init__(self, V, u, dt, mesh):
        self.output = File("eddy_viscosity.pvd")
        self._V = V
        self.mesh = mesh
        self.k = Function(self._V)
        self.k_old = Function(self._V)
        self.eps = Function(self._V)
        self.eps_old = Function(self._V)
        self.eddy_viscosity = Function(self._V)
        self.eddy_viscosity_old = Function(self._V)
        self.u = u

        # Initial conditions
        self.k_old.interpolate(Expression("1.0e-6"))
        self.eps_old.interpolate(Expression("1.0e-6"))
        self.eddy_viscosity_old.interpolate(Expression("1.0e-6"))
        
        self.dt = dt
        self.bcs = []#[DirichletBC(self._V, Expression("0.0"), (1,2,3,4))]
         
        # Create solvers
        k_lhs, k_rhs = self._k_eqn(u, self.dt, self.k_old, self.eps_old, self.eddy_viscosity_old)
        k_problem = LinearVariationalProblem(k_lhs, k_rhs, self.k, bcs=self.bcs)
        self._k_solver = LinearVariationalSolver(k_problem)
        self._k_solver.parameters["linear_solver"] = "lu"

        eps_lhs, eps_rhs = self._eps_eqn(u, self.dt, self.k_old, self.eps_old, self.eddy_viscosity_old)
        eps_problem = LinearVariationalProblem(eps_lhs, eps_rhs, self.eps, bcs=self.bcs)
        self._eps_solver = LinearVariationalSolver(eps_problem)
        self._eps_solver.parameters["linear_solver"] = "lu"

        eddy_viscosity_lhs, eddy_viscosity_rhs = self._eddy_viscosity_eqn(self.k_old, self.eps_old, self.eddy_viscosity_old)
        eddy_viscosity_problem = LinearVariationalProblem(eddy_viscosity_lhs, eddy_viscosity_rhs, self.eddy_viscosity, bcs=self.bcs)
        self._eddy_viscosity_solver = LinearVariationalSolver(eddy_viscosity_problem)
        self._eddy_viscosity_solver.parameters["linear_solver"] = "lu"
        
        return

    def streamline_upwind(self, w, keps, u):
        scaling_factor = 0.5
        cellsize = CellSize(self.mesh)
        magnitude = self.magnitude(u)

        u_nodes = magnitude.vector()
        near_zero = numpy.array([1.0e-9 for i in range(len(u_nodes))])
        u_nodes.set_local(numpy.maximum(u_nodes.array(), near_zero))
        
        grid_pe = self.grid_peclet_number(1.0e-6, magnitude, cellsize)

        # Bound the values for grid_pe below by 1.0e-9 for numerical stability reasons. 
        grid_pe_nodes = grid_pe.vector()
        values = numpy.array([1.0e-9 for i in range(len(grid_pe_nodes))])
        grid_pe_nodes.set_local(numpy.maximum(grid_pe_nodes.array(), values))

        k_bar = ( (1.0/tanh(grid_pe)) - (1.0/grid_pe) ) * cellsize * magnitude
        F = scaling_factor*(k_bar/(magnitude**2))*inner(dot(grad(w), u), dot(grad(keps), u))*dx
        return F

    def magnitude(self, u):
        w = TestFunction(self._V)
        magnitude = TrialFunction(self._V)
        solution = Function(self._V)
        a = w*magnitude*dx
        L = w*sqrt(dot(u, u))*dx
        solve(a == L, solution, bcs=[])
        return solution

    def grid_peclet_number(self, diffusivity, magnitude, cellsize):
        w = TestFunction(self._V)
        grid_pe = TrialFunction(self._V)
        solution = Function(self._V)
        a = w*grid_pe*dx
        L = w*(magnitude*cellsize)/(2.0*diffusivity)*dx
        solve(a == L, solution, bcs=[])
        return solution
   
    def _strain_rate_tensor(self, u):
        S = 0.5*(grad(u) + grad(u).T)
        return S

    def _k_eqn(self, u, dt, k_old, eps_old, eddy_viscosity_old):

        w = TestFunction(self._V)
        k = TrialFunction(self._V)
        
        # Mass for k
        M_k = (1.0/dt)*(inner(w, k) - inner(w, k_old))*dx
      
        # Advection for k
        A_k = inner(w, dot(u, grad(k)))*dx
      
        # Diffusion for k
        D_k = -inner(grad(w), (1e-6 + eddy_viscosity_old/sigma_k)*grad(k))*dx

        S = self._strain_rate_tensor(u)
        second_invariant = 0.0
        dim = len(u)
        for i in range(0, dim):
           for j in range(0, dim):
              second_invariant += 2.0*(S[i,j]**2)
              
        # Production for k
        P_k = inner(w, eddy_viscosity_old*second_invariant)*dx
      
        # Absorption for k
        ABS_k = -inner(w, eps_old)*dx

        # The full weak form of the equation for k
        F_k = M_k + A_k - D_k - P_k - ABS_k
        
        F_k += self.streamline_upwind(w, k, u)

        return lhs(F_k), rhs(F_k)
        
    def _eps_eqn(self, u, dt, k_old, eps_old, eddy_viscosity_old):

        w = TestFunction(self._V)
        eps = TrialFunction(self._V)
        
        # Mass for epsilon
        M_eps = (1.0/dt)*(inner(w, eps) - inner(w, eps_old))*dx
      
        # Advection for epsilon
        A_eps = inner(w, dot(u, grad(eps)))*dx
      
        # Diffusion for epsilon
        D_eps = -inner(grad(w), (1e-6 + eddy_viscosity_old/sigma_eps)*grad(eps))*dx

        S = self._strain_rate_tensor(u)
        second_invariant = 0.0
        dim = len(u)
        for i in range(0, dim):
           for j in range(0, dim):
              second_invariant += 2.0*(S[i,j]**2)      
      
        # Production for epsilon
        P_eps = C1*(eps/k_old)*inner(w, eddy_viscosity_old*second_invariant)*dx
      
        # Absorption for epsilon
        ABS_eps = -C2*(eps_old**2/k_old)*inner(w, eps)*dx

        # The full weak form of the equation for epsilon
        F_eps = M_eps + A_eps - D_eps - P_eps - ABS_eps
        
        F_eps += self.streamline_upwind(w, eps, u)
        
        return lhs(F_eps), rhs(F_eps)
     
    def _eddy_viscosity_eqn(self, k_old, eps_old, eddy_viscosity_old):
        # Eddy viscosity
        
        w = TestFunction(self._V)
        eddy_viscosity = TrialFunction(self._V)
        
        F_eddy_viscosity = inner(w, eddy_viscosity)*dx - inner(w, Cnu*((k_old**2)/eps_old))*dx

        return lhs(F_eddy_viscosity), rhs(F_eddy_viscosity)

    def solve(self, u):
        """ Update the eddy viscosity solution for the current velocity.

        :returns: The eddy viscosity.
        """
        
#        self.u.assign(u)

        # k solve
        k_lhs, k_rhs = self._k_eqn(u, self.dt, self.k_old, self.eps_old, self.eddy_viscosity_old)
        solve(k_lhs == k_rhs, self.k, bcs=[])
        self.k_old.assign(self.k)

        nodes = self.k_old.vector()
        near_zero = numpy.array([1.0e-6 for i in range(len(nodes))])
        nodes.set_local(numpy.maximum(nodes.array(), near_zero))        
        print "k: ", nodes.array(), max(nodes)        

        #v = self.k_old.vector()
        #values = []
        #node_min, node_max = v.local_range()
        #for i in range(0, node_max-node_min):
        #    if(v[i] < 1.0e-9):
        #        values.append(1.0e-9)
        #    else:
        #        values.append(v[i][0])
        #v.set_local(numpy.array(values))
        #v.apply("insert")

        # eps solve
        eps_lhs, eps_rhs = self._eps_eqn(u, self.dt, self.k_old, self.eps_old, self.eddy_viscosity_old)
        solve(eps_lhs == eps_rhs, self.eps, bcs=[])
        self.eps_old.assign(self.eps)

        nodes = self.eps_old.vector()
        near_zero = numpy.array([1.0e-6 for i in range(len(nodes))])
        nodes.set_local(numpy.maximum(nodes.array(), near_zero))
        print "eps: ", nodes.array(), max(nodes)

        #v = self.eps.vector()
        #values = []
        #node_min, node_max = v.local_range()
        #for i in range(0, node_max-node_min):
        #    if(v[i] < 1.0e-9):
        #        values.append(1.0e-9)
        #    else:
        #        values.append(v[i][0])
        #v.set_local(numpy.array(values))
        #v.apply("insert")
        #self.eps_old.assign(self.eps)

        eddy_viscosity_lhs, eddy_viscosity_rhs = self._eddy_viscosity_eqn(self.k_old, self.eps_old, self.eddy_viscosity_old)
        solve(eddy_viscosity_lhs == eddy_viscosity_rhs, self.eddy_viscosity, bcs=[])
        self.eddy_viscosity_old.assign(self.eddy_viscosity)

        nodes = self.eddy_viscosity_old.vector()
        near_zero = numpy.array([1.0e-6 for i in range(len(nodes))])
        nodes.set_local(numpy.maximum(nodes.array(), near_zero))

        print "eddy viscosity: ", nodes.array(), max(nodes)

        #near_zero = numpy.array([1.0e-9 for i in range(len(nodes))])
        #nodes.set_local(numpy.maximum(nodes.array(), near_zero))
        #v = self.eddy_viscosity.vector()
        #values = []
        #node_min, node_max = v.local_range()
        #for i in range(0, node_max-node_min):
        #    if(v[i] < 1.0e-9):
        #        values.append(1.0e-9)
        #    else:
        #        values.append(v[i][0])
        #print numpy.array(values)
        #v.set_local(numpy.array(values))
        #v.apply("insert")
        #self.eddy_viscosity_old.assign(self.eddy_viscosity)

        self.output << self.eddy_viscosity_old
                
        return self.eddy_viscosity_old
