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

    def __init__(self, V, u, dt):
        self._V = V
        self.k = Function(self._V)
        self.k_old = Function(self._V)
        self.eps = Function(self._V)
        self.eps_old = Function(self._V)
        self.eddy_viscosity = Function(self._V)
        self.eddy_viscosity_old = Function(self._V)
        self.u = u

        # Initial conditions
        self.k_old.interpolate(Expression(1.0e-9))
        self.eps_old.interpolate(Expression(1.0e-9))
        self.eddy_viscosity_old.interpolate(Expression(1.0e-9))  
        
        self.dt = dt
        self.bcs = [DirichletBC(self._V, Expression("0.0"), (1,2,3,4))]
         
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

    def _k_eqn(self, u, dt, k_old, eps_old, eddy_viscosity_old):

        w = TestFunction(self._V)
        k = TrialFunction(self._V)
        u = self.u
        
        # Mass for k
        M_k = (1.0/dt)*(inner(w, k) - inner(w, k_old))*dx
      
        # Advection for k
        A_k = inner(w, div(u*k))*dx
      
        # Diffusion for k
        D_k = -inner(grad(w), (1e-6 + eddy_viscosity_old/sigma_k)*grad(k))*dx
      
        # Production for k
        P_k = inner(w, eddy_viscosity_old*inner(grad(u), grad(u)))*dx
      
        # Absorption for k
        ABS_k = -inner(w_k, eps_old)*dx

        # The full weak form of the equation for k
        F_k = M_k + A_k - D_k - P_k - ABS_k

        return lhs(F_k), rhs(F_k)
        
     def _eps_eqn(self, u, dt, k_old, eps_old, eddy_viscosity_old):

        w = TestFunction(self._V)
        eps = TrialFunction(self._V)
        u = self.u
        
        # Mass for epsilon
        M_eps = (1.0/dt)*(inner(w, eps) - inner(w, eps_old))*dx
      
        # Advection for epsilon
        A_eps = inner(w, div(u*eps))*dx
      
        # Diffusion for epsilon
        D_eps = -inner(grad(w), (1e-6 + eddy_viscosity_old/sigma_eps)*grad(eps))*dx
      
        # Production for epsilon
        P_eps = C1*(eps/k_old)*inner(w, eddy_viscosity_old*inner(grad(u), grad(u)))*dx
      
        # Absorption for epsilon
        ABS_eps = -C2*(eps_old**2/k_old)*inner(w, eps)*dx

        # The full weak form of the equation for epsilon
        F_eps = M_eps + A_eps - D_eps - P_eps - ABS_eps
      
        return lhs(F_eps), rhs(F_eps)
     
     def _eddy_viscosity_eqn(self, k_old, eps_old, eddy_viscosity_old):
        # Eddy viscosity
        
        w = TestFunction(self._V)
        eddy_viscosity = TrialFunction(self._V)
        
        F_eddy_viscosity = inner(w, eddy_viscosity)*dx - inner(eddy_viscosity, Cnu*((k_old**2)/eps_old))*dx

        return lhs(F_eddy_viscosity), rhs(F_eddy_viscosity)

    def solve(self, u):
        """ Update the eddy viscosity solution for the current velocity.

        :returns: The eddy viscosity.
        """
        
        self.u.assign(u)
        
        self._k_solver.solve()
        self.k_old.assign(self.k)

        nodes = self.k_old.vector()
        near_zero = numpy.array([1.0e-9 for i in range(len(nodes))])
        nodes.set_local(numpy.maximum(nodes.array(), near_zero))
        
        self._eps_solver.solve()
        self.eps_old.assign(self.eps)
        
        nodes = self.eps_old.vector()
        near_zero = numpy.array([1.0e-9 for i in range(len(nodes))])
        nodes.set_local(numpy.maximum(nodes.array(), near_zero))
        
        self._eddy_viscosity_solver.solve()
        self.eddy_viscosity_old.assign(self.eddy_viscosity)
        
        nodes = self.eddy_viscosity_old.vector()
        near_zero = numpy.array([1.0e-9 for i in range(len(nodes))])
        nodes.set_local(numpy.maximum(nodes.array(), near_zero))
        
        return self.eddy_viscosity
