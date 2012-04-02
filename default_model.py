import numpy
import memoize
from functionals import DefaultFunctional, build_turbine_cache
from dolfin import *
from turbines import *
from dolfin_adjoint import *

class DefaultModel:
    def __init__(self, config):
        # Hide the configuration since changes would break the memorize algorithm. 
        self.__config__ = config

        def j_and_dj(m):
          adj_reset()

          # Change the control variables to the config parameters
          config.params["turbine_friction"] = m[:len(config.params["turbine_friction"])]
          mp = m[len(config.params["turbine_friction"]):]
          config.params["turbine_pos"] = numpy.reshape(mp, (-1, 2))

          set_log_level(30)
          debugging["record_all"] = True

          W = config.params['element_type'](config.mesh)

          # Get initial conditions
          state = Function(W, name="Current_state")
          state.interpolate(config.get_sin_initial_condition()())

          # Set the control values
          U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
          U = U.collapse() # Recompute the DOF map
          tf = Function(U, name = "friction") 
          tfd = Function(U, name = "friction_derivative") 

          # Set up the turbine friction field using the provided control variable
          tf.interpolate(Turbines(config.params))

          # Scale the turbine size by 0.5 for the functional definition. This is used for obtaining 
          # a physical power curve.
          turbine_cache = build_turbine_cache(config.params, U, turbine_size_scaling=0.5)
          functional = DefaultFunctional(config.params, turbine_cache)

          # Solve the shallow water system
          j, djdm = sw_lib.sw_solve(W, config, state, time_functional=functional, turbine_field = tf)
          J = TimeFunctional(functional.Jt(state), static_variables = [turbine_cache["turbine_field"]], dt = config.params["dt"])
          adj_state = sw_lib.adjoint(state, config.params, J, until={"name": "friction", "timestep": 0, "iteration": 0})

          # Let J be the functional, m the parameter and u the solution of the PDE equation F(u) = 0.
          # Then we have 
          # dJ/dm = (\partial J)/(\partial u) * (d u) / d m + \partial J / \partial m
          #               = adj_state * \partial F / \partial u + \partial J / \partial m
          # In this particular case m = turbine_friction, J = \sum_t(ft) 
          dj = [] 
          v = adj_state.vector()
          # Compute the derivatives with respect to the turbine friction
          for n in range(len(config.params["turbine_friction"])):
            tfd.interpolate(Turbines(config.params, derivative_index_selector=n, derivative_var_selector='turbine_friction'))
            dj.append( v.inner(tfd.vector()) )

          # Compute the derivatives with respect to the turbine position
          for n in range(len(config.params["turbine_pos"])):
            for var in ('turbine_pos_x', 'turbine_pos_y'):
              tfd.interpolate(Turbines(config.params, derivative_index_selector=n, derivative_var_selector=var))
              dj.append( v.inner(tfd.vector()) )
          dj = numpy.array(dj)  
          
          # Now add the \partial J / \partial m term
          dj += djdm

          return j, dj 

        self.j_and_dj_mem = Memoize.MemoizeMutable(j_and_dj)

    def j(self, m):
      return self.j_and_dj_mem(m)[0]

    def dj(self, m):
      return self.j_and_dj_mem(m)[1]

    def initial_control(self):
        # We use the current turbine settings as the intial control
        self.__config__ = config
        res = config.params['turbine_friction'].tolist()
        res += numpy.reshape(config.params['turbine_pos'], -1).tolist()
        return numpy.array(res)
