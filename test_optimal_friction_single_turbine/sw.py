''' Test description:
 - this test' setup is similar to 'example_single_turbine_friction_vs_power_plot'
 - a single turbine (with a bump function as friction distribution)
 - shallow water model with implicit timestepping scheme to avoid oscillations in the turbine areas 
 - control: turbine friction, initially zero
 - the functional is \int C * f * ||u||**3 where C is a constant
 - in order to avoid the global maximum +oo, the friction coefficient is limited to 0 <= f <= 1.0 
 - the plot in 'example_single_turbine_friction_vs_power_plot' suggestes that the optimal friction coefficient is at about 0.0204
 '''

import sys
import sw_config 
import sw_lib
import numpy
import Memoize
import ipopt 
import IPOptUtils
from functionals import DefaultFunctional, build_turbine_cache
from dolfin import *
from sw_utils import test_initial_condition_adjoint, test_gradient_array, pprint
from turbines import *
from dolfin_adjoint import *

# Global counter variable for vtk output
count = 0

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  numpy.random.seed(21) 
  config = sw_config.DefaultConfiguration(nx=20, ny=10)
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  config.params["finish_time"] = period/2
  config.params["dt"] = config.params["finish_time"]/10
  pprint("Wave period (in h): ", period/60/60)
  config.params["dump_period"] = 1
  config.params["verbose"] = 0
  # We need a implicit scheme to avoid oscillations in the turbine areas.
  config.params["theta"] = 1.0

  # Start at rest state
  config.params["start_time"] = period/4 

  # Turbine settings
  config.params["friction"] = 0.0025
  #config.params["quadratic_friction"] = True
  config.params["turbine_pos"] = [[1500., 500.]]
  # The turbine friction is the control variable 
  config.params["turbine_friction"] = numpy.ones(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 600
  config.params["turbine_y"] = 600

  return config

def initial_control(config):
  # We use the current turbine settings as the intial control
  res = config.params['turbine_friction'].tolist()
  return numpy.array(res)

def j_and_dj(m):
  adj_reset()


  # Change the control variables to the config parameters
  config.params["turbine_friction"] = m[:len(config.params["turbine_friction"])]
  mp = m[len(config.params["turbine_friction"]):]

  set_log_level(30)
  debugging["record_all"] = True

  W=sw_lib.p1dgp2(config.mesh)

  # Set initial conditions
  state = Function(W, name="Current_state")
  state.interpolate(config.get_sin_initial_condition()())

  # Set the control values
  U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
  U = U.collapse() # Recompute the DOF map
  tf = Function(U, name="turbine function") # The turbine function
  tfd = Function(U, name="derivative of the turbine function") # The derivative turbine function

  # Set up the turbine friction field using the provided control variable
  tf.interpolate(Turbines(config.params))

  global count
  count+=1
  sw_lib.save_to_file_scalar(tf, "turbines_t=."+str(count)+".x")

  turbine_cache = build_turbine_cache(config.params, U, turbine_size_scaling=0.5)
  functional = DefaultFunctional(config.params, turbine_cache)

  # Solve the shallow water system
  j, djdm = sw_lib.sw_solve(W, config, state, turbine_field = tf, time_functional=functional)
  J = TimeFunctional(functional.Jt(state), static_variables = [turbine_cache["turbine_field"]], dt=config.params["dt"])
  adj_state = sw_lib.adjoint(state, config.params, J, until=1) # The first annotation is the idendity operator for the turbine field

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

j_and_dj_mem = Memoize.MemoizeMutable(j_and_dj)
def j(m):
  j = j_and_dj_mem(m)[0] * 10**-6
  pprint('Evaluating j(', m.__repr__(), ')=', j)
  return j 

def dj(m):
  dj = j_and_dj_mem(m)[1] * 10**-6
  # Return only the derivatives with respect to the friction
  dj = dj[:len(config.params['turbine_friction'])]
  pprint('Evaluating dj(', m.__repr__(), ')=', dj)
  return dj 

config = default_config()
m0 = initial_control(config)

p = numpy.random.rand(len(m0))
minconv = test_gradient_array(j, dj, m0, seed=0.0001, perturbation_direction=p)
if minconv < 1.98:
  pprint("The gradient taylor remainder test failed.")
  sys.exit(1)

# If this option does not produce any ipopt outputs, delete the ipopt.opt file
g = lambda m: []
dg = lambda m: []

f = IPOptUtils.IPOptFunction()
# Overwrite the functional and gradient function with our implementation
f.objective= j 
f.gradient= dj 

nlp = ipopt.problem(len(m0), 
                    0, 
                    f, 
                    numpy.zeros(len(m0)), 
                    # Set the maximum friction value to 1.0 to enforce the local minimum at 0.122
                    1.0*numpy.ones(len(m0)))
nlp.addOption('mu_strategy', 'adaptive')
nlp.addOption('tol', 1e-4)
nlp.addOption('print_level', 5)
nlp.addOption('check_derivatives_for_naninf', 'yes')
# Add the internal scaling method so that the first derivtive is arount 1.0
#nlp.addOption('nlp_scaling_max_gradient', 2.0)
# A -1.0 objective scaling factor transforms the min problem to a max problem.
nlp.addOption('obj_scaling_factor', -1.0)
# Use an approximate Hessian since we do not have second order information.
nlp.addOption('hessian_approximation', 'limited-memory')
nlp.addOption('max_iter', 13)

m, info = nlp.solve(m0)
pprint(info['status_msg'])
pprint("Solution of the primal variables: m=%s\n" % repr(m))
pprint("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))
pprint("Objective=%s\n" % repr(info['obj_val']))

if info['status'] != 0 or abs(m-0.0204) > 0.0005: 
  pprint("The optimisation algorithm did not find the correct solution: Expected m = 0.0204, but got m = " + str(m) + ".")
  sys.exit(1) 
