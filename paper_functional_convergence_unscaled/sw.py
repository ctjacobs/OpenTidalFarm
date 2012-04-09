''' This tests checks that the power output of a single turbine in a periodic domain is independent of its position ''' 
import sys
import configuration 
import numpy
from dirichlet_bc import DirichletBCSet
import IPOptUtils
import ipopt
from helpers import test_gradient_array
from reduced_functional import ReducedFunctional
from dolfin import *
set_log_level(ERROR)

if len(sys.argv) not in [2, 3]:
    print "Missing command line argument: y position of the turbine"
    print "Usage: sw.py [--fine] y_position"
    sys.exit(1)

turbine_y_pos = float(sys.argv[-1])

nx = 100
ny = 33
if len(sys.argv) == 3 and sys.argv[1] == '--fine':
    nx = 2*nx
    ny = 2*ny

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  numpy.random.seed(21) 
  config = configuration.DefaultConfiguration(nx = nx, ny = ny)
  print "The vertical mesh element size is %f." % (config.params["basin_y"]/ny) 
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  info("Wave period (in h): %f" % (period/60/60) )
  config.params["dump_period"] = 10000
  config.params["verbose"] = 0

  # Start at rest state
  config.params["start_time"] = period/4
  config.params["dt"] = period/50
  config.params["finish_time"] = 3.*period/4 
  config.params["theta"] = 0.6
  config.params["include_advection"] = True 
  config.params["include_diffusion"] = True 
  config.params["diffusion_coef"] = 2.0
  config.params["newton_solver"] = True 
  config.params["picard_iterations"] = 20
  config.params["linear_solver"] = "default"
  config.params["preconditioner"] = "default"
  config.params["controls"] = ["turbine_pos"]
  info_green("Approximate CFL number (assuming a velocity of 2): " +str(2*config.params["dt"]/config.mesh.hmin())) 

  config.params["bctype"] = "strong_dirichlet"
  config.params['functional_turbine_scaling'] = 0.5
  bc = DirichletBCSet(config)
  bc.add_analytic_u(config.left)
  bc.add_analytic_u(config.right)
  bc.add_periodic_sides()
  config.params["strong_bc"] = bc

  dolfin.parameters['form_compiler']['cpp_optimize'] = True
  dolfin.parameters['form_compiler']['cpp_optimize_flags'] = '-O3'

  # Turbine settings
  config.params["quadratic_friction"] = True
  config.params["friction"] = 0.0025
  # The turbine position is the control variable 
  config.params["turbine_pos"] = [[1500., turbine_y_pos]] 

  info_blue("Deployed " + str(len(config.params["turbine_pos"])) + " turbines.")
  # Choosing a friction coefficient of > 0.02 ensures that overlapping turbines will lead to
  # less power output.
  config.params["turbine_friction"] = 0.2*numpy.ones(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 200
  config.params["turbine_y"] = 200

  return config

config = default_config()
model = ReducedFunctional(config, scaling_factor = 10**-4, plot = True)
m0 = model.initial_control()
print "Functional value for m0 = ", m0, ": ", model.j(m0)
print "Derivative value for m0 = ", m0, ": ", model.dj(m0)
