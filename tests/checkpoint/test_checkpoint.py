''' Test description:
 - single turbine (with constant friction distribution) whose size exceeds the size of the domain
 - constant velocity profile with an initial x-velocity of 2.
 - control: turbine friction
 - the mini model will compute a x-velocity of 2/(f + 1) wher ef is the turbine friction.
 - the functional is \int C * f * ||u||**3 where C is a constant
 - hence we maximise C * f * ( 2/(f + 1) )**3, f > 0 which has the solution f = 0.5

 Note: The solution is known only because we use a constant turbine friction distribution. 
       However this turbine model is not differentiable at its boundary, and this is why
       the turbine size has to exceed the domain.
 '''

from opentidalfarm import *

class TestCheckpoint(object):

    def default_config(self):
        config = configuration.DefaultConfiguration(nx=20, ny=10, finite_element=finite_elements.p1dgp2)
        config.set_domain(domains.RectangularDomain(3000, 1000, 20, 10))
        config.params["verbose"] = 0
      
        # dt is used in the functional only, so we set it here to 1.0
        config.params["dt"] = 1.0
        # Turbine settings
        config.params["turbine_pos"] = [[500., 500.]]
        # The turbine friction is the control variable 
        config.params["turbine_friction"] = 12.0*numpy.random.rand(len(config.params["turbine_pos"]))
        config.params["turbine_x"] = 8000
        config.params["turbine_y"] = 8000
        config.params['controls'] = ['turbine_friction']
        config.params["functional_final_time_only"] = True
        config.params["dump_period"] = -1
        config.params["output_turbine_power"] = False
      
        k = pi/config.domain.basin_x
        config.params['initial_condition'] = SinusoidalInitialCondition(config, 2.0, k, config.params['depth'])
      
        return config

    def test_speedup_is_larger_than_ten(self):
        config = self.default_config()
        config.params["save_checkpoints"] = True
        config.info()
        friction0 = config.params["turbine_friction"]

        rf = ReducedFunctional(config, forward_model = mini_model.mini_model_solve)
        bounds = [0, 100]
        
        # First optimize without checkpoints
        maxiter = 2
        t = Timer("First optimisation")
        m = maximize(rf, bounds=bounds, method="SLSQP", scale=1e-3, options={'maxiter': maxiter}) 
        t1 = t.stop()

        # Then optimize again
        config.params["turbine_friction"] = friction0
        maxiter = 2
        t = Timer("First optimisation")
        m = maximize(rf, bounds=bounds, method="SLSQP", scale=1e-3, options={'maxiter': maxiter}) 
        t2 = t.stop()

        # Check that speedup is significant
        assert t1/t2 > 10