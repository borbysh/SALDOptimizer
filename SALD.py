#Stochastic Optimization by Simulated Anealing and Langevin Dynamics
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
from tensorflow.python.eager import context

class SALDOptimizer(optimizer.Optimizer):
  def __init__(self, learning_rate=0.1, beta=0.99995, fric=100,  T0=1e-6,
               use_locking=False, name="SALDOptimizer"):
#beta is a Simulated Annealing parameter. In the range from 0.9999 to 1
#if beta=1 the temperature is constant and Simulated Annealing is not applied
#T0 is the temperature. Default T0=1e-6
#fric is a parameter of the Langevin Dynamics. Default fric=100
#If fric=0 the algorithm is equivalent to Velocity-Verlet algorithm
    super(SALDOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta = beta
    self._T0 = T0
    self._fric = fric
    self._lr_t = None
    self._beta_t = None
    self._T0_t = None
    self._fric_t = None

  def _get_beta_accumulators(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("beta_power", graph=graph))

  def _prepare(self):
    lr = self._call_if_callable(self._lr)
    beta = self._call_if_callable(self._beta)
    T0 = self._call_if_callable(self._T0)
    fric=self._call_if_callable(self._fric)
    self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
    self._beta_t = ops.convert_to_tensor(beta, name="beta")
    self._T0_t = ops.convert_to_tensor(T0, name="T0")
    self._fric_t = ops.convert_to_tensor(fric, name="fric")

  def _create_slots(self, var_list):
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(initial_value=self._beta,
                                   name="beta_power",
                                   colocate_with=first_var)
    for v in var_list:
      self._zeros_slot(v, "v", self._name)
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "m2", self._name)
      self._zeros_slot(v, "g", self._name)

  def _apply_dense(self, grad, var):
#  Vanden-Eijnden, Eric, and Giovanni Ciccotti.
# "Second-order integrators for Langevin equations with holonomic constraints."
#  Chemical physics letters 429.1-3 (2006): 310-316
#  Equation 23 with zero sigma
    beta_power = self._get_beta_accumulators()
    beta_power = math_ops.cast(beta_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    fric_t = math_ops.cast(self._fric_t, var.dtype.base_dtype)
    beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
    T0_t = math_ops.cast(self._T0_t, var.dtype.base_dtype)

    beta1=0.9
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - beta1) 
    m0_t=state_ops.assign(m, m, use_locking=self._use_locking)
    m_t = state_ops.assign(m, m * beta1  ,
                           use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
       m_t = state_ops.assign_add(m, m_scaled_g_values,
                                  use_locking=self._use_locking)
    beta2=0.99 #0.999 is OK too
    m2 = self.get_slot(var, "m2")
 
    v_scaled_g_values = ((grad-m_t) * (grad-m_t)) * (1 - beta2)
    m2_t = state_ops.assign(m2, m2 * beta2, use_locking=self._use_locking)
    with ops.control_dependencies([m2_t]):
      m2_t = state_ops.assign_add(m2, v_scaled_g_values,
                                 use_locking=self._use_locking)

    RMS=math_ops.sqrt(m2_t) 
    #In Adam friction is measured by sqrt(m2) 
    #In Adamax it is different. Better friction choices are possible
    gold = self.get_slot(var, "g") #past gradient
    gold_t = state_ops.assign(gold, gold, use_locking=self._use_locking)
    mass=1.0

    T0=T0_t*beta_power #Simulated annealing: temperature is  multiplied by beta each step 
    friction=fric_t*(RMS/mass) #if friction=0 we get Velocity-Verlet
    Acc0=-gold_t/mass
    Acc1=-grad/mass      

    v = self.get_slot(var, "v") #past velocity
    v_t = state_ops.assign(v, v, use_locking=self._use_locking)

    dv=-lr_t*friction*(0.5*v_t +lr_t*(Acc0-friction*v_t)/8)#Vanden-Einden term
    v_t = state_ops.assign_add(v, 0.5*lr_t*Acc0+dv, use_locking=self._use_locking)
    step=lr_t*v_t 

    dv=-lr_t*friction*(0.5*v_t+lr_t*(Acc1-friction*v_t)/8)#Vanden-Einden term
    v_t = state_ops.assign_add(v, 0.5*lr_t*Acc1+dv, use_locking=self._use_locking)

    Ekin2=math_ops.reduce_mean(v_t*v_t)
    T=Ekin2/3+1e-10  
     
    gammaT=1.0
    fT=math_ops.sqrt(1+gammaT*lr_t*(T0/T-1)) #Berendsen thermostat     
#    fT=0 # if fT=0 we get SGD optimizer
    v_t=state_ops.assign(v, fT*v, use_locking=self._use_locking) #scaling velocity to get desired temperature
    #save old gradient value
    gold_t = state_ops.assign(gold, grad, use_locking=self._use_locking)

    var_update = state_ops.assign_add(var,
                                      step,  
                                      use_locking=self._use_locking)
    
    return control_flow_ops.group(*[var_update,  gold_t,
                                     v_t, m_t, m2_t])

  def _resource_apply_dense(self, grad, var):
    beta_power = self._get_beta_accumulators()
    return self._apply_dense(
        grad, var)

  def _finish(self, update_ops, name_scope):
    with ops.control_dependencies(update_ops):
      beta_power = self._get_beta_accumulators()
      with ops.colocate_with(beta_power):
        update_beta = beta_power.assign(
            beta_power * self._beta_t, use_locking=self._use_locking)
    return control_flow_ops.group(*update_ops + [update_beta],
                                  name=name_scope)

  def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
