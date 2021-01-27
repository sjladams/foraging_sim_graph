local = True

total_time = 400

default_grid_size = [100,52]      # [200,100] / [100,50]
default_domain = [20,10]           # [40,20] / [20,10]
default_nest_location = [5., 5.]    # [5.,4.] / [5., 4.]
default_food_location = [15.,5.]   # [30.,14.] / [15., 6.]
default_beacon_grid = [10,8]        # [20,16] / [10,8]

default_N_batch = 5
default_N_total = 300 # 500 / 100

# ampFactor = 30
kappa=1   #1
lam= 1#1
rew=1
default_rho = 0.01 #0.0001
default_epsilon = 0.1 #0.05 #5
DEBUG=1
dt=0.5   # \TODO put to 0.5
default_target_range=0.5   #dt   # 1
# default_var = 10
clip_range = 1 #2.

elips_a = 0.008    # 0.002 / 0.008
elips_c = 0.03     # 0.009 / 0.03
elips_ampl = 1

offset = 1 # 1e-6
# threshold = (1-default_rho)**90 * offset
threshold = 1e-6
# n = np.log(threshold/offset)/np.log(1-default_rho)
step_threshold = 1e-9 #1e-3   # 1e-7

move_type = 'add_switch' #'der'/ 'add' / 'add_switch'

numeric_margin = 1e-6