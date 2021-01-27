import numpy as np
from simulation import *

simulation = Simulations()

simulation.plt_3d(to_plot='W')
simulation.plt_beacons(to_plot='W')

for t in range(0,int(total_time/dt)):
    # print('time is:' + str(t))
    # print({count:item.cl_beac for count,item in enumerate(simulation.ants.ants)})
    # print(simulation.beacons.check_weights(to_show = 'W1'))
    simulation.sim_step(t, switch_time=0)
    # simulation.plt_beacons(to_plot='W1')
    if t % 2 ==0:  #5
        simulation.plt_beacons(to_plot='W1',fig_tag=t)
        simulation.plt_beacons(to_plot='W2',fig_tag=t)
        simulation.plt_beacons(to_plot='W', fig_tag=t)

        simulation.plt_3d(to_plot='W1',fig_tag=t)
        simulation.plt_3d(to_plot='W2',fig_tag=t)
        simulation.plt_3d(to_plot='W', fig_tag=t)

        # simulation.plt_beacons(to_plot='W1')
        # simulation.plt_beacons(to_plot='W2')
        # simulation.plt_beacons(to_plot='W')
        # simulation.plt_beacons(to_plot='W1')

        # simulation.plt_range_beacons()

simulation.plot_trips(int(total_time/dt),fig_tag='test_case')


# test = [item.mode[1] for item in simulation.ants.ants]
# test = sum(test)

# plt.plot([item.nt[1][0] for item in simulation.ants.ants],
#          [item.nt[1][1] for item in simulation.ants.ants], 'g*')
# plt.plot([default_nest_location[0], default_food_location[0]],
#          [default_nest_location[1], default_food_location[1]], 'r*')
# # plt.plot(list(itertools.chain.from_iterable(self.grid.X)),
# #          list(itertools.chain.from_iterable(self.grid.Y)), 'b*')
# plt.xlim(-10, 30)
# plt.ylim(-10, 20)
# plt.show()


# trips_per_an = [simulation.ants.ants[ant_tag].trips for ant_tag in simulation.ants.ants]
# aaa = max(trips_per_an)
#
# distance = np.sqrt( (default_nest_location[0]-default_food_location[0])**2 +
#                     (default_nest_location[1]-default_food_location[1])**2)
#
# min_time_trip = distance / dt