from beacons import *
from configuration import *
from domain import *
from ants import *

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simps
from scipy.spatial import KDTree
from scipy.ndimage.interpolation import shift
from scipy.spatial import Voronoi, voronoi_plot_2d

if local:
    FOLDER_LOCATION = './figures/'
else:
    FOLDER_LOCATION = './figures_manuels_desk/'

class Simulations:
    def __init__(self,grid_size=default_grid_size, beacon_grid=default_beacon_grid,
                 nest_location=default_nest_location, food_location=default_food_location, N_total=default_N_total,
                 N_batch = default_N_batch, rho=default_rho,domain=default_domain):
        self.N_total = N_total
        self.N_batch = N_batch
        self.rho = rho
        self.grid = Grid(grid_size=grid_size,domain=domain)
        self.total_trips = dict()

        self.beacons = Beacons(self.grid, beacon_grid=beacon_grid)
        # self.beacons.initialize_weights()
        # self.beacons.update_masks()
        # self.beacons.update_neighbours_beacons()

        self.ants = Ants(nest_location, food_location, epsilon=default_epsilon)
        self.ants.release_ants(1,list(self.beacons.beacons.keys()))
        self.ants.steps(self.beacons)
        self.switch_step()

        # self.update_beacon_weights()
        # self.beacons.update_neighbours_beacons()

        self.grid.update_graph_weights(self.beacons)
        # self.beacons.update_m_c_beacons(self.grid.W)

    def sim_step(self, time_step):
        N_till_now = (time_step+1)*self.N_batch
        # if time_step < 75:
        if N_till_now < self.N_total:
            self.ants.release_ants(self.N_batch, list(self.beacons.beacons.keys()))
        self.ants.steps(self.beacons)

        self.beacons.evaporate_weights(rho=self.rho)

        self.update_beacon_weights()
        self.beacons.update_beacon_configuration(position_changed=False)

        self.grid.update_graph_weights(self.beacons)

    def sim_step_mov_beac(self,time_step,switch_time=250):
        self.sim_step(time_step)

        if time_step >= switch_time:
            # self.beacons.move_step(self.grid.W)
            self.switch_step()
            self.grid.update_graph_weights(self.beacons)

        self.store_nr_trips(time_step)

    def update_ant_beacon_connection(self):
        self.ants.find_closest_beacon(self.beacons)
        self.beacons.fnc_ants_at_beacons(self.ants.ants)

    def switch_step(self):
        self.ants.update_weights(self.beacons)

        old_ants = self.ants.ants.copy()
        tags_changed = []
        for ant_tag in old_ants:
            if sum(self.ants.ants[ant_tag].w) <= 0. and np.random.uniform() <1 :

                tags_changed += [ant_tag]

                self.beacons.beacons[ant_tag] = Beacon(self.ants.ants[ant_tag].nt[1], ant_tag)
                self.beacons.update_beacon_configuration(just_initialized=True)
                self.initialize_beacon_weights(ant_tag)
                self.beacons.update_beacon_configuration(position_changed=False)

                del self.ants.ants[ant_tag]
                self.update_ant_beacon_connection()

                # check beacons one by one:
                self.ants.update_weights(self.beacons)

        weight_dict = self.beacons.check_weights(to_check = 'W',thres=threshold)
        ant_dict = self.beacons.check_ants()

        # to comment:
        old_beacons = self.beacons.beacons.copy()
        for beac_tag in old_beacons:
            if beac_tag not in weight_dict and beac_tag not in ant_dict and beac_tag not in tags_changed:
                self.ants.ants[beac_tag] = Ant(default_nest_location, default_food_location, beac_tag,
                                               init_location=self.beacons.beacons[beac_tag].pt[1])
                del self.beacons.beacons[beac_tag]

                self.beacons.update_beacon_configuration()
                self.update_ant_beacon_connection()


    def reward(self, weights,rew,ants_at_beacon):
        # return ampFactor * self.rho * (1+lam*max(weights) + rew)/self.N
        # return ampFactor * (lam * max(weights) + rew) / ants_at_beacon
        # return ampFactor * (lam * max(weights)) / (self.N *ants_at_beacon) +rew
        # return ampFactor * (lam * max(weights) + rew) / (self.N * ants_at_beacon)
        # return ampFactor * self.rho * (0.5 + lam * max(weights) + rew) / (self.N * ants_at_beacon)
        # return ampFactor * self.rho * ( lam * max(weights, default=0) + rew) / (ants_at_beacon)
        return self.rho * (lam * max(weights, default=0) + rew) / (ants_at_beacon)
        # return 1

    def initialize_beacon_weights(self,tag):
        # neigh_beac_weigh = self.ants.ants[tag].find_neigh_beacon_weights(self.beacons)
        self.ants.ants[tag].find_neigh_beacons(self.beacons)
        W1_weights = [self.beacons.beacons[beac_tag].w[0] for beac_tag in self.ants.ants[tag].neigh]
        W2_weights = [self.beacons.beacons[beac_tag].w[0] for beac_tag in self.ants.ants[tag].neigh]

        if self.ants.ants[tag]._reached_nest():
            self.beacons.beacons[tag].w[0] += self.reward(W1_weights, rew, 1)
        else:
            self.beacons.beacons[tag].w[0] += self.reward(W1_weights, 0, 1)

        if self.ants.ants[tag]._reached_food():
            self.beacons.beacons[tag].w[1] += self.reward(W2_weights, rew, 1)
        else:
            self.beacons.beacons[tag].w[1] += self.reward(W2_weights, 0, 1)

    def update_beacon_weights(self):
        self.ants.find_closest_beacon(self.beacons)
        self.beacons.fnc_ants_at_beacons(self.ants.ants)

        for ant_tag in self.ants.ants:
            # neigh_beac_weigh = self.ants.ants[ant_tag].find_neigh_beacon_weights(self.beacons)
            self.ants.ants[ant_tag].find_neigh_beacons(self.beacons)
            W1_weights = [self.beacons.beacons[tag].w[0] for tag in self.ants.ants[ant_tag].neigh]
            W2_weights = [self.beacons.beacons[tag].w[0] for tag in self.ants.ants[ant_tag].neigh]

            if self.ants.ants[ant_tag].mode[0]==0:
                if self.ants.ants[ant_tag]._reached_nest():
                    self.beacons.beacons[self.ants.ants[ant_tag].cl_beac].w[0] += self.reward(W1_weights,
                                                        rew, self.beacons.beacons[self.ants.ants[ant_tag].cl_beac].ants_at_beacon)
                elif self.ants.ants[ant_tag]._reached_food():
                    self.beacons.beacons[self.ants.ants[ant_tag].cl_beac].w[1] += self.reward(W2_weights,
                                                        rew, self.beacons.beacons[self.ants.ants[ant_tag].cl_beac].ants_at_beacon)
                else:
                    self.beacons.beacons[self.ants.ants[ant_tag].cl_beac].w[0] += self.reward(W1_weights,
                                                        0, self.beacons.beacons[self.ants.ants[ant_tag].cl_beac].ants_at_beacon)
            if self.ants.ants[ant_tag].mode[0]==1:
                if self.ants.ants[ant_tag]._reached_food():
                    self.beacons.beacons[self.ants.ants[ant_tag].cl_beac].w[1] += self.reward(W2_weights,
                                                        rew, self.beacons.beacons[self.ants.ants[ant_tag].cl_beac].ants_at_beacon)
                elif self.ants.ants[ant_tag]._reached_nest():
                    self.beacons.beacons[self.ants.ants[ant_tag].cl_beac].w[0] += self.reward(W1_weights,
                                                        rew, self.beacons.beacons[self.ants.ants[ant_tag].cl_beac].ants_at_beacon)
                else:
                    self.beacons.beacons[self.ants.ants[ant_tag].cl_beac].w[1] += self.reward(W2_weights,
                                                        0, self.beacons.beacons[self.ants.ants[ant_tag].cl_beac].ants_at_beacon)

        # for beac_tag in self.beacons.beacons:
        #     neigh_beac_weigh = self.beacons.beacons[beac_tag].neigh_weigh
        #
        #     if self.beacons.beacons[beac_tag].nest_in_range():
        #         self.beacons.beacons[beac_tag].w[0] += self.reward(neigh_beac_weigh[0], rew,
        #                                                            self.beacons.beacons[beac_tag].ants_at_beacon)
        #     else:
        #         self.beacons.beacons[beac_tag].w[0] += self.reward(neigh_beac_weigh[0], 0,
        #                                                            self.beacons.beacons[beac_tag].ants_at_beacon)
        #
        #     if self.beacons.beacons[beac_tag].food_in_range():
        #         self.beacons.beacons[beac_tag].w[1] += self.reward(neigh_beac_weigh[1], rew,
        #                                                            self.beacons.beacons[beac_tag].ants_at_beacon)
        #     else:
        #         self.beacons.beacons[beac_tag].w[1] += self.reward(neigh_beac_weigh[1], 0,
        #                                                            self.beacons.beacons[beac_tag].ants_at_beacon)


    def plt(self, to_plot='W'):
        # vor = Voronoi([item.pt[1] for item in self.beacons.beacons])
        # voronoi_plot_2d(vor, show_vertices=False)

        if to_plot == 'W1':
            plt.contourf(self.grid.X, self.grid.Y, self.grid.W1)  # ,levels=np.linspace(0,1,10))
        elif to_plot == 'W2':
            plt.contourf(self.grid.X, self.grid.Y, self.grid.W2)  # ,levels=np.linspace(0,1,10))
        elif to_plot == 'W':
            plt.contourf(self.grid.X, self.grid.Y, self.grid.W)  # ,levels=np.linspace(0,1,10))

        # plt.plot([item.pt[1][0] for item in self.beacons.beacons],
        #          [item.pt[1][1] for item in self.beacons.beacons], 'r*')
        plt.plot([item.nt[1][0] for item in self.ants.ants if item.mode[1] == 0],
                 [item.nt[1][1] for item in self.ants.ants if item.mode[1] == 0], 'g*')
        plt.plot([item.nt[1][0] for item in self.ants.ants if item.mode[1] == 1],
                 [item.nt[1][1] for item in self.ants.ants if item.mode[1] == 1], 'y*')
        plt.plot([default_nest_location[0], default_food_location[0]],
                 [default_nest_location[1], default_food_location[1]], 'r*')
        # plt.plot(list(itertools.chain.from_iterable(self.grid.X)),
        #          list(itertools.chain.from_iterable(self.grid.Y)), 'b*')

        plt.xlim(0-5, self.grid.domain[0]+5)
        plt.ylim(0-5, self.grid.domain[1]+5)
        plt.colorbar()
        plt.show()

    def plt_3d(self, to_plot='W', fig_tag=None):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.gca(projection='3d')

        locations_non_influenced_points = self.beacons.non_influenced_points()
        value_non_influenced_points = np.zeros(locations_non_influenced_points.shape[0])

        locations_beacons = np.array([self.beacons.beacons[beac_tag].pt[1] for beac_tag in self.beacons.beacons])

        if to_plot == 'W1':
            w_beacons = np.array([self.beacons.beacons[beac_tag].w[0] for beac_tag in self.beacons.beacons])
        elif to_plot == 'W2':
            w_beacons = np.array([self.beacons.beacons[beac_tag].w[1] for beac_tag in self.beacons.beacons])
        else:
            w_beacons = np.array([self.beacons.beacons[beac_tag].w[0] + self.beacons.beacons[beac_tag].w[1]
                         for beac_tag in self.beacons.beacons])

        W_beacons = griddata(np.concatenate((locations_beacons,locations_non_influenced_points)),
                             np.concatenate((w_beacons,value_non_influenced_points)),  (self.grid.X, self.grid.Y), method='linear')

        ax.plot_surface(self.grid.X, self.grid.Y, W_beacons,
                        cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=True)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        if to_plot == 'W1' and fig_tag:
            plt.savefig(FOLDER_LOCATION + 'W1_3d/' + str(to_plot) + '_' + str(fig_tag) + '.png')
            plt.close()
        elif to_plot == 'W2' and fig_tag:
            plt.savefig(FOLDER_LOCATION + 'W2_3d/' + str(to_plot) + '_' + str(fig_tag) + '.png')
            plt.close()
        elif to_plot == 'W' and fig_tag:
            plt.savefig(FOLDER_LOCATION + 'W_3d/' + str(to_plot) + '_' + str(fig_tag) + '.png')
            plt.close()
        else:
            plt.show()

    def plt_beacons(self, to_plot='W', fig_tag=None):
        # vor = Voronoi([item.pt[1] for item in self.beacons.beacons])
        if len(self.beacons.beacons) >3:
            vor = Voronoi([self.beacons.beacons[beac_tag].pt[1] for beac_tag in self.beacons.beacons])
            voronoi_plot_2d(vor, show_vertices=False)

        locations_non_influenced_points = self.beacons.non_influenced_points()
        value_non_influenced_points = np.zeros(locations_non_influenced_points.shape[0])

        locations_beacons = np.array([self.beacons.beacons[beac_tag].pt[1] for beac_tag in self.beacons.beacons])

        if to_plot == 'W1':
            w_beacons = np.array([self.beacons.beacons[beac_tag].w[0] for beac_tag in self.beacons.beacons])
        elif to_plot == 'W2':
            w_beacons = np.array([self.beacons.beacons[beac_tag].w[1] for beac_tag in self.beacons.beacons])
        else:
            w_beacons = np.array([self.beacons.beacons[beac_tag].w[0] + self.beacons.beacons[beac_tag].w[1]
                         for beac_tag in self.beacons.beacons])

        W_beacons = griddata(np.concatenate((locations_beacons,locations_non_influenced_points)),
                             np.concatenate((w_beacons,value_non_influenced_points)) + 0.00000000001,  (self.grid.X, self.grid.Y), method='linear')

        if not np.max(W_beacons):
            max_range = 0.001
        else:
            max_range = np.max(W_beacons)
        plt.contourf(self.grid.X, self.grid.Y, W_beacons,levels=np.linspace(0,
                        max_range,100))
        # plt.contourf(self.grid.X, self.grid.Y, W_beacons)

        # if to_plot == 'W1':
        #     for beac_tag in self.beacons.check_weights(to_check='W1'):
        #         item = self.beacons.beacons[beac_tag]
        #         size = (item.w[0] / max(self.beacons.check_weights(to_check='W1').values())) * 10
        #         plt.plot([item.pt[1][0]], [item.pt[1][1]], 'o', color='black', markersize=size)
        # elif to_plot == 'W2':
        #     for beac_tag in self.beacons.check_weights(to_check='W2'):
        #         item = self.beacons.beacons[beac_tag]
        #         size = (item.w[1] / max(self.beacons.check_weights(to_check='W2').values())) * 10
        #         plt.plot([item.pt[1][0]], [item.pt[1][1]], 'o', color='black', markersize=size)
        # elif to_plot == 'W':
        #     for beac_tag in self.beacons.check_weights(to_check='W'):
        #         item = self.beacons.beacons[beac_tag]
        #         size = (item.w[1] / max(self.beacons.check_weights(to_check='W').values())) * 10
        #         plt.plot([item.pt[1][0]], [item.pt[1][1]], 'o', color='black', markersize=size)


        plt.plot([default_nest_location[0], default_food_location[0]],
                 [default_nest_location[1], default_food_location[1]], 'r*')

        plt.plot([self.ants.ants[ant_tag].nt[1][0] for ant_tag in self.ants.ants if
                  self.ants.ants[ant_tag].mode[1] == 0],
                 [self.ants.ants[ant_tag].nt[1][1] for ant_tag in self.ants.ants if
                  self.ants.ants[ant_tag].mode[1] == 0], 'g*')
        plt.plot([self.ants.ants[ant_tag].nt[1][0] for ant_tag in self.ants.ants if
                  self.ants.ants[ant_tag].mode[1] == 1],
                 [self.ants.ants[ant_tag].nt[1][1] for ant_tag in self.ants.ants if
                  self.ants.ants[ant_tag].mode[1] == 1], 'y*')

        plt.xlim(0, self.grid.domain[0])
        plt.ylim(0, self.grid.domain[1])
        plt.colorbar()
        if to_plot == 'W1' and fig_tag:
            plt.savefig(FOLDER_LOCATION + 'W1/' + str(to_plot) + '_' + str(fig_tag) + '.png')
            plt.close()
        elif to_plot == 'W2' and fig_tag:
            plt.savefig(FOLDER_LOCATION +'W2/' + str(to_plot) + '_' + str(fig_tag) + '.png')
            plt.close()
        elif to_plot == 'W' and fig_tag:
            plt.savefig(FOLDER_LOCATION +'W/' + str(to_plot) + '_' + str(fig_tag) + '.png')
            plt.close()
        else:
            plt.show()


    def store_nr_trips(self,t):
        self.total_trips[t] = sum([self.ants.ants[ant_tag].trips for ant_tag in self.ants.ants])

    def plot_trips(self,total_time,fig_tag=None):
        trips_sequence = [self.total_trips[time] for time in range(0,total_time)]

        plt.plot(range(0,total_time), trips_sequence, 'r')

        if fig_tag:
            plt.savefig(FOLDER_LOCATION + 'total_trips_' + str(fig_tag) + '.png')
            plt.close()
        else:
            plt.show()



    def plt_range_beacons(self, fig_tag=None):
        # vor = Voronoi([item.pt[1] for item in self.beacons.beacons])
        fig, ax = plt.subplots(figsize=(12, 6))

        # if len(self.beacons.beacons) >3:
        #     vor = Voronoi([self.beacons.beacons[beac_tag].pt[1] for beac_tag in self.beacons.beacons])
        #     voronoi_plot_2d(vor, show_vertices=False)

        for beac_tag in self.beacons.beacons:
            circle = plt.Circle(self.beacons.beacons[beac_tag].pt[1], clip_range , color='r',fill=False)
            ax.add_patch(circle)


        for count, ant_tag in enumerate(self.ants.ants):
            if count ==1:
                circle = plt.Circle(self.ants.ants[ant_tag].nt[1], clip_range , color='g',fill=False)
                ax.add_patch(circle)

        plt.plot([default_nest_location[0], default_food_location[0]],
                 [default_nest_location[1], default_food_location[1]], 'r*')

        plt.plot([self.ants.ants[ant_tag].nt[1][0] for ant_tag in self.ants.ants if
                  self.ants.ants[ant_tag].mode[1] == 0],
                 [self.ants.ants[ant_tag].nt[1][1] for ant_tag in self.ants.ants if
                  self.ants.ants[ant_tag].mode[1] == 0], 'g*')
        plt.plot([self.ants.ants[ant_tag].nt[1][0] for ant_tag in self.ants.ants if
                  self.ants.ants[ant_tag].mode[1] == 1],
                 [self.ants.ants[ant_tag].nt[1][1] for ant_tag in self.ants.ants if
                  self.ants.ants[ant_tag].mode[1] == 1], 'y*')

        plt.plot([self.beacons.beacons[beac_tag].pt[1][0] for beac_tag in self.beacons.beacons],
                 [self.beacons.beacons[beac_tag].pt[1][1] for beac_tag in self.beacons.beacons], 'b*')

        # plt.xlim(0, self.grid.domain[0])
        # plt.ylim(0, self.grid.domain[1])

        ax.set_xlim((0, self.grid.domain[0]))
        ax.set_ylim((0, self.grid.domain[1]))

        if fig_tag:
            plt.savefig(FOLDER_LOCATION + 'range_beacons' + str(fig_tag) + '.png')
            plt.close()
        else:
            plt.show()