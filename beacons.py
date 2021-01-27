import numpy as np
from scipy.ndimage.interpolation import shift
from configuration import *
import itertools
from scipy.integrate import simps
from scipy.spatial import KDTree

def bound(value, abs_bound=0.05):
    return max(-abs_bound, min(abs_bound, value))

class Beacon:
    def __init__(self, location, beac_tag):
        self.pt = np.array([location, location])
        self.w = np.array([0., 0.])
        self.var = [np.NaN, np.NaN]
        self.amp = [np.NaN, np.NaN]
        self.mt = [np.NaN, np.NaN]
        self.ct = [[np.NaN, np.NaN], [np.NaN, np.NaN]]
        self.neigh = []
        # self.neigh_weigh = [[np.NaN], [np.NaN]]
        self.beac_tag = beac_tag

    def fnc_ants_at_beacon(self, ants):
        count = 0
        for ant_tag in ants:
            if ants[ant_tag].cl_beac == self.beac_tag:
                count += 1
        self.ants_at_beacon = count

    def amplitude(self):
        # self.amp[0] = self.w[0] + 0.1*max(self.neigh_weigh[0] + [self.w[0]], default = 0.1*self.w[0])
        self.amp[0] = self.w[0] + 0.1 * self.w[0]
        # self.amp[1] = self.w[1] + 0.1*max(self.neigh_weigh[1]+ [self.w[1]], default=0.1 * self.w[1])
        self.amp[1] = self.w[1] + 0.1 * self.w[1]

    def variance(self):
        if (1 - (self.w[0] / self.amp[0])) < numeric_margin:
            print('how is this possible')
        if self.w[0] > numeric_margin:
            # self.var[0] = -clip_range**2/(2*np.log(1- (self.w[0]/ampFactor) ))
            self.var[0] = -clip_range ** 2 / (2 * np.log(1 - (self.w[0] / self.amp[0])))
        else:
            self.var[0] = None

        if self.w[1] > numeric_margin:
            # self.var[1] = -clip_range ** 2 / (2 * np.log(1 - (self.w[1] / ampFactor)))
            self.var[1] = -clip_range ** 2 / (2 * np.log(1 - (self.w[1] / self.amp[1])))

        else:
            self.var[1] = None


    def nest_in_range(self):
        return np.linalg.norm(self.pt[1] - self.nest_location) <= self.target_range + numeric_margin

    def food_in_range(self):
        return np.linalg.norm(self.pt[1] - self.food_location) <= self.target_range + numeric_margin


class Beacons:
    def __init__(self, grid, beacon_grid=default_beacon_grid):
        self.grid = grid
        self.beacon_grid = beacon_grid

        self.beacons = dict()

        self.masks = []
        self.map_closest_beacon = []
        self.n = len(self.beacons)

    def update_amplitude(self):
        for beac_tag in self.beacons:
            self.beacons[beac_tag].amplitude()

    def update_variance(self):
        for beac_tag in self.beacons:
            self.beacons[beac_tag].variance()

    def update_beacon_configuration(self, position_changed=True, weights_changed=True, just_initialized = False ):
        if position_changed:
            self.update_masks()
        # if weights_changed:
        #     self.update_neighbours_weights()
        # TODO check why just_initialized condition is needed
        if weights_changed and not just_initialized:
            self.update_amplitude()
            self.update_variance()

    def move_step(self, W):
        self.update_m_c_beacons(W)
        self.update_locations()
        self.update_beacon_configuration()


    def non_influenced_points(self):
        influenced_points_dict = dict()
        for beac_tag in self.beacons:
            influenced_points_dict[beac_tag] = [point for point in self.grid.points if
                               (np.linalg.norm(np.array(point) - self.beacons[beac_tag].pt[1]) <= clip_range)]
        influenced_points = list(set([item for l in influenced_points_dict.values() for item in l]))
        non_influenced_points = [point for point in self.grid.points if point not in influenced_points]

        return np.array(non_influenced_points)

    def update_masks(self):
        self.tree = KDTree([self.beacons[beac_tag].pt[1] for beac_tag in self.beacons])
        self.map_closest_beacon = np.array([[self.tree.query([self.grid.X[cy][cx], self.grid.Y[cy][cx]])[1] for cx in
                                             range(len(self.grid.x))] for cy in range(len(self.grid.y))])
        self.masks = {beac_tag: (self.map_closest_beacon == count) * 1 for count, beac_tag in enumerate(self.beacons)}

        for beac_tag in self.masks:
            extended_mask = self.extend_mask(self.masks[beac_tag])
            self.beacons[beac_tag].neigh = [tag for tag in self.masks if
                                            True in (extended_mask + self.masks[tag] >= 2) and tag != beac_tag]

    # def update_neighbours_weights(self):
    #     for beac_tag in self.masks:
    #         self.beacons[beac_tag].neigh_weigh = [[self.beacons[tag].w[0] for tag in self.beacons[beac_tag].neigh],
    #                                               [self.beacons[tag].w[1] for tag in self.beacons[beac_tag].neigh]]

    def evaporate_weights(self, rho=default_rho):
        for beac_tag in self.beacons:
            self.beacons[beac_tag].w *= (1 - rho)

    def initialize_weights(self):
        for beac_tag in self.beacons:
            self.beacons[beac_tag].w = np.array([0., 0.])

    def update_m_c_beacons(self, W):
        for beac_tag in self.beacons:
            self.beacons[beac_tag].mt[0] = self.beacons[beac_tag].mt[1]
            self.beacons[beac_tag].ct[0][0] = self.beacons[beac_tag].ct[0][1]
            self.beacons[beac_tag].ct[1][0] = self.beacons[beac_tag].ct[1][1]
            self.beacons[beac_tag].mt[1] = simps(simps(W * self.masks[beac_tag], self.grid.x), self.grid.y)
            self.beacons[beac_tag].ct[0][1] = simps(simps(W * self.grid.X * self.masks[beac_tag], self.grid.x),
                                                    self.grid.y) / self.beacons[beac_tag].mt[1]
            self.beacons[beac_tag].ct[1][1] = simps(simps(W * self.grid.Y * self.masks[beac_tag], self.grid.x),
                                                    self.grid.y) / self.beacons[beac_tag].mt[1]

    def update_locations(self):
        for beac_tag in self.beacons:
            if beac_tag not in [0,1]:
                delta_ct = [(self.beacons[beac_tag].ct[0][1] - self.beacons[beac_tag].ct[0][0]),
                            (self.beacons[beac_tag].ct[1][1] - self.beacons[beac_tag].ct[1][0])]
                delta_x = (delta_ct[0] / dt) - kappa * (self.beacons[beac_tag].pt[1][0] - self.beacons[beac_tag].ct[0][1]) + \
                          delta_ct[0] * self.neigh_control_term(self.beacons[beac_tag])[0]
                delta_y = (delta_ct[1] / dt) - kappa * (self.beacons[beac_tag].pt[1][1] - self.beacons[beac_tag].ct[1][1]) + \
                          delta_ct[1] * self.neigh_control_term(self.beacons[beac_tag])[1]

                self.beacons[beac_tag].pt[0] = self.beacons[beac_tag].pt[1]
                self.beacons[beac_tag].pt[1] = [self.beacons[beac_tag].pt[0][0] + bound(delta_x * dt),
                                                self.beacons[beac_tag].pt[0][1] + bound(delta_y * dt)]

    def neigh_control_term(self, beacon):
        move_i = [0, 0]
        for count_j in beacon.neigh:
            j = self.beacons[count_j]
            move_j = [j.pt[0][1] - j.pt[0][0], j.pt[1][1] - j.pt[1][0]]
            if move_j[0] != 0:
                move_i[0] += (1 / move_j[0]) * ((j.ct[0][1] - j.ct[0][0]) / dt - kappa * (j.pt[0][1] - j.ct[0][1]))
            if move_j[1] != 0:
                move_i[1] += (1 / move_j[1]) * ((j.ct[1][1] - j.ct[1][0]) / dt - kappa * (j.pt[1][1] - j.ct[1][1]))
        return move_i

    def fnc_ants_at_beacons(self, ants):
        for beac_tag in self.beacons:
            self.beacons[beac_tag].fnc_ants_at_beacon(ants)

    @staticmethod
    def extend_mask(mask):
        return ((shift(mask, [1, 1], cval=0) + shift(mask, [1, -1], cval=0) + shift(mask, [-1, 1], cval=0) +
                 shift(mask, [-1, -1], cval=0) + shift(mask, [0, -1], cval=0) + shift(mask, [0, 1], cval=0) +
                 shift(mask, [1, 0], cval=0) + shift(mask, [-1, 0], cval=0)) >= 1) * 1


    def check_weights(self,to_check = 'W1', thres=0.):
        if to_check == 'W1':
            return {beac_tag: self.beacons[beac_tag].w[0] for beac_tag in self.beacons if
                    self.beacons[beac_tag].w[0] > thres}
        elif to_check == 'W2':
            return {beac_tag: self.beacons[beac_tag].w[1] for beac_tag in self.beacons if
                    self.beacons[beac_tag].w[1] > thres}
        elif to_check == 'W':
            return {beac_tag: self.beacons[beac_tag].w[0] + self.beacons[beac_tag].w[1] for beac_tag in self.beacons if
                    self.beacons[beac_tag].w[0] > thres or self.beacons[beac_tag].w[1] > thres}

    def check_ants(self, thres=0):
        return {beac_tag: self.beacons[beac_tag].ants_at_beacon for beac_tag in self.beacons if
                self.beacons[beac_tag].ants_at_beacon > thres}

# test = {beac_tag: simulation.beacons.beacons[beac_tag].w[0] + simulation.beacons.beacons[beac_tag].w[1] for beac_tag in
#         simulation.beacons.beacons if
#         simulation.beacons.beacons[beac_tag].w[0] > 0.00001 or simulation.beacons.beacons[beac_tag].w[1] > 0.00001}
