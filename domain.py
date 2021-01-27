import numpy as np
from configuration import *
from scipy.spatial import KDTree

class Grid:
    def __init__(self,grid_size=default_grid_size,domain=default_domain):
        self.grid_size = grid_size
        self.domain = domain
        self.x = np.linspace(0,domain[0],grid_size[0]+1)
        self.y = np.linspace(0,domain[1],grid_size[1]+1)

        self.X, self.Y = np.meshgrid(self.x,self.y)
        self.W1 = np.zeros(self.X.shape)
        self.W2 = np.zeros(self.X.shape)

        # self.grid_points_elem = [[x_elem, y_elem] for x_elem in range(0, len(self.x)) for y_elem in range(0, len(self.y))]
        # self.grid_points_cor = [[self.x[x_elem], self.y[y_elem]] for x_elem in range(0, len(self.x))
        #                         for y_elem in range(0, len(self.y))]
        # self.grid_points = {count: {'elem': [x_elem, y_elem], 'cor': [self.x[x_elem], self.y[y_elem]]} for
        #                count, [x_elem, y_elem] in enumerate(self.grid_points_elem)}
        # self.tree = KDTree(self.grid_points_cor)

    def update_graph_weights(self,beacons):
        # self.W1 = gaussian(self.X, self.Y, beacons.beacons,0) + \
        #           elips_gaussian(self.X, self.Y)
        # self.W2 = gaussian(self.X, self.Y, beacons.beacons,1) + \
        #             elips_gaussian(self.X, self.Y)
        # self.W = gaussian(self.X, self.Y, beacons.beacons,0) + \
        #          gaussian(self.X, self.Y, beacons.beacons,1) + \
        #          elips_gaussian(self.X, self.Y)

        self.W1 = gaussian(self.X, self.Y, beacons.beacons,0)
        self.W2 = gaussian(self.X, self.Y, beacons.beacons, 1)
        self.W = self.W1 + self.W2 + np.ones(self.W1.shape)*offset
        # self.W = self.W1 + self.W2 + np.ones(self.W1.shape)

    # def find_closest_grid_point(self,point):
    #     return self.grid_points[self.tree.query(point)[1]]

def mapper(fnc):
    def inner(x, y, beacons,w_type):
        value = 0
        # for beacon in beacons:
        #     value += fnc(x, y, beacon,w_type)
        for beac_tag in beacons:
            value += fnc(x, y, beacons[beac_tag],w_type)
        return value
    return inner

def shift_clip(value, beacon, w_type):
    # new_value = value - beacon.w[w_type] * np.exp(-(clip_range**2/(2*beacon.var)))
    if beacon.var[w_type]:
        # new_value = value - ampFactor * np.exp(-(clip_range ** 2 / (2 * beacon.var[w_type])))
        new_value = value - beacon.amp[w_type] * np.exp(-(clip_range ** 2 / (2 * beacon.var[w_type])))
    else:
        new_value = np.zeros(value.shape)
    return np.clip(new_value, 0, np.inf)

@mapper
def gaussian(x,y, beacon,w_type):
    # value = beacon.w[w_type] * np.exp(-((x - beacon.pt[1][0]) ** 2 + (y - beacon.pt[1][1]) ** 2) / (2 * beacon.var))
    if beacon.var[w_type]:
        # value = ampFactor * np.exp(-((x - beacon.pt[1][0]) ** 2 + (y - beacon.pt[1][1]) ** 2) / (2 * beacon.var[w_type]))
        value = beacon.amp[w_type] * np.exp(
            -((x - beacon.pt[1][0]) ** 2 + (y - beacon.pt[1][1]) ** 2) / (2 * beacon.var[w_type]))
    else:
        value = np.zeros(x.shape)
    return shift_clip(value,beacon, w_type)

@mapper
def drv_gaussian(x,y, beacon,w_type):
    # value = beacon.w[w_type] * np.exp(-((x - beacon.pt[1][0]) ** 2 + (y - beacon.pt[1][1]) ** 2) / (2 * beacon.var))
    if beacon.var[w_type]:
        # value = ampFactor * np.exp(
        #     -((x - beacon.pt[1][0]) ** 2 + (y - beacon.pt[1][1]) ** 2) / (2 * beacon.var[w_type]))
        value = beacon.amp[w_type] * np.exp(
            -((x - beacon.pt[1][0]) ** 2 + (y - beacon.pt[1][1]) ** 2) / (2 * beacon.var[w_type]))
        to_return = np.array([-(x - beacon.pt[1][0]) / (beacon.var[w_type]), -(y - beacon.pt[1][1]) / (beacon.var[w_type])]) * shift_clip(
            value, beacon, w_type)
    else:
        to_return = np.zeros(2)
    return to_return


def elips_gaussian(x,y,a=elips_a,c=elips_c,ampl=elips_ampl):
    value = ampl*np.exp(-a*(x-default_domain[0]/2)**2 - c*(y-default_domain[1]/2)**2)
    return np.clip(value,0,0.5)

def x_drv_elips_gaussian(x,y,a=elips_a,c=elips_c,ampl=elips_ampl):
    value = elips_gaussian(x,y,a=a,c=c,ampl=ampl)
    if value == 0.5:
        return 0
    else:
        return -2*a*(x-default_domain[0]/2)*value

def y_drv_elips_gaussian(x,y,a=elips_a,c=elips_c,ampl=elips_ampl):
    value = elips_gaussian(x,y,a=a,c=c,ampl=ampl)
    if value == 0.5:
        return 0
    else:
        return -2*c*(y-default_domain[1]/2)*value

