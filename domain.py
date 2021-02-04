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

        self.points = [tuple([cx, cy]) for cx in self.x for cy in self.y]

        self.ref_lines = [np.array([[0,0],[0,domain[1]]]),
                         np.array([[0,domain[1]],[domain[0],domain[1]]]),
                         np.array([[domain[0],domain[1]],[domain[0],0]]),
                         np.array([[domain[0],0],[0,0]])]

        # self.W1 = np.zeros(self.X.shape)
        # self.W2 = np.zeros(self.X.shape)

    # def update_graph_weights(self,beacons):
    #     self.W1 = gaussian(self.X, self.Y, beacons.beacons,0)
    #     self.W2 = gaussian(self.X, self.Y, beacons.beacons, 1)
    #     self.W = self.W1 + self.W2 + np.ones(self.W1.shape)*offset

    @staticmethod
    def normalize(item):
        return item / np.linalg.norm(item)

    @staticmethod
    def perpendicular(a):
        b = np.empty_like(a)
        #     b[0] = -a[1]
        #     b[1] = a[0]
        b[0] = a[1]
        b[1] = -a[0]
        return b

    @staticmethod
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def check_in_domain(self, point):
        if (point[0] > 0 and point[0] < self.domain[0] and \
                point[1] > 0 and point[1] < self.domain[1]):
            return True
        else:
            return False

    def line_intersection(self, ant_locations):
        for count, ref_line in enumerate(self.ref_lines):
            xdiff = np.array([ref_line[0][0] - ref_line[1][0], ant_locations[0][0] - ant_locations[1][0]])
            ydiff = np.array([ref_line[0][1] - ref_line[1][1], ant_locations[0][1] - ant_locations[1][1]])

            div = self.det(xdiff, ydiff)
            if div != 0:
                d = (self.det(*ref_line), self.det(*ant_locations))
                inter = np.array([self.det(d, xdiff) / div, self.det(d, ydiff) / div])
                if np.linalg.norm(inter - ant_locations[0]) < np.linalg.norm(ant_locations[0] - ant_locations[1]):
                    return count, inter
        return None, None

    def obstacle_avoidance(self, start_point, move):
        if not self.check_in_domain(start_point + move):
            index_line, inter = self.line_intersection(np.array([start_point, start_point + move]))
            if isinstance(index_line, int):
                ref_line = self.ref_lines[index_line]
                ref_vec = np.array([ref_line[1][0] - ref_line[0][0], ref_line[1][1] - ref_line[0][1]])
                return inter + self.normalize(self.perpendicular(ref_vec)) * (
                            np.linalg.norm(move) - np.linalg.norm(start_point - inter))
            else:
                return start_point + move
        else:
            return start_point + move


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

