import cv2
from matplotlib import pyplot as plt
import numpy as np


class AreaController:
    def __init__(self, working_areas, size):
        self.working_areas = working_areas
        self.size = size
        self.number_of_people_vector = list()
        self.total_working_area_numpy = np.zeros(size)
        for i in range(len(working_areas)):
            self.number_of_people_vector.append(list())

    def add_area(self, area):
        self.working_areas.append(area)
        self.number_of_people_vector.append(list())

    def run(self, heat_map):
        self.heat_map_size = heat_map.shape
        for i, working_area in enumerate(self.working_areas):
            working_area_numpy = np.zeros(self.size)
            working_area_numpy[working_area[0][0]:working_area[1][0], working_area[0][1]:working_area[1][1]] = 1
            working_area_numpy = cv2.resize(working_area_numpy, (self.heat_map_size[1], self.heat_map_size[0]))
            area_heat_map = heat_map * working_area_numpy
            number_of_people = np.sum(area_heat_map)
            self.number_of_people_vector[i].append(number_of_people)
            self.total_working_area_numpy[working_area[0][0]:working_area[1][0], working_area[0][1]:working_area[1][1]] = 1

    def get_total_working_area_numpy(self):
        return cv2.resize(self.total_working_area_numpy, (self.heat_map_size[1], self.heat_map_size[0]))

    def draw_people_count(self):
        palette = plt.get_cmap('Set1')
        num = 0
        fig, ax = plt.subplots()
        for i in range(len(self.number_of_people_vector)):
            num += 1
            ax.plot(self.number_of_people_vector[i], color=palette(num), linewidth=1, alpha=0.9, label=i)
        plt.legend(loc=2)
        plt.title("People count in areas", loc='left', fontsize=12, fontweight=0, color='orange')
        plt.xlabel("Frame")
        plt.ylabel("People")
        plt.savefig('output/output.png')

        for i in range(len(self.number_of_people_vector)):
            fig, ax = plt.subplots()
            ax.plot(self.number_of_people_vector[i])
            plt.savefig('output/output' + str(i) + '.png')
