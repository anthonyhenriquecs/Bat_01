# Bat_01

import numpy as np
from random import random, uniform

import self as self
from numba import float32, int32
from numba.experimental import jitclass

bat_algorithm_spec = [
    ('x_range', float32[:]),
    ('y_range', float32[:]),
    ('Population_Size', int32)
    ('Population', float32[:,:])
    ('Fitness', float32[:,:])
]

@jitclass(bat_algorithm_spec)
class Bat_Algorithm(object):
    def __int__(self, optimiser, Population_Size=100, Num_Movements = 100):
        #Informações de função
        self.x_range = optimiser.x_range
        self.y_range = optimiser.y_range

        #População de morcegos e Dinamica.
        self.Population_Size = Population_Size
        self.Population, self.Fitness = Initialise_Population(optimiser)




    def Initialise_Population(optmiser):
        Population = np.empty(shape=(self.Population_Size,2), dtype=np.float32)
        Fitness = np.empty(shape=(self.Population_Size, 1), dtype=np.float32)

        for i in range(self.Population_Size):
            Population[i,0] = (self.x_range[1] - self.x_range[0]) * float32(random()) + self.x_range[0]
            Population[i, 1] = (self.x_range[1] - self.x_range[0]) * float32(random()) + self.x_range[0]
            Fitness[i,0] = optmiser.Query(Population[i, 0], Population[i,1])

        return Population ,Fitness
