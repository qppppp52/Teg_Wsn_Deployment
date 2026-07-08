"""完整部署解对象"""
import numpy as np


class Solution:
    def __init__(self, num_candidates):
        self.x = np.zeros(num_candidates, dtype=np.int8)
        self.y = np.zeros(num_candidates, dtype=np.int8)
        self.c = np.zeros((num_candidates, num_candidates), dtype=np.int8)
        self.p_tx = np.zeros((num_candidates, num_candidates))
        self.n_sink_sensor = np.zeros(num_candidates, dtype=int)
        self.n_sink_ap = np.zeros(num_candidates, dtype=int)
        self.z_sink_sensor = [[] for _ in range(num_candidates)]
        self.z_sink_ap = [[] for _ in range(num_candidates)]
        self.sensor_power_consumption = np.zeros(num_candidates)
        self.ap_power_consumption = np.zeros(num_candidates)
        self.sensor_harvest_power = np.zeros(num_candidates)
        self.ap_harvest_power = np.zeros(num_candidates)
        self.coverage = 0.0
        self.throughput = 0.0
        self.cv = float("inf")
        self.cv_energy = 0.0
        self.cv_link = 0.0
        self.cv_capacity = 0.0
        self.cv_sink = 0.0
        self.cv_service = 0.0
        self.feasible = False
        self.objectives = np.zeros(2)
        self.metadata = {}

    def copy(self):
        new = Solution(len(self.x))
        new.x = self.x.copy()
        new.y = self.y.copy()
        new.c = self.c.copy()
        new.p_tx = self.p_tx.copy()
        new.n_sink_sensor = self.n_sink_sensor.copy()
        new.n_sink_ap = self.n_sink_ap.copy()
        new.z_sink_sensor = [list(z) for z in self.z_sink_sensor]
        new.z_sink_ap = [list(z) for z in self.z_sink_ap]
        new.sensor_power_consumption = self.sensor_power_consumption.copy()
        new.ap_power_consumption = self.ap_power_consumption.copy()
        new.sensor_harvest_power = self.sensor_harvest_power.copy()
        new.ap_harvest_power = self.ap_harvest_power.copy()
        new.coverage = self.coverage
        new.throughput = self.throughput
        new.cv = self.cv
        new.cv_energy = self.cv_energy
        new.cv_link = self.cv_link
        new.cv_capacity = self.cv_capacity
        new.cv_sink = self.cv_sink
        new.cv_service = self.cv_service
        new.feasible = self.feasible
        new.objectives = self.objectives.copy()
        return new
