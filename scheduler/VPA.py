from .Scheduler import *
from copy import deepcopy
from simulator.sim_environment.Datacenter import *
import time
from datetime import datetime, timedelta

class VPA(Scheduler):
    def __init__(self, path, datacenter):
        super().__init__(path=path)

        self.datacenter = datacenter
        self.dataset_processed = False
        
    def placement(self):
        # as there is no pod scaling for GKE and no image change. Thus, sending back the first decision

        prev_decision = deepcopy(self.datacenter.containers_hosts)
        prev_decision_tuple = deepcopy(self.datacenter.containers_hosts_tuple)
        prev_ip_model_image = deepcopy(self.datacenter.ip_model_image)

        return prev_decision_tuple, self.datacenter.containers_hosts_obj, 0