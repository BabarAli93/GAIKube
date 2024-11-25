import os
from copy import deepcopy
import pathlib
import numpy as np
import time
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
import pandas as pd

from simulator.sim_environment.Datacenter import *
from simulator.utils.constants import (
    CONFIGS_PATH
)
from workload.Workload import *
from scheduler.Scheduler import *
from scheduler.Model_Switch import *

def warmup():
    config_file_path = os.path.join(CONFIGS_PATH, 'datacenter.json')
    with open(config_file_path) as cf: config = json.loads(cf.read)

    pp.pprint(config)

    generator_config = deepcopy(config)
    del generator_config['notes']

    return None


if __name__ == "__main__":
    warmup()
