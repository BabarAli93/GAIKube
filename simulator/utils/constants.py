import os

# dfined by the user

import pathlib

p = pathlib.Path(__file__)

PROJECT_PATH = "/home/babarali/Extended_ProKube"
DATA_PATH = os.path.join(PROJECT_PATH, "data")

# generated baesd on the users' path
#DATACENTER_PATH = os.path.join(DATA_PATH, "clusters")
TRAIN_RESULTS_PATH = os.path.join(DATA_PATH, "train-results")
TESTS_RESULTS_PATH = os.path.join(DATA_PATH, "test-results")

CONFIGS_PATH = os.path.join(PROJECT_PATH, 'data', "configs")
DATASETS_PATH = os.path.join(PROJECT_PATH, 'datasets')
BACKUP_PATH = os.path.join(DATA_PATH, "backup")
PLOTS_PATH = os.path.join(DATA_PATH, "plots")
BITBRAINS_PATH = os.path.join(DATASETS_PATH, "bitbrains")
#ALIBABA_PATH = os.path.join(DATA_PATH, "alibaba")

def _create_dirs():
    """
    create directories if they don't exist
    """

    if not os.path.exists(TRAIN_RESULTS_PATH):
        os.makedirs(TRAIN_RESULTS_PATH)
    if not os.path.exists(CONFIGS_PATH):
        os.makedirs(CONFIGS_PATH)
    if not os.path.exists(BACKUP_PATH):
        os.makedirs(BACKUP_PATH)
    if not os.path.exists(TESTS_RESULTS_PATH):
        os.makedirs(TESTS_RESULTS_PATH)
    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)
    if not os.path.exists(BITBRAINS_PATH):
        os.makedirs(BITBRAINS_PATH)

# _create_dirs()

# ENVS = {
#     'sim-scheduler': SimSchedulerEnv,
#     'kube-scheduler': KubeSchedulerEnv,
# }
#
# ENVSMAP = {
#     'sim-scheduler': 'SimSchedulerEnv-v0',
#     'sim-binpacking': 'SimBinpackingEnv-v0',
#     'kube-scheduler': 'KubeSchedulerEnv-v0',
#     'kube-binpacking': 'KubeBinpackingEnv-v0',
#     'kube-greedy': 'KubeGreedyEnv-v0',
# }


if __name__ == "__main__":
    _create_dirs()
