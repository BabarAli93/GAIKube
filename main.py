import click
import os
import json
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
from copy import deepcopy
import pathlib
from PIL import Image
import numpy as np
import time
import csv
import pandas as pd
import warnings
#from datetime import datetime

from simulator.sim_environment.Datacenter import *
from simulator.utils.constants import (
    CONFIGS_PATH, 
    BITBRAINS_PATH
)
from workload.Workload import *
from scheduler.Scheduler import *
from scheduler.Latency_Cost import *
from scheduler.Model_Switch import *
from scheduler.Latency import *
from scheduler.Cost import *
from scheduler.VPA import *
from scheduler.GAIKUBE import *

from datetime import datetime, timedelta

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=DeprecationWarning)#

#NUM_STEPS = 50
sla_threshold = ()
timestamp = time.time()
fieldnames = ['time', 'cid', 'model_name', 'file_name', 'propagation_delay (s)', 'processing_delay (ms)', 'e2e_delay (s)', 'location', 'core']
serverfields = ['num_moves', 'num_scalings', 'cost']
nodesfields = [' mid', 'cpu_cap', 'ram_cap', 'cpu_alloc', 'ram_alloc', 'cpu_req', 'ram_req', 'cpu_remain', 'ram_remain']
ip_model_image= dict()


def initializeEnvironment(client_path: str, server_path: str):    
    #start_time = datetime.now()
    #print("Start time:", start_time)
    config_file_path = os.path.join(CONFIGS_PATH, 'datacenter.json')
    with open(config_file_path) as cf: config = json.loads(cf.read())

    pp.pprint(config)
    

    generator_config = deepcopy(config)
    del generator_config['notes']
    datacenter = DatacenterGeneration(**generator_config)

    datacenter.generateCluster() ## creating a collection of multi region clusters and it will generate hosts

    warmed = datacenter.warmup()

    ip_model_image, cost = datacenter.randomDeployment() # an array where index is service id and value is node id
    
    workload = WorkloadGenerator() # noting to pass for constructor
    
    sla = np.zeros(len(ip_model_image))
    client_stats = []

    print('Experiment Started!!')

    duration = timedelta(minutes=1)
    start_time = datetime.now()
    while datetime.now() - start_time < duration:
        for idx, value in ip_model_image.items():
            response = workload.client_request(ip=value[0], version=value[1], cid=idx)
            if response:
                client_stats.append(response)
                response['location'] = datacenter.containers_locations[idx][:-2]
                response['core'] = datacenter.containers_request[idx][1]
                with open(client_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(response)

                # if response['e2e_delay (s)'] > 1:
                #     sla[idx] += 1
        
    #scheduler = Latency_Cost(dataset_path, datacenter) # provide parameters if there are any to initialize constructor
    scheduler = Model_Switch(client_path, datacenter, config['nums']['num_containers']) 
    #scheduler = VPA(client_path, datacenter)
    #scheduler = GAIKUBE(client_path, datacenter, BITBRAINS_PATH)
    #scheduler = Cost(dataset_path, datacenter) 

    server_stat = {
        'num_moves': 0,
        'num_scalings': 0,
        'cost': cost
    }
    with open(server_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=serverfields)
                writer.writerow(server_stat)
    return datacenter, workload, scheduler, client_stats, sla, server_stat


def stepExperiment(datacenter, workload, scheduler, client_stats, sla_stats, client_path, server_path):

    prev_decision, prev_containers_hosts_obj, num_scalings = scheduler.placement()
    ip_model_image, num_moves, cost = datacenter.gkems_migrate(prev_decision, prev_containers_hosts_obj)

    server_stat = {
        'num_moves': num_moves,
        'num_scalings': num_scalings,
        'cost': cost
        }

    with open(server_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=serverfields)
        writer.writerow(server_stat)

    duration = timedelta(minutes=1)
    start_time = datetime.now()
    while datetime.now() - start_time < duration:
        for idx, value in ip_model_image.items():
            response = workload.client_request(ip=value[0], version=value[1], cid=idx)
            if response:
                client_stats.append(response)
                response['location'] = datacenter.containers_locations[idx][:-2]
                response['core'] = datacenter.containers_request[idx][1]
                with open(client_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(response)

                #if response['e2e_delay (s)'] > 1:
                #    sla_stats[idx] += 1
                #    if sla_stats[idx] >= sla_threshold:
                        # Notify here, for example:
                #        print(f"SLA violation threshold reached for container {idx}")
                #        sla_stats[idx] = 0
                #else:
                #    # Reset the counter for this IP since there is no violation
                #    sla_stats[idx] = 0
            #print(f'SLA Stats: {sla_stats}')

    return server_stat, client_stats

def generate_csv(path: str):
    """ 
    This function generates CSV file to log content if does not exist

    """
    base_name = 'clients_stats_msgke.csv'
    server_base = 'server_stats_msgke.csv'
    nodes_base = 'nodes_utils_stats_msgke.csv'
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    experiment_folder = os.path.join(path, 'experiment_logs/ms_gke')
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    client_file = f"{base_name[:-4]}_{timestamp}.csv"
    client_path = os.path.join(experiment_folder, client_file)

    with open(client_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    print('Client CSV created...')
    
    # server CSV
    server_csv = f"{server_base[:-4]}_{timestamp}.csv"
    server_csv_path = os.path.join(experiment_folder, server_csv)
    with open(server_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=serverfields)
        writer.writeheader()
    print('Server CSV created...')

    nodes_csv = f"{nodes_base[:-4]}_{timestamp}.csv"
    nodes_path = os.path.join(experiment_folder, nodes_csv)
    with open(nodes_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=nodesfields)
        writer.writeheader()
    print('Nodes util CSV created...')

    return client_path, server_csv_path, nodes_path

if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    columns = ['mid', 'cpu', 'ram', 'source']
    path = pathlib.Path(__file__).parent.resolve()
    client_path, server_path, nodes_path = generate_csv(path)

    cleint_path = str(client_path)
    server_path = str(server_path)
    
    server_stats = list()
    datacenter, workload, scheduler, client_stats, sla, server_stat = initializeEnvironment(client_path, server_path)
    server_stats.append(server_stat)
    
    #for num in range(NUM_STEPS):

    #duration = timedelta(minutes=90)
    #start_time = datetime.now()
    #while datetime.now() - start_time < duration:
    for i in range(43):
        print(f'\n##################   Iteration {i}   #######################')
        server_stat, client_stats = stepExperiment(datacenter=datacenter, workload=workload, scheduler=scheduler,
                                                    client_stats=client_stats, sla_stats=sla, client_path=client_path, server_path=server_path)
        server_stats.append(server_stat)

    df_host_capacity = pd.concat(datacenter.host_capacity_list, ignore_index=True)
    df_hosts_remaining = pd.concat(datacenter.hosts_remaining_list, ignore_index=True)
    df_host_alloc = pd.concat(datacenter.host_alloc_list, ignore_index=True)
    df_hosts_requested = pd.concat(datacenter.hosts_requested_list, ignore_index=True)
    df_host_capacity.columns = ['mid', 'cpu_cap', 'ram_cap'] 
    df_host_alloc.columns = ['mid_temp', 'cpu_alloc', 'ram_alloc']
    df_hosts_requested.columns = ['mid_temp', 'cpu_req', 'ram_req']
    df_hosts_remaining.columns = ['mid_temp', 'cpu_remain', 'ram_remain']  

    combined_df = pd.concat([df_host_capacity, df_host_alloc, df_hosts_requested, df_hosts_remaining], ignore_index=False, axis=1)
    combined_df.drop(columns=['mid_temp'], inplace=True)
    combined_df.to_csv(nodes_path, index=False)

    df_client = pd.DataFrame(client_stats)
    df_client.to_csv('client_stats_sla_v1.csv', index=False)
    df_server = pd.DataFrame(server_stats)
    df_server.to_csv('server_stats_sla_v1.csv', index=False)
        
