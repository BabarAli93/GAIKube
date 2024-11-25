from .Scheduler import *
from copy import deepcopy
from simulator.sim_environment.Datacenter import *
import time
from datetime import datetime, timedelta


import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import timesfm
import matplotlib.dates as mdates
import pulp

# get the 30 minutes prediction of each machine
# get the current placements
# get the current utilization
# migrate if CPU is overloading
# Scale down to less accurate model in overload
# scale up to less load
# choose best accuracy model
# ms for sla


class GAIKUBE(Scheduler):
    def __init__(self, path, datacenter, bitbrains_path):
        super().__init__(path=path)

        self.datacenter = datacenter
        self.dataset_processed = False
        ## there are three folder naming fast, rnd and synthetic_data
        self.client_path = path
        self.bitbrains_path = bitbrains_path
        self.fast_path = os.path.join(bitbrains_path, 'fastStorage/2013-8/')
        self.training = False
        self.dataset_processed = False 
        self.df_modified = None 
        self.core_model = {
            500: ['yolov5n', 'babarkhan93/yolotorch-prop:0.3'],
            1000: ['yolov5s', 'babarkhan93/yolostorch-prop:0.3'],
            2000: ['yolov5m', 'babarkhan93/yolomtorch:0.3']
        }
        self.prev_violations = {service_id: float('inf') for service_id in range(self.datacenter.num_containers)}

    def placement(self):
        if not self.dataset_processed:
            df_pred = self.timesfm()
            nppreds = self.pred_sequence(df_pred) 
            self.dataset_processed = True
        else:
            nppreds = self.pred_sequence(self.df_modified) 

        prev_decision_tuple = deepcopy(self.datacenter.containers_hosts_tuple)
        sla_violations = self.sla_violations()

        results = np.any(nppreds >= 80, axis=1)
        indices = np.where(results)[0] # indices are nodes where there is expected Greater CPU utilization.
        # Given our sorted nodes scenario, index 0 -> 2 core, index 1 - > 4 core, index 3 -> 6 core
        if len(indices) > 0:
            print('Possible Scale Down')
            modified_sla_set = self.handle_overload(indices, sla_violations)
        else:
            modified_sla_set = self.handle_normal(sla_violations)

        self.model_switching(modified_sla_set)
        print('##### New Decisons #####')
        print(self.datacenter.containers_hosts_tuple)
        for key, val in self.datacenter.ip_model_image.items():
            print(f'Container Id: {key}, Model: {val[1]}')
        print(f'Predictions: {nppreds}')


        #num_moves = sum(1 for prev, new in zip(prev_decision_tuple, self.datacenter.containers_hosts_tuple) if prev[0] != new[0])
        num_scalings = sum(1 for prev, new in zip(prev_decision_tuple, self.datacenter.containers_hosts_tuple) if prev[1] != new[1])

        return prev_decision_tuple, self.datacenter.containers_hosts_obj, num_scalings
    
    def handle_overload(self, overload_indices, sla_violations):

        # making use of indices to find which containers has this node as a value
        containers_on_node = []
        sla_violations['pass'] = None
        host_requested_local = deepcopy(self.datacenter.hosts_resources_req)
        for i in range(len(overload_indices)):
            # it is a list of of arrays where each array represents the indices of containers hosted in this node
            containers_on_node.append(np.where(self.datacenter.containers_hosts == overload_indices[i])[0])
        # now need to sort containers of each node to migrate the high power onces first
        # use containers_hosts_tuple
        sorted_overload_indices = {}
        
        for node_index in range(len(containers_on_node)):
            # Extract indices from overload_nodes
            extracted_indices = containers_on_node[node_index]
            # Sort the indices based on the second element of the corresponding tuples in descending order
            sorted_indices = sorted(extracted_indices, key=lambda idx: self.datacenter.containers_hosts_tuple[idx][1], reverse=True)
            # Store the sorted indices in the dictionary
            sorted_overload_indices[overload_indices[node_index]] = sorted_indices
         # sorted_overload_indices holds the container ids in th asceding orrder of their requested CPU
        
        for prev_node, val in sorted_overload_indices.items():
            for _, val in enumerate(val):
                placed = False
                cpu = self.datacenter.containers_hosts_tuple[val][1]
                hosts_remain_list, _ = self.host_container_list()
                #for new_node in range(self.datacenter.num_hosts):
                #    if new_node != prev_node:
                while hosts_remain_list:
                        host = hosts_remain_list.pop()
                        while cpu > min(self.core_model.keys())-1:
                            new_cpu = self.datacenter.hosts_resources_req[host[0]][1] + cpu
                            cond = (self.datacenter.hosts_resources_req[host[0], 1:]+self.datacenter.containers_request[val][2] < self.datacenter.hosts_resources_alloc[host[0], 1:])
                            if ((new_cpu / self.datacenter.hosts_resources_alloc[host[0]][1])*100 < 80) and np.all(cond):
                                new_node = host[0]
                                self.datacenter.containers_hosts_tuple[val] = (new_node, cpu) # TODO: run and test it 
                                self.datacenter.containers_hosts[val] = new_node

                                self.datacenter.hosts_resources_req[prev_node][1:] -=  self.datacenter.containers_request[val][1:]
                                self.datacenter.hosts_resources_remain[prev_node][1:] += self.datacenter.containers_request[val][1:]
                                self.datacenter.containers_request[val][1] = cpu
                                self.datacenter.hosts_resources_remain[new_node][1:] -= self.datacenter.containers_request[val][1:]
                                self.datacenter.hosts_resources_req[new_node][1:] += self.datacenter.containers_request[val][1:]
                                self.datacenter.hosts_resources_util[:, 1:]  = (self.datacenter.hosts_resources_req[:,1:]/self.datacenter.hosts_resources_alloc[:, 1:])*100
                                model_image = self.core_model[cpu]
                                self.datacenter.ip_model_image[val] = ('', model_image[0], model_image[1])
                                sla_violations.loc[sla_violations['cid'] == val, 'pass'] = True
                                hosts_remain_list, _ = self.host_container_list()
                                placed = True
                                break
                            else:
                                cpu = int(cpu/2)
                        if placed:
                            break

                ## if the load is managed then break the loop and go to next overloaded node
                if self.datacenter.hosts_resources_util[prev_node][1] < 80:
                    #
                    break
                else:
                    print(f'No destination found to lessen load of node: {prev_node}')

        
        return sla_violations
    
    def handle_normal(self, sla_violations):
        # this should scale up to best model and 
        # for all the containers, try to go to 2 core and find desitnation for it. If not found, go to 1 core
        hosts_remain_list, container_request_list = self.host_container_list()
        sla_violations['pass'] = None

        # iterate over all these candidate condtainers, and get their respective containers_hosts
        for _ in range(len(container_request_list)): # TODO: May have to move to while insread of for
            placed = False
            item = container_request_list.pop()
            service_id, prev_node = item[0], self.datacenter.containers_hosts[item[0]] # lowest cpu container id and current node
            while hosts_remain_list:
                cpu = max(self.core_model.keys()) # maximum cpu a container can attain
                host = hosts_remain_list.pop()
                while cpu > item[1]: # while cpu is greater than the current cpu of this pod
                    host_new_cpu = self.datacenter.hosts_resources_req[host[0],1] + cpu # adding current cpu to the newly cpu
                    cond = (self.datacenter.hosts_resources_req[host[0], 1:]+self.datacenter.containers_request[service_id][2] < self.datacenter.hosts_resources_alloc[host[0], 1:])
                    if ((host_new_cpu / self.datacenter.hosts_resources_alloc[host[0]][1])*100 < 80) and np.all(cond):
                        new_node = host[0]
                        # modify all the things here and go to next pod
                        self.datacenter.containers_hosts_tuple[service_id] = (host[0], cpu)
                        self.datacenter.containers_hosts[service_id] = host[0]

                        self.datacenter.hosts_resources_req[prev_node][1:] -=  self.datacenter.containers_request[service_id][1:]
                        self.datacenter.hosts_resources_remain[prev_node][1:] += self.datacenter.containers_request[service_id][1:]
                        self.datacenter.containers_request[service_id][1] = cpu

                        self.datacenter.hosts_resources_remain[new_node][1:] -= self.datacenter.containers_request[service_id][1:]
                        self.datacenter.hosts_resources_req[new_node][1:] += self.datacenter.containers_request[service_id][1:]
                        self.datacenter.hosts_resources_util[:, 1:]  = (self.datacenter.hosts_resources_req[:,1:]/self.datacenter.hosts_resources_alloc[:, 1:])*100

                        model_image = self.core_model[cpu]
                        self.datacenter.ip_model_image[service_id] = ('', model_image[0], model_image[1])
                        sla_violations.loc[sla_violations['cid'] == service_id, 'pass'] = True
                        hosts_remain_list, container_request_list = self.host_container_list()
                        placed = True
                        # after modifying stuff, update the
                        break
                    else:
                        cpu = int(cpu/2)
                if placed:
                    break

        return sla_violations
    
    #### Model-Switching Code 
    def model_switching(self, sla_violations):

        for service_id, node_id in enumerate(self.datacenter.containers_hosts):
            row = sla_violations[sla_violations['cid']==service_id]
            if row['pass'].squeeze():
                continue
            current_sla_violation = row['sla_violation_rate'].squeeze()
            model_name, current_core = row['model_name'].squeeze(), row['core'].squeeze()
            best_core = next(core for core, value in self.core_model.items() if value[0] == model_name)
            placed = True
            hosts_remain_list, _ = self.host_container_list()
            if current_sla_violation > 0.1:
                if model_name == 'yolov5m':
                    tuple = ('', 'yolov5s', 'babarkhan93/yolostorch-prop:0.3')
                    placed = False
                elif model_name == 'yolov5s':
                    tuple = ('', 'yolov5n', 'babarkhan93/yolotorch-prop:0.3')
                    placed = False
            elif current_sla_violation == 0 and self.prev_violations[service_id] > 0:
                if model_name == 'yolov5n':# and best_core != current_core:
                    tuple = ('', 'yolov5s', 'babarkhan93/yolostorch-prop:0.3')
                    placed = False
                elif model_name == 'yolov5s':# and best_core != current_core:
                    tuple = ('', 'yolov5m', 'babarkhan93/yolomtorch:0.3')
                    placed = False
            
            cpu = current_core
            prev_node = self.datacenter.containers_hosts[service_id]
            while not placed and hosts_remain_list:
                host = hosts_remain_list.pop()
                
                host_new_cpu = self.datacenter.hosts_resources_req[host[0],1] + cpu # adding current cpu to the newly cpu
                cond = (self.datacenter.hosts_resources_req[host[0], 1:]+self.datacenter.containers_request[service_id][2] < self.datacenter.hosts_resources_alloc[host[0], 1:])
                if ((host_new_cpu / self.datacenter.hosts_resources_alloc[host[0]][1])*100 < 80) and np.all(cond):
                    new_node = host[0]
                    #print('Possible Allotment!')
                    # modify all the things here and go to next pod
                    self.datacenter.containers_hosts_tuple[service_id] = (host[0], cpu)
                    self.datacenter.containers_hosts[service_id] = host[0]

                    self.datacenter.hosts_resources_req[prev_node][1:] -=  self.datacenter.containers_request[service_id][1:]
                    self.datacenter.hosts_resources_remain[prev_node][1:] += self.datacenter.containers_request[service_id][1:]
                    self.datacenter.containers_request[service_id][1] = cpu

                    self.datacenter.hosts_resources_remain[new_node][1:] -= self.datacenter.containers_request[service_id][1:]
                    self.datacenter.hosts_resources_req[new_node][1:] += self.datacenter.containers_request[service_id][1:]
                    self.datacenter.hosts_resources_util[:, 1:]  = (self.datacenter.hosts_resources_req[:,1:]/self.datacenter.hosts_resources_alloc[:, 1:])*100

                    hosts_remain_list, container_request_list = self.host_container_list()
                    self.datacenter.ip_model_image[service_id] = tuple
                    placed = True
                    # after modifying stuff, update the
                    break

            self.prev_violations[service_id] = current_sla_violation

        #self.counter = 24 # resetting the counter
        #return prev_decision_tuple, self.datacenter.containers_hosts_obj, 0
        return None
    

    def sla_violations(self):

        df = pd.read_csv(self.client_path)
        core_replacement = {'half':500, 'one': 1000, 'two': 2000}
        df['core'] = df['core'].replace(core_replacement)

        df.drop(columns=['location', 'file_name', 'propagation_delay (s)', 'e2e_delay (s)'], inplace=True)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        now = df['time'].max()
        five_minute_ago = now - timedelta(minutes=1)
        df_last_minute = df[df['time'] >= five_minute_ago]
        df_last_minute['sla_violation'] = df_last_minute['processing_delay (ms)'] > 700

        # Group by 'core' to count total records and SLA violations
        sla_stats = df_last_minute.groupby(['cid', 'model_name', 'core']).agg(
            total_records=('processing_delay (ms)', 'size'),
            sla_violations=('sla_violation', 'sum')
        ).reset_index()

        # Calculate SLA violation rate
        sla_stats['sla_violation_rate'] = sla_stats['sla_violations'] / sla_stats['total_records']

        return sla_stats
    
    def host_container_list(self):
        # getting the Hosts with remaining of lowest container free available resources and sorting
        candidate_hosts = self.datacenter.hosts_resources_remain[self.datacenter.hosts_resources_remain[:, 1] > min(self.core_model.keys())]
        hosts_sorted_indices = np.argsort(candidate_hosts[:,1])
        hosts_remain_sort = candidate_hosts[hosts_sorted_indices]
        hosts_remain_list = list(hosts_remain_sort) 

        # get list of containers with lesser than the maximum attainable CPU allowed to them
        candidate_containers = self.datacenter.containers_request[self.datacenter.containers_request[:, 1] < max(self.core_model.keys())]
        sorted_indices = np.argsort(-candidate_containers[:, 1])
        containers_request_sort = candidate_containers[sorted_indices]
        container_request_list = list(containers_request_sort)

        return hosts_remain_list, container_request_list

    #### TimesFM Processing, Training, Predictions and Prediction Sequencing code is all below

    def pred_sequence(self, df):

        """"
            This function recived a dataframe, it creates a numpy.ndarray of each core as row and 6 predicted values in each row
            It further truncates the first occurances of each core from dataframe and return modified dataframe, so that
            we have predict from the next timestamp onwards
            """
        unique_ids = [2, 4, 6]
        num_values = 6         # next 6 predictions
        result = np.zeros((len(unique_ids), num_values))

        # Extracting values for each unique_id
        for i, uid in enumerate(unique_ids):
            values = df[df['unique_id'] == uid]['timesfm'].values[:num_values]
            result[i, :len(values)] = values

        first_row_indices = df.groupby('unique_id').head(1).index
        df_dropped = df.drop(first_row_indices)
        self.df_modified = df_dropped

        return result

    def timesfm(self):

        df = self.df_bitbrains()
        df_synth = self.df_gai()

        df = pd.concat([df, df_synth])
        df.set_index('ds', inplace=True)
        df.reset_index(inplace=True)

        filepath = os.path.join(self.bitbrains_path, 'times_prediction.csv')
        if os.path.exists(filepath):
            df_pred = pd.read_csv(filepath)
            df_pred = df_pred[['ds', 'unique_id', 'timesfm', 'timesfm-q-0.6', 'CPU usage [%]']]
            ## [1434:1743]
            df_pred = df_pred.iloc[1554:1689]
        else:
            df_pred = self.timesfm_training(df)
            df_pred = df_pred[['ds', 'unique_id', 'timesfm', 'timesfm-q-0.6', 'CPU usage [%]']]
            ##[1434:1743]
            df_pred = df_pred.iloc[1554:1689]

        return df_pred

    def timesfm_training(self, df):

        horizon_len = 6
        df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)

        tfm = timesfm.TimesFm(
            context_len=288,
            horizon_len=horizon_len,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend="cpu",
            )
        tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

        def timesfm_pred(pred_length, df_train, df_test):
            all_preds = []
            iter = int(df_test.shape[0]/(pred_length*3))
            inf_time = []

            for i in range(iter):

                s_time = time.time()
                predictions = tfm.forecast_on_df(
                inputs=df_train,
                freq='5min',  # monthly
                value_name="target"
                )
                inf = time.time() - s_time
                inf_time.append(inf)

                all_preds.append(predictions)
                test_values = df_test.loc[df_test['ds'].isin(predictions['ds'])].reset_index(drop=True)
                # Add the corresponding test values for the next iteration
                df_train = pd.concat([df_train, test_values]).reset_index(drop=True)

            all_preds = pd.concat(all_preds).reset_index()

            return all_preds, inf_time
        
        timefm_preds, inf_time = timesfm_pred(horizon_len, df_train, df_test)
        timefm_preds = timefm_preds.sort_values(by=['ds', 'unique_id'])
        timefm_preds.set_index('ds', inplace=True)
        timefm_preds.reset_index(inplace=True)
        timefm_preds.drop(columns=['index'], inplace=True)

        df_test = df_test.sort_values(by=['ds', 'unique_id'])
        df_test.set_index('ds', inplace=True)
        df_test.reset_index(inplace=True)

        timefm_preds['CPU usage [%]'] = df_test['target']

        timefm_preds.to_csv(os.path.join(self.bitbrains_path, 'times_prediction.csv'))

        return timefm_preds

    def df_bitbrains(self): 


        def df_processing(df):

            df.columns = df.columns.str.replace('\t', '')
            df.insert(0, "DateTime", df['Timestamp [ms]'].apply(lambda x: datetime.datetime.fromtimestamp(x).replace(second=0, microsecond=0)))
            df = df.drop(columns=['Timestamp [ms]'])
            df.set_index('DateTime', inplace=True)
            df = df.sort_index()
            df = df.resample('5min').ffill()
            df = df[['CPU cores', 'CPU usage [%]']]
            return df

        df2 = pd.read_csv(os.path.join(self.fast_path, '983.csv'), sep=';')
        df4 = pd.read_csv(os.path.join(self.fast_path, '980.csv'), sep=';')
        df6 = pd.read_csv(os.path.join(self.fast_path, '943.csv'), sep=';')

        dfs_list = [df2, df4, df6]
        dfs = [df_processing(df) for df in dfs_list]
        df2 = dfs[0]
        df4 = dfs[1]
        df6 = dfs[2]

        merged_df = pd.concat(dfs)
        merged_df = merged_df.sort_index()
        df = merged_df
        df.reset_index(inplace=True)
        df.rename(columns={"CPU cores": "unique_id", "DateTime": "ds", "CPU usage [%]": "target"}, inplace=True)

        return df

    def df_gai(self):
        """
            This code processes Dopper GAN Data 
            """
        def time_reindex(df, start_datetime, end_datetime, core):

            intervals_per_day = 288
            one_day_range = pd.date_range(start=start_datetime, periods=intervals_per_day, freq='5min')
            unique_example_ids = df['example_id'].unique()
            new_timestamps = pd.date_range(start=start_datetime, end=end_datetime, freq='5min')

            # Ensure the length matches the number of rows per example_id
            if len(new_timestamps) < intervals_per_day * len(unique_example_ids):
                raise ValueError("Not enough timestamps to cover all rows")

            # Create a new column for updated DateTime
            df['UpdatedDateTime'] = pd.NaT

            # Update the DateTime for each example_id
            for idx, example_id in enumerate(unique_example_ids):
                start_idx = idx * intervals_per_day
                end_idx = start_idx + intervals_per_day
                df.loc[df['example_id'] == example_id, 'UpdatedDateTime'] = new_timestamps[start_idx:end_idx].values

            df.drop(columns=['DateTime'], inplace=True)
            df['DateTime'] = df['UpdatedDateTime']
            df.set_index('DateTime', inplace=True)
            df.sort_index()

            df.drop(columns=['example_id', 'UpdatedDateTime'], inplace=True)
            df.insert(0, 'CPU cores', core)
            df.reset_index(inplace=True)

            return df        

        df2_synth = pd.read_csv(os.path.join(self.bitbrains_path, 'synthetic_data/twocore_synth_v2.csv'))
        df2_synth = df2_synth[df2_synth['example_id'] < 5]
        df4_synth = pd.read_csv(os.path.join(self.bitbrains_path, 'synthetic_data/fourcore_synth_v2.csv'))
        df4_synth = df4_synth[df4_synth['example_id'] < 5]
        df6_synth = pd.read_csv(os.path.join(self.bitbrains_path, 'synthetic_data/sixcore_synth_v2.csv'))
        df6_synth = df6_synth[df6_synth['example_id'] < 5]
        start_datetime = pd.Timestamp('2013-09-11 14:40:00')
        end_datetime = pd.Timestamp('2013-09-16 14:35:00')
        df2_synth = time_reindex(df2_synth, start_datetime, end_datetime, core=2)
        df4_synth = time_reindex(df4_synth, start_datetime, end_datetime, core=4)
        df6_synth = time_reindex(df6_synth, start_datetime, end_datetime, core=6)

        merged_synth = pd.concat([df2_synth, df4_synth, df6_synth])
        merged_synth.reset_index(inplace=True)
        merged_synth = merged_synth.sort_values(by=['DateTime', 'CPU cores'])
        merged_synth = merged_synth[['DateTime', 'CPU cores', 'CPU usage [%]']]
        merged_synth.set_index('DateTime', inplace=True)
        merged_synth.reset_index(inplace=True)
        merged_synth.rename(columns={"CPU cores": "unique_id", "DateTime": "ds", "CPU usage [%]": "target"}, inplace=True)

        return merged_synth

