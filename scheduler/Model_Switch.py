from .Scheduler import *
from copy import deepcopy
from simulator.sim_environment.Datacenter import *
import time
from datetime import datetime, timedelta

class Model_Switch(Scheduler):
    def __init__(self, path, datacenter, num_containers):
        super().__init__(path=path)

        self.datacenter = datacenter
        self.dataset_processed = False
        self.start_time = time.time()  # Track the start time
        #self.counter = 24
        #self.records = self.counter * num_containers
        self.prev_violations = {service_id: float('inf') for service_id in range(num_containers)}


    def placement(self):

        prev_decision = deepcopy(self.datacenter.containers_hosts)
        prev_decision_tuple = deepcopy(self.datacenter.containers_hosts_tuple)
        prev_ip_model_image = deepcopy(self.datacenter.ip_model_image)

        # if self.counter > 0:
        #     self.counter -= 1
        #     return prev_decision_tuple, self.datacenter.containers_hosts_obj, 0

        violation_rate = self.dataset_processing()
        for service_id, node_id in enumerate(self.datacenter.containers_hosts):
            row = violation_rate[violation_rate['cid']==service_id]
            current_sla_violation = row['sla_violation_rate'].squeeze()
            model_name = row['model_name'].squeeze()
            if current_sla_violation > 0.1:
                if model_name == 'yolov5m':
                    self.datacenter.ip_model_image[service_id] = ('', 'yolov5s', 'babarkhan93/yolostorch-prop:0.3')
                elif model_name == 'yolov5s':
                    self.datacenter.ip_model_image[service_id] = ('', 'yolov5n', 'babarkhan93/yolotorch-prop:0.3')
            elif current_sla_violation == 0 and self.prev_violations[service_id] > 0:
                if model_name == 'yolov5n':
                    self.datacenter.ip_model_image[service_id] = ('', 'yolov5s', 'babarkhan93/yolostorch-prop:0.3')
                elif model_name == 'yolov5s':
                    self.datacenter.ip_model_image[service_id] = ('', 'yolov5m', 'babarkhan93/yolomtorch:0.3')
            
            self.prev_violations[service_id] = current_sla_violation

        #self.counter = 24 # resetting the counter
        return prev_decision_tuple, self.datacenter.containers_hosts_obj, 0


    def dataset_processing_counter(self):

        df = self.dataset_reading()

        df.drop(columns=['time', 'location', 'file_name', 'propagation_delay (s)', 'e2e_delay (s)'], inplace=True)
        df = df.tail(self.records)
        df['sla_violation'] = df['processing_delay (ms)'] > 700
    
        # Group by 'core' to count total records and SLA violations
        sla_stats = df.groupby(['cid', 'model_name', 'core']).agg(
            total_records=('processing_delay (ms)', 'size'),
            sla_violations=('sla_violation', 'sum')
        ).reset_index()

        # Calculate SLA violation rate
        sla_stats['sla_violation_rate'] = sla_stats['sla_violations'] / sla_stats['total_records']
        return sla_stats
    

    def dataset_processing(self):

        df = self.dataset_reading()

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