import os
import pathlib
import requests
import time
import json
import numpy as np
from PIL import Image
from tenacity import retry, wait_exponential, stop_after_attempt


class WorkloadGenerator:
    def __init__(self):

        path = pathlib.Path(__file__).parent.parent.resolve()
        self.dir_path = path.joinpath('images')
        self.save_path = path.joinpath('results')
        self.logs_path = path.joinpath('logs')
        self.image_path = self.dir_path.joinpath('6.jpg')

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
    def client_request(self, ip, version, cid):
        item = '6.jpg'
        s_time = time.time()
        headers = {'clientside': str(s_time)}
        with open(self.image_path, 'rb') as f:
            files = {'img': f}
            response = requests.post(f'http://{ip}:5000/detect', files=files, headers=headers, timeout=(10,30))

        if response.status_code == 200:
            current_time = time.time()
            json_response = json.loads(response.text)
            img_list = json_response['image']
            img_np = np.array(img_list, dtype=np.uint8)
            image = Image.fromarray(img_np)
            image.save(os.path.join(self.save_path, f"{item}"))
            e_time = time.time()
            e2e_delay = e_time - s_time
            server_client = json_response['serverclientprop']
            server_side_prop = current_time - server_client # we should not cosider images saving time in propagation time calculations
            process_time = json_response['proc_time'] * 1000
            client_side_prop = json_response['clientsideprop']
            prop_time = client_side_prop + server_side_prop

            return {
            'time': time.time(),
            'cid': cid,
            'model_name': version,
            'file_name': item,
            'propagation_delay (s)': prop_time,
            'processing_delay (ms)': process_time,
            'e2e_delay (s)': e2e_delay
        }
        else:
            print('Unsuccessful response!')
            return None 


        # store the values in a dataframe or a csv or a pandas dataframe

