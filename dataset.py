import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)
from util import config, file_dir

from datetime import datetime
import numpy as np
import arrow
import metpy.calc as mpcalc
from metpy.units import units
from torch.utils import data


class HazeData(data.Dataset):

    def __init__(self, graph,
                       hist_len=1,
                       pred_len=24,
                       dataset_num=1,
                       flag='Train',
                       ):

        if flag == 'Train':
            start_time_str = 'train_start'
            end_time_str = 'train_end'
        elif flag == 'Val':
            start_time_str = 'val_start'
            end_time_str = 'val_end'
        elif flag == 'Test':
            start_time_str = 'test_start'
            end_time_str = 'test_end'
        else:
            raise Exception('Wrong Flag!')

        #! config에서 날짜 가져오기
        self.start_time = self._get_time(config['dataset'][dataset_num][start_time_str])
        self.end_time = self._get_time(config['dataset'][dataset_num][end_time_str])
        self.data_start = self._get_time(config['dataset']['data_start'])
        self.data_end = self._get_time(config['dataset']['data_end'])

        self.knowair_fp = file_dir['knowair_fp']

        self.graph = graph

        self._load_npy(),        #! self.knowair, self.feature, self.pm25 생성
        self._gen_time_arr()     #! self.time_arrow, self.time_arr 생성
        self._process_time()
        self._process_feature()
        self.feature = np.float32(self.feature)
        self.pm25 = np.float32(self.pm25)
        self._calc_mean_std()
        seq_len = hist_len + pred_len
        self._add_time_dim(seq_len)
        self._norm()

    def _norm(self):
        self.feature = (self.feature - self.feature_mean) / self.feature_std
        self.pm25 = (self.pm25 - self.pm25_mean) / self.pm25_std

    def _add_time_dim(self, seq_len):

        def _add_t(arr, seq_len):
            t_len = arr.shape[0]        #! 변수 길이
            assert t_len > seq_len      #! 전체 길이가 seq_len보다 길어야함
            arr_ts = []
            for i in range(seq_len, t_len):         #! 처음부터 끝까지 1 간격으로 이동하면서 학습 batch 생성(stack)
                arr_t = arr[i-seq_len:i]
                arr_ts.append(arr_t)
            arr_ts = np.stack(arr_ts, axis=0)
            return arr_ts

        self.pm25 = _add_t(self.pm25, seq_len)      #! 위 함수 통해서 train:test 로 묶인 행렬 생성
        self.feature = _add_t(self.feature, seq_len)
        self.time_arr = _add_t(self.time_arr, seq_len)

    def _calc_mean_std(self):
        self.feature_mean = self.feature.mean(axis=(0,1))
        self.feature_std = self.feature.std(axis=(0,1))
        self.wind_mean = self.feature_mean[-2:]
        self.wind_std = self.feature_std[-2:]
        self.pm25_mean = self.pm25.mean()
        self.pm25_std = self.pm25.std()

    def _process_feature(self):                             #! graph 데이터로부터 feature 생성
        metero_var = config['data']['metero_var']           #! 전체 변수
        metero_use = config['experiments']['metero_use']    #! 활용 변수
        metero_idx = [metero_var.index(var) for var in metero_use]      #! 활용 변수중 활용하는 변수 column만
        self.feature = self.feature[:,:,metero_idx]         #! 활용하는 변수만 추출

        u = self.feature[:, :, -2] * units.meter / units.second #! 'u_component_of_wind+950'    #! 단위 
        v = self.feature[:, :, -1] * units.meter / units.second #! 'v_component_of_wind+950'
        speed = 3.6 * mpcalc.wind_speed(u, v)._magnitude        #! 풍속 계산 (필요?) 삭제
        direc = mpcalc.wind_direction(u, v)._magnitude          #! 풍향 계산 (필요?) 삭제

        h_arr = []
        w_arr = []
        for i in self.time_arrow:
            print(f" dataset_process_feature i : {i}")
            h_arr.append(i.hour)
            print(f" dataset_process_feature h_arr : {i.hour}")
            w_arr.append(i.isoweekday())
            print(f" dataset_process_feature w_arr : {i.isoweekday()}")
        h_arr = np.stack(h_arr, axis=-1)
        w_arr = np.stack(w_arr, axis=-1)
        h_arr = np.repeat(h_arr[:, None], self.graph.node_num, axis=1)      #! axis 추가 (newaxis object)
        w_arr = np.repeat(w_arr[:, None], self.graph.node_num, axis=1)

        self.feature = np.concatenate([self.feature, h_arr[:, :, None], w_arr[:, :, None],
                                       speed[:, :, None], direc[:, :, None]
                                       ], axis=-1)          #! 쌓아서 반환

    def _process_time(self):
        start_idx = self._get_idx(self.start_time)
        end_idx = self._get_idx(self.end_time)
        self.pm25 = self.pm25[start_idx: end_idx+1, :]
        self.feature = self.feature[start_idx: end_idx+1, :]
        self.time_arr = self.time_arr[start_idx: end_idx+1]
        self.time_arrow = self.time_arrow[start_idx: end_idx + 1]

    def _gen_time_arr(self):            #! 여기서는 필요없어서 안쓰는듯?
        self.time_arrow = []
        self.time_arr = []
        for time_arrow in arrow.Arrow.interval('hour', self.data_start, self.data_end.shift(hours=+3), 3):
            print('time_arrow : ', time_arrow)
            self.time_arrow.append(time_arrow[0])
            self.time_arr.append(time_arrow[0].timestamp)
        self.time_arr = np.stack(self.time_arr, axis=-1)

    def _load_npy(self):
        self.knowair = np.load(self.knowair_fp)     #! 데이터 불러오기
        self.feature = self.knowair[:,:,:-1]        #! 마지막 열 전까지는 기상 데이터
        self.pm25 = self.knowair[:,:,-1:]           #! 맨 마지막 열 값이 pm25 값

    def _get_idx(self, t):                          #! 데이터 시작 날짜와 비교해서 index 도출 후 반환
        t0 = self.data_start
        print(t.timestamp())
        print(t0.timestamp())
        return int((t.timestamp() - t0.timestamp()) / (60 * 60 * 3))

    def _get_time(self, time_yaml):
        arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1])
        return arrow_time

    def __len__(self):
        return len(self.pm25)

    def __getitem__(self, index):
        # print('self.pm25[index] : ', self.pm25[index])
        # print('self.feature[index] : ', self.feature[index])
        # print('self.time_arr[index] : ', self.time_arr[index])
        # print('self.time_arr[index][0] : ', self.time_arr[index][0].timestamp())
        
        return self.pm25[index], self.feature[index], 1 #, self.time_arr[index]

if __name__ == '__main__':
    from graph import Graph
    graph = Graph()
    '''
    하나의 데이터에서 세개 분리
    '''
    train_data = HazeData(graph, flag='Train')
    val_data = HazeData(graph, flag='Val')
    test_data = HazeData(graph, flag='Test')

    print(len(train_data))
    print(len(val_data))
    print(len(test_data))
