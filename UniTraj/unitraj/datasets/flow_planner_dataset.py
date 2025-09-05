import os
import pickle
import shutil
from collections import defaultdict
from multiprocessing import Pool
import h5py
import numpy as np
import torch
from metadrive.scenario.scenario_description import MetaDriveType
from scenarionet.common_utils import read_scenario, read_dataset_summary
from torch.utils.data import Dataset
from tqdm import tqdm

from unitraj.datasets import common_utils
from unitraj.datasets.common_utils import get_polyline_dir, find_true_segments, generate_mask, is_ddp, \
    get_kalman_difficulty_multi_agent, get_trajectory_type_multi_agent, interpolate_polyline, \
    get_actions_multi_agent, wrap_angle
from unitraj.datasets.types import object_type, polyline_type
from unitraj.utils.visualization import check_loaded_data
from functools import lru_cache

from .base_dataset import BaseDataset

default_value = 0
object_type = defaultdict(lambda: default_value, object_type)
polyline_type = defaultdict(lambda: default_value, polyline_type)


class FlowPlannerDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        self.scene_centric = True
        super().__init__(config, is_validation)

    def load_data(self):
        self.data_loaded = {}
        if self.is_validation:
            print('Loading validation data...')
        else:
            print('Loading training data...')

        for cnt, data_path in enumerate(self.data_path):
            phase, dataset_name = data_path.split('/')[-2],data_path.split('/')[-1]
            observe_type = self.config.get('observe_type', None)
            predict_type = self.config.get('predict_type', None)
            self.cache_path = os.path.join(
                self.config['cache_path'], 
                '' if observe_type is None and predict_type is None else f'observe_{observe_type}-predict_{predict_type}',
                f"{self.config['max_num_agents']}_agents-{self.config['action_len']}_action_len", 
                dataset_name, phase
                )

            data_usage_this_dataset = self.config['max_data_num'][cnt]
            self.starting_frame = self.config['starting_frame'][cnt]
            if self.config['use_cache'] or is_ddp():
                file_list = self.get_data_list(data_usage_this_dataset)
            else:
                if os.path.exists(self.cache_path) and self.config.get('overwrite_cache', False) is False:
                    print('Warning: cache path {} already exists, skip '.format(self.cache_path))
                    file_list = self.get_data_list(data_usage_this_dataset)
                else:
                    # summary_list: [pkl_path]; mapping: pkl_path: bucket id
                    _, summary_list, mapping = read_dataset_summary(data_path, check_file_existence=self.config['check_file_existence'])

                    if os.path.exists(self.cache_path):
                        shutil.rmtree(self.cache_path)
                    os.makedirs(self.cache_path, exist_ok=True)
                    if self.config['debug']:
                        process_num = 1
                    else:
                        process_num = os.cpu_count()//2
                    print('Using {} processes to load data...'.format(process_num))

                    data_splits = np.array_split(summary_list, process_num)

                    data_splits = [(data_path, mapping, list(data_splits[i]), dataset_name) for i in range(process_num)]
                    # save the data_splits in a tmp directory
                    os.makedirs('tmp', exist_ok=True)
                    for i in range(process_num):
                        with open(os.path.join('tmp', '{}.pkl'.format(i)), 'wb') as f:
                            pickle.dump(data_splits[i], f)

                    # results = self.process_data_chunk(0)
                    with Pool(processes=process_num) as pool:
                        results = pool.map(self.process_data_chunk, list(range(process_num)))

                    # concatenate the results
                    file_list = {}
                    for result in results:
                        file_list.update(result)

                    with open(os.path.join(self.cache_path, 'file_list.pkl'), 'wb') as f:
                        pickle.dump(file_list, f)

                    data_list = list(file_list.items())
                    np.random.shuffle(data_list)
                    if not self.is_validation:
                        # randomly sample data_usage number of data
                        file_list = dict(data_list[:data_usage_this_dataset])

            print('Loaded {} samples from {}'.format(len(file_list), data_path))
            self.data_loaded.update(file_list)

            if self.config['store_data_in_memory']:
                print('Loading data into memory...')
                for data_path in file_list.keys():
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    self.data_loaded_memory.append(data)
                print('Loaded {} data into memory'.format(len(self.data_loaded_memory)))

        self.data_loaded_keys = list(self.data_loaded.keys())
        print('Data loaded')

    def process_data_chunk(self, worker_index):
        with open(os.path.join("tmp", "{}.pkl".format(worker_index)), "rb") as f:
            data_chunk = pickle.load(f)
        file_list = {}
        data_path, mapping, data_list, dataset_name = data_chunk
        hdf5_path = os.path.join(self.cache_path, f"{worker_index}.h5")

        with h5py.File(hdf5_path, 'w') as f:
            for cnt, file_name in enumerate(data_list):
                if worker_index == 0 and cnt % max(int(len(data_list) / 10), 1) == 0:
                    print(f'{cnt}/{len(data_list)} data processed', flush=True)
                scenario = read_scenario(data_path, mapping, file_name, centralize=self.config['centralize']) # TODO

                try:
                    output = self.preprocess(scenario)

                    output = self.process(output)

                    output = self.postprocess(output)

                except Exception as e:
                    print('Warning: {} in {}'.format(e, file_name))
                    output = None

                if output is None: continue
                
                # TODO: verify the file saving logic is correct
                for i, record in enumerate(output):
                    grp_name = (
                        dataset_name
                        + "-"
                        + str(worker_index)
                        + "-"
                        + str(cnt)
                        + "-"
                        + str(i)
                    )
                    grp = f.create_group(grp_name)
                    for key, value in record.items():
                        if isinstance(value, str):
                            value = np.bytes_(value)
                        grp.create_dataset(key, data=value)
                    file_info = {}
                    kalman_difficulty = np.stack([x['kalman_difficulty_multi_agent'] for x in output])
                    file_info['kalman_difficulty_multi_agent'] = kalman_difficulty
                    file_info['h5_path'] = hdf5_path
                    file_list[grp_name] = file_info
                del scenario
                del output

        return file_list

    def preprocess(self, scenario):
        traffic_lights = scenario['dynamic_map_states']
        tracks = scenario['tracks']                        # {id: {type:['P','C','V'], state:{...}, metadata:{...}}}
        map_feat = scenario['map_features']

        past_length = self.config['past_len']
        future_length = self.config['future_len']
        total_steps = past_length + future_length
        starting_fame = self.starting_frame
        ending_fame = starting_fame + total_steps
        trajectory_sample_interval = self.config['trajectory_sample_interval']
        frequency_mask = generate_mask(past_length - 1, total_steps, trajectory_sample_interval)

        track_infos = {
            'object_id': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            'object_type': [],
            'trajs': []
        }

        for k, v in tracks.items(): # for each agent in the scenario

            state = v["state"]
            for key, value in state.items():
                if len(value.shape) == 1:
                    state[key] = np.expand_dims(value, axis=-1)
            all_state = [
                state["position"],
                state["length"],
                state["width"],
                state["height"],
                wrap_angle(state["heading"]),
                state["velocity"],
                state["valid"],
            ]
            # type, x,y,z,l,w,h,heading,vx,vy,valid
            all_state = np.concatenate(all_state, axis=-1)

            if all_state.shape[0] < ending_fame:
                all_state = np.pad(
                    all_state, ((ending_fame - all_state.shape[0], 0), (0, 0))
                )
            all_state = all_state[starting_fame:ending_fame]

            assert all_state.shape[0] == total_steps, f'Error: {all_state.shape[0]} != {total_steps}'

            track_infos['object_id'].append(k)
            track_infos['object_type'].append(object_type[v['type']])
            track_infos['trajs'].append(all_state)

        track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)   # (A, T, S)
        # scenario['metadata']['ts'] = scenario['metadata']['ts'][::sample_inverval]
        track_infos['trajs'][..., -1] *= frequency_mask[np.newaxis]
        scenario['metadata']['ts'] = scenario['metadata']['ts'][:total_steps]

        # x,y,z,type
        map_infos = {
            'lane': [],
            'road_line': [],
            'road_edge': [],
            'stop_sign': [],
            'crosswalk': [],
            'speed_bump': [],
        }
        polylines = []
        point_cnt = 0
        for k, v in map_feat.items():
            polyline_type_ = polyline_type[v['type']]
            if polyline_type_ == 0:
                continue

            cur_info = {'id': k}
            cur_info['type'] = v['type']
            if polyline_type_ in [1, 2, 3]:
                cur_info['speed_limit_mph'] = v.get('speed_limit_mph', None)
                cur_info['interpolating'] = v.get('interpolating', None)
                cur_info['entry_lanes'] = v.get('entry_lanes', None)
                if 'is_sdc_route' in v.keys():
                    cur_info['is_sdc_route'] = v['is_sdc_route']
                else:
                    cur_info['is_sdc_route'] = False
                try:  # ? this is probably inconsistent after interpolating each polyline
                    cur_info['left_boundary'] = [
                        {
                        'start_index': x['self_start_index'], 
                        'end_index': x['self_end_index'],
                        'feature_id': x['feature_id'],
                        'boundary_type': 'UNKNOWN'  # roadline type
                        } for x in v['left_neighbor']
                    ]
                    cur_info['right_boundary'] = [
                        {
                        'start_index': x['self_start_index'], 
                        'end_index': x['self_end_index'],
                        'feature_id': x['feature_id'],
                        'boundary_type': 'UNKNOWN'  # roadline type
                        } for x in v['right_neighbor']
                    ]
                except:
                    cur_info['left_boundary'] = []
                    cur_info['right_boundary'] = []
                polyline = v['polyline']
                polyline = interpolate_polyline(polyline)
                map_infos['lane'].append(cur_info)
            elif polyline_type_ in [6, 7, 8, 9, 10, 11, 12, 13]:
                try:
                    polyline = v['polyline']
                except:
                    polyline = v['polygon']
                polyline = interpolate_polyline(polyline)
                map_infos['road_line'].append(cur_info)
            elif polyline_type_ in [15, 16]:
                polyline = v['polyline']
                polyline = interpolate_polyline(polyline)
                cur_info['type'] = 7
                map_infos['road_line'].append(cur_info)
            elif polyline_type_ in [17]:    # stopsign
                cur_info['lane_ids'] = v['lane']
                cur_info['position'] = v['position']
                map_infos['stop_sign'].append(cur_info)
                polyline = v['position'][np.newaxis]
            elif polyline_type_ in [18]:
                map_infos['crosswalk'].append(cur_info)
                polyline = v['polygon']
            elif polyline_type_ in [19]:
                map_infos['crosswalk'].append(cur_info)
                polyline = v['polygon']
            if polyline.shape[-1] == 2:
                polyline = np.concatenate((polyline, np.zeros((polyline.shape[0], 1))), axis=-1)
            try:
                cur_polyline_dir = get_polyline_dir(polyline)       # (N_p, 3)
                type_array = np.zeros([polyline.shape[0], 1])       
                type_array[:] = polyline_type_                      # (N_p, 1)
                cur_polyline = np.concatenate((polyline, cur_polyline_dir, type_array), axis=-1) # (N_p, 7)
            except:
                cur_polyline = np.zeros((0, 7), dtype=np.float32)
            polylines.append(cur_polyline)
            cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
            point_cnt += len(cur_polyline)

        try:
            polylines = np.concatenate(polylines, axis=0).astype(np.float32)
        except:
            polylines = np.zeros((0, 7), dtype=np.float32)
        map_infos['all_polylines'] = polylines

        dynamic_map_infos = {
            'lane_id': [],
            'state': [],
            'stop_point': []
        }
        for k, v in traffic_lights.items():  # (num_timestamp)
            lane_id, state, stop_point = [], [], []
            for cur_signal in v['state']['object_state']:  # (num_observed_signals)
                lane_id.append(str(v['lane']))
                state.append(cur_signal)
                if type(v['stop_point']) == list:
                    stop_point.append(v['stop_point'])
                else:
                    stop_point.append(v['stop_point'].tolist())
            # lane_id = lane_id[::sample_inverval]
            # state = state[::sample_inverval]
            # stop_point = stop_point[::sample_inverval]
            lane_id = lane_id[:total_steps]
            state = state[:total_steps]
            stop_point = stop_point[:total_steps]
            dynamic_map_infos['lane_id'].append(np.array([lane_id]))
            dynamic_map_infos['state'].append(np.array([state]))
            dynamic_map_infos['stop_point'].append(np.array([stop_point]))

        ret = {
            'track_infos': track_infos,
            'dynamic_map_infos': dynamic_map_infos,
            'map_infos': map_infos
        }
        ret.update(scenario['metadata'])
        ret['timestamps_seconds'] = ret.pop('ts')
        ret['current_time_index'] = self.config['past_len'] - 1
        ret['sdc_track_index'] = track_infos['object_id'].index(ret['sdc_id'])

        if self.config['predict_type'] == 'ego': 
            tracks_to_predict = {
                'track_index': [ret['sdc_track_index']],
                'difficulty': [0],
                'object_type': [MetaDriveType.VEHICLE]
            }
        elif self.config['predict_type'] == 'all':
            filtered_tracks = self.trajectory_filter(ret)
            sample_list = list(filtered_tracks.keys())
            tracks_to_predict = {
                'track_index': [track_infos['object_id'].index(id) for id in sample_list if
                                id in track_infos['object_id']],
                'object_type': [track_infos['object_type'][track_infos['object_id'].index(id)] for id in sample_list if
                                id in track_infos['object_id']],
            }
        elif self.config['predict_type'] == 'interested':
            
            sample_list = list(ret['tracks_to_predict'].keys())  # + ret.get('objects_of_interest', [])
            sample_list = list(set(sample_list))
            tracks_to_predict = {
                'track_index': [track_infos['object_id'].index(id) for id in sample_list if
                                id in track_infos['object_id']],
                'object_type': [track_infos['object_type'][track_infos['object_id'].index(id)] for id in sample_list if
                                id in track_infos['object_id']],
            }
        else:
            assert False, f"predict type {self.config['predict_type']} is not supported"

        if self.config['observe_type'] == 'ego':
            tracks_to_predict['observe_frame_index'] = [ret['sdc_track_index']]
        elif self.config['observe_type'] == 'interested':
            sample_list = list(ret['tracks_to_predict'].keys())
            sample_list = list(set(sample_list))
            tracks_to_predict['observe_frame_index'] = [track_infos['object_id'].index(id) for id in sample_list if
                                                        id in track_infos['object_id'] and track_infos['object_type'][track_infos['object_id'].index(id)]==MetaDriveType.VEHICLE]

        elif self.config['observe_type'] == 'all':
            filtered_tracks = self.trajectory_filter(ret)
            sample_list = list(filtered_tracks.keys())
            tracks_to_predict['observe_frame_index'] = [track_infos['object_id'].index(id) for id in sample_list if
                                                        id in track_infos['object_id'] and track_infos['object_type'][track_infos['object_id'].index(id)]==MetaDriveType.VEHICLE]

        ret['tracks_to_predict'] = tracks_to_predict

        ret['map_center'] = scenario['metadata'].get('map_center', np.zeros(3))[np.newaxis]

        ret['track_length'] = total_steps
        return ret

    def process(self, internal_format):

        info = internal_format
        scene_id = info['scenario_id']

        sdc_track_index = info['sdc_track_index']
        current_time_index = info['current_time_index']
        timestamps = np.array(info['timestamps_seconds'][:current_time_index + 1], dtype=np.float32)

        track_infos = info['track_infos']

        track_index_to_predict_wo_ego = np.array(info['tracks_to_predict']['track_index'])
        if sdc_track_index in track_index_to_predict_wo_ego:
            track_index_to_predict = track_index_to_predict_wo_ego
        else: 
            track_index_to_predict = np.append(track_index_to_predict_wo_ego, sdc_track_index)

        observe_frame_index_wo_ego = np.array(info['tracks_to_predict']['observe_frame_index'])
        if sdc_track_index in observe_frame_index_wo_ego:
            observe_frame_index = observe_frame_index_wo_ego
        else: 
            observe_frame_index = np.append(observe_frame_index_wo_ego, sdc_track_index)

        obj_types = np.array(track_infos['object_type'])
        obj_trajs_full = track_infos['trajs']  # (num_objects, num_timestamp, 10)
        obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]     # (N_a, T_h, S)
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]   # (N_a, T_f, S)

        # # further filter out interested agents based on type and vaild
        # center_objects, track_index_to_predict = self.get_interested_agents(
        #     track_index_to_predict=track_index_to_predict,
        #     obj_trajs_full=obj_trajs_full,
        #     current_time_index=current_time_index,
        #     obj_types=obj_types, scene_id=scene_id
        # )  # (N_ia, 10), (N_ia)

        observe_objects, observe_frame_index = self.get_observed_agents(
            observe_frame_index=observe_frame_index,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            scene_id=scene_id
        )

        if observe_objects is None: return None      # (N-obj, S)

        sample_num = observe_objects.shape[0]


        (
            obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, 
            obj_trajs_future_state, obj_trajs_future_mask, obj_trajs_future_final_valid_idx,
            track_index_to_predict,
        ) = self.get_agent_data_in_ego_frame(
            center_objects=observe_objects, 
            obj_trajs_past=obj_trajs_past, 
            obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict,
            sdc_track_index=sdc_track_index,
            timestamps=timestamps, 
            obj_types=obj_types,
        )

        ret_dict = {
            'scenario_id': np.array([scene_id]),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict,  # used to select center-features
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,

            'center_objects_world': observe_objects,
            'center_objects_id': np.array(track_infos['object_id'])[observe_frame_index],
            'center_objects_type': np.array(track_infos['object_type'])[observe_frame_index],
            'map_center': info['map_center'],

            'obj_trajs_future_state': obj_trajs_future_state,
            'obj_trajs_future_mask': obj_trajs_future_mask,
            # 'center_gt_trajs': center_gt_trajs,
            # 'center_gt_trajs_mask': center_gt_trajs_mask,
            'obj_trajs_future_final_valid_idx': obj_trajs_future_final_valid_idx,
            # 'center_gt_trajs_src': obj_trajs_full[track_index_to_predict]
        }

        if info['map_infos']['all_polylines'].__len__() == 0:
            info['map_infos']['all_polylines'] = np.zeros((2, 7), dtype=np.float32)
            print(f'Warning: empty HDMap {scene_id}')

        if self.config.manually_split_lane:
            assert False
            map_polylines_data, map_polylines_mask, map_polylines_center = self.get_manually_split_map_data(
                center_objects=observe_objects, map_infos=info['map_infos'])
        else:
            map_polylines_data, map_polylines_mask, map_polylines_center, map_route_lanes, map_route_lanes_mask = self.get_map_data(
                center_objects=observe_objects, map_infos=info['map_infos'])

        ret_dict['map_polylines'] = map_polylines_data
        ret_dict['map_polylines_mask'] = map_polylines_mask.astype(bool)
        ret_dict['map_polylines_center'] = map_polylines_center
        ret_dict['map_route_lanes'] = map_route_lanes
        ret_dict['map_route_lanes_mask'] = map_route_lanes_mask.astype(bool)

        # masking out unused attributes to Zero
        masked_attributes = self.config['masked_attributes']
        if 'z_axis' in masked_attributes:
            ret_dict['obj_trajs'][..., 2] = 0
            ret_dict['map_polylines'][..., 2] = 0
        if 'size' in masked_attributes:
            ret_dict['obj_trajs'][..., 3:6] = 0
        if 'velocity' in masked_attributes:
            ret_dict['obj_trajs'][..., 25:27] = 0
        if 'acceleration' in masked_attributes:
            ret_dict['obj_trajs'][..., 27:29] = 0
        if 'heading' in masked_attributes:
            ret_dict['obj_trajs'][..., 23:25] = 0

        # change every thing to float32
        for k, v in ret_dict.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float64:
                ret_dict[k] = v.astype(np.float32)

        ret_dict['map_center'] = ret_dict['map_center'].repeat(sample_num, axis=0)
        ret_dict['dataset_name'] = [info['dataset']] * sample_num

        ret_list = []
        for i in range(sample_num):
            ret_dict_i = {}
            for k, v in ret_dict.items():
                ret_dict_i[k] = v[i]
            ret_list.append(ret_dict_i)

        return ret_list

    def postprocess(self, output):

        # Add the trajectory difficulty
        get_kalman_difficulty_multi_agent(output)

        # Add the trajectory type (stationary, straight, right turn...)
        get_trajectory_type_multi_agent(output)

        # Add gt actions (setting 1: (vx, vy, yaw_rate), setting 2: (acc, steer))
        get_actions_multi_agent(output, action_len=self.config['action_len'], dt=self.config['dt'])

        return output

    def get_agent_data_in_ego_frame(
            self, 
            center_objects, 
            obj_trajs_past, 
            obj_trajs_future, 
            track_index_to_predict,
            sdc_track_index, 
            timestamps,
            obj_types
    ):

        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )   # (N_selected_agents, N_a, T_h, S)

        object_onehot_mask = np.zeros((num_center_objects, num_objects, num_timestamps, 4)) # mark agenet type, track_to_predict, ego_index
        object_onehot_mask[:, obj_types == 1, :, 0] = 1
        object_onehot_mask[:, obj_types == 2, :, 1] = 1
        object_onehot_mask[:, obj_types == 3, :, 2] = 1
        object_onehot_mask[:, sdc_track_index, :, 3] = 1

        object_time_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1)) # (N_sa, N_a, T_h, T_h+1)
        for i in range(num_timestamps):
            object_time_embedding[:, :, i, i] = 1
        object_time_embedding[:, :, :, -1] = timestamps

        object_heading_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
        object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

        vel = obj_trajs[:, :, :, 7:9]           # (N_sa, N_a, T_h, 2)
        vel_pre = np.roll(vel, shift=1, axis=2)
        acce = (vel - vel_pre) / 0.1
        acce[:, :, 0, :] = acce[:, :, 1, :]     # (N_sa, N_a, T_h, 2)

        obj_trajs_data = np.concatenate([
            obj_trajs[:, :, :, 0:6],    # 6
            object_onehot_mask,         # 4
            object_time_embedding,      # 21 because T_h=20, this can change for different experiment setting
            object_heading_embedding,   # 2
            obj_trajs[:, :, :, 7:9],    # 2
            acce,                       # 2
        ], axis=-1)  # (N_na, N_a, T_h, 37)

        # print([col.nonzero()[0] for col in obj_trajs_data[:, :, -1, 9]])

        obj_trajs_mask = obj_trajs[:, :, :, -1]     # (N_sa, N_a, T_h)
        obj_trajs_data[obj_trajs_mask == 0] = 0.

        obj_trajs_future = obj_trajs_future.astype(np.float32)
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )   # (N_selected_agents, N_a, T_f, S)
        obj_trajs_future_state = obj_trajs_future[:, :, :, [0, 1, 6, 7, 8, 3, 4, 5]]  # (x, y, heading, vx, vy, l, w, h)
        obj_trajs_future_mask = obj_trajs_future[:, :, :, -1]
        obj_trajs_future_state[obj_trajs_future_mask == 0] = 0.

        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(obj_trajs_past[:, -1, -1] == 0) # (N_a,)
        interested_mask = np.zeros(valid_past_mask.shape, dtype=bool)
        interested_mask[track_index_to_predict] = True
        interested_mask = interested_mask[valid_past_mask]
        track_index_to_predict = np.where(interested_mask)[0]

        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask] # (N_sa, Num_agents_valid_at_curr_t, T_h, )
        obj_trajs_data = obj_trajs_data[:, valid_past_mask] # (N_sa, Num_a_at_curr_t, T_h, 37)
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask] # (N_sa, Num_agents_valid_at_curr_t, T_f, 37)
        obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]   # (N_sa, Num_agents_valid_at_curr_t, T_f, )

        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)   # (N_sa, Num_a_at_curr_t, 3)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

        max_num_agents = self.config['max_num_agents']
        object_dist_to_center = np.linalg.norm(obj_trajs_data[:, :, -1, 0:2], axis=-1)

        object_dist_to_center[obj_trajs_mask[..., -1] == 0] = 1e10
        topk_idxs = np.argsort(object_dist_to_center, axis=-1)[:, :max_num_agents] # (N_sa, min(N_a_at_curr_t, N_max_agent))
        interested_mask = topk_idxs[:, None, :] == track_index_to_predict[None, :, None]
        positions = np.argmax(interested_mask, axis=-1)
        found_mask = np.any(interested_mask, axis=-1)
        track_index_to_predict = positions[found_mask]

        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)

        obj_trajs_data = np.take_along_axis(obj_trajs_data, topk_idxs, axis=1)
        obj_trajs_mask = np.take_along_axis(obj_trajs_mask, topk_idxs[..., 0], axis=1)
        obj_trajs_pos = np.take_along_axis(obj_trajs_pos, topk_idxs, axis=1)
        obj_trajs_last_pos = np.take_along_axis(obj_trajs_last_pos, topk_idxs[..., 0], axis=1)
        obj_trajs_future_state = np.take_along_axis(obj_trajs_future_state, topk_idxs, axis=1)
        obj_trajs_future_mask = np.take_along_axis(obj_trajs_future_mask, topk_idxs[..., 0], axis=1)
        
        obj_trajs_data = np.pad(obj_trajs_data, ((0, 0), (0, max_num_agents - obj_trajs_data.shape[1]), (0, 0), (0, 0)))
        obj_trajs_mask = np.pad(obj_trajs_mask, ((0, 0), (0, max_num_agents - obj_trajs_mask.shape[1]), (0, 0)))
        obj_trajs_pos = np.pad(obj_trajs_pos, ((0, 0), (0, max_num_agents - obj_trajs_pos.shape[1]), (0, 0), (0, 0)))
        obj_trajs_last_pos = np.pad(obj_trajs_last_pos,
                                    ((0, 0), (0, max_num_agents - obj_trajs_last_pos.shape[1]), (0, 0)))
        obj_trajs_future_state = np.pad(obj_trajs_future_state,
                                        ((0, 0), (0, max_num_agents - obj_trajs_future_state.shape[1]), (0, 0), (0, 0)))
        obj_trajs_future_mask = np.pad(obj_trajs_future_mask,
                                       ((0, 0), (0, max_num_agents - obj_trajs_future_mask.shape[1]), (0, 0)))

        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_future_state.shape
        obj_trajs_future_final_valid_idx = np.ones((num_center_objects, num_objects), dtype=np.int64) * -1  # (N_sa, Num_a_at_curr_t)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_future_mask[:, :, k] > 0
            obj_trajs_future_final_valid_idx[cur_valid_mask] = k
        return (obj_trajs_data, obj_trajs_mask.astype(bool), obj_trajs_pos, obj_trajs_last_pos,
                obj_trajs_future_state, obj_trajs_future_mask, obj_trajs_future_final_valid_idx,
                track_index_to_predict)

    def get_observed_agents(self, observe_frame_index, obj_trajs_full, current_time_index, scene_id):
        observe_objects_list = []
        observe_frame_index_selected = []

        for k in range(len(observe_frame_index)):
            obj_idx = observe_frame_index[k]

            if obj_trajs_full[obj_idx, current_time_index, -1] == 0:
                print(f'Warning: obj_idx={obj_idx} is not valid at time step {current_time_index}, scene_id={scene_id}')
                continue

            observe_objects_list.append(obj_trajs_full[obj_idx, current_time_index])  # current state of the object
            observe_frame_index_selected.append(obj_idx)
        if len(observe_objects_list) == 0:
            print(f'Warning: no center objects at time step {current_time_index}, scene_id={scene_id}')
            return None, []
        observe_objects = np.stack(observe_objects_list, axis=0) 
        observe_frame_index = np.array(observe_frame_index_selected)
        return observe_objects, observe_frame_index

    def get_map_data(self, center_objects, map_infos):

        num_center_objects = center_objects.shape[0]

        def transform_to_center_coordinates(neighboring_polylines):
            neighboring_polylines[:, :, 0:3] -= center_objects[:, None, 0:3]
            neighboring_polylines[:, :, 0:2] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, 0:2],
                angle=-center_objects[:, 6]
            )
            neighboring_polylines[:, :, 3:5] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, 3:5],
                angle=-center_objects[:, 6]
            )

            return neighboring_polylines

        polylines = np.expand_dims(map_infos['all_polylines'].copy(), axis=0).repeat(num_center_objects, axis=0)

        map_polylines = transform_to_center_coordinates(neighboring_polylines=polylines)
        num_of_src_polylines = self.config['max_num_roads']
        num_of_src_route_lanes = self.config['max_num_route_lanes']
        map_infos['polyline_transformed'] = map_polylines

        all_polylines = map_infos['polyline_transformed']
        max_points_per_lane = self.config.get('max_points_per_lane', 20)
        line_type = self.config.get('line_type', [])
        map_range = self.config.get('map_range', None)
        center_offset = self.config.get('center_offset_of_map', (30.0, 0))
        num_agents = all_polylines.shape[0]
        polyline_list = []
        polyline_mask_list = []
        route_lanes_mask_list = []

        for k, v in map_infos.items():
            if k == 'all_polylines' or k not in line_type:
                continue
            if len(v) == 0:
                continue
            for polyline_dict in v:
                polyline_index = polyline_dict.get('polyline_index', None)
                polyline_segment = all_polylines[:, polyline_index[0]:polyline_index[1]]
                polyline_segment_x = polyline_segment[:, :, 0] - center_offset[0]
                polyline_segment_y = polyline_segment[:, :, 1] - center_offset[1]
                in_range_mask = (abs(polyline_segment_x) < map_range) * (abs(polyline_segment_y) < map_range)
                route_lane_mask = polyline_dict.get('is_sdc_route', False)

                segment_index_list = []
                for i in range(polyline_segment.shape[0]):
                    segment_index_list.append(find_true_segments(in_range_mask[i]))
                max_segments = max([len(x) for x in segment_index_list])

                segment_list = np.zeros([num_agents, max_segments, max_points_per_lane, 7], dtype=np.float32)
                segment_mask_list = np.zeros([num_agents, max_segments, max_points_per_lane], dtype=np.int32)
                route_lane_mask_list = np.zeros([num_agents, max_segments, max_points_per_lane], dtype=bool)

                for i in range(polyline_segment.shape[0]):
                    if in_range_mask[i].sum() == 0:
                        continue
                    segment_i = polyline_segment[i]
                    segment_index = segment_index_list[i]
                    for num, seg_index in enumerate(segment_index):
                        segment = segment_i[seg_index]
                        if segment.shape[0] > max_points_per_lane:
                            segment_list[i, num] = segment[
                                np.linspace(0, segment.shape[0] - 1, max_points_per_lane, dtype=int)]
                            segment_mask_list[i, num] = 1
                            route_lane_mask_list[i, num] = route_lane_mask
                        else:
                            segment_list[i, num, :segment.shape[0]] = segment
                            segment_mask_list[i, num, :segment.shape[0]] = 1
                            route_lane_mask_list[i, num, :segment.shape[0]] = route_lane_mask

                polyline_list.append(segment_list)
                polyline_mask_list.append(segment_mask_list)
                route_lanes_mask_list.append(route_lane_mask_list)
        if len(polyline_list) == 0: return np.zeros((num_agents, 0, max_points_per_lane, 7)), np.zeros(
            (num_agents, 0, max_points_per_lane))
        batch_polylines = np.concatenate(polyline_list, axis=1)
        batch_polylines_mask = np.concatenate(polyline_mask_list, axis=1)
        batch_route_lanes_mask = np.concatenate(route_lanes_mask_list, axis=1)

        polyline_xy_offsetted = batch_polylines[:, :, :, 0:2] - np.reshape(center_offset, (1, 1, 1, 2))
        polyline_center_dist = np.linalg.norm(polyline_xy_offsetted, axis=-1).sum(-1) / np.clip(
            batch_polylines_mask.sum(axis=-1).astype(float), a_min=1.0, a_max=None)
        polyline_center_dist[batch_polylines_mask.sum(-1) == 0] = 1e10
        topk_idxs = np.argsort(polyline_center_dist, axis=-1)[:, :num_of_src_polylines]

        # Ensure topk_idxs has the correct shape for indexing
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        map_polylines = np.take_along_axis(batch_polylines, topk_idxs, axis=1)
        map_polylines_mask = np.take_along_axis(batch_polylines_mask, topk_idxs[..., 0], axis=1)
        map_route_lanes_mask = np.take_along_axis(batch_route_lanes_mask, topk_idxs[..., 0], axis=1)

        # extract lane routes 
        lane_routes_mask = np.sum(map_route_lanes_mask, axis=-1) > 0
        map_route_lanes = map_polylines[lane_routes_mask][None, :num_of_src_route_lanes]
        map_route_lanes_mask = map_route_lanes_mask[lane_routes_mask][None, :num_of_src_route_lanes]
        map_route_lanes = np.pad(map_route_lanes,
                               ((0, 0), (0, num_of_src_route_lanes - map_route_lanes.shape[1]), (0, 0), (0, 0)))
        map_route_lanes_mask = np.pad(map_route_lanes_mask,
                                    ((0, 0), (0, num_of_src_route_lanes - map_route_lanes_mask.shape[1]), (0, 0)))

        # pad map_polylines and map_polylines_mask to num_of_src_polylines
        map_polylines = np.pad(map_polylines,
                               ((0, 0), (0, num_of_src_polylines - map_polylines.shape[1]), (0, 0), (0, 0)))
        map_polylines_mask = np.pad(map_polylines_mask,
                                    ((0, 0), (0, num_of_src_polylines - map_polylines_mask.shape[1]), (0, 0)))

        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].astype(float)).sum(
            axis=-2)  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / np.clip(map_polylines_mask.sum(axis=-1).astype(float)[:, :, None], a_min=1.0,
                                                  a_max=None)  # (num_center_objects, num_polylines, 3)

        xy_pos_pre = map_polylines[:, :, :, 0:3]
        xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]

        map_types = map_polylines[:, :, :, -1]
        map_polylines = map_polylines[:, :, :, :-1]
        # one-hot encoding for map types, 14 types in total, use 20 for reserved types
        map_types = np.eye(20)[map_types.astype(int)]

        map_polylines = np.concatenate((map_polylines, xy_pos_pre, map_types), axis=-1)
        map_polylines[map_polylines_mask == 0] = 0

        return map_polylines, map_polylines_mask, map_polylines_center, map_route_lanes, map_route_lanes_mask
