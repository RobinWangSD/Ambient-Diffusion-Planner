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


class FlowDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        self.scene_centric = True
        super().__init__(config, is_validation)

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

        # further filter out interested agents based on type and vaild
        center_objects, track_index_to_predict = self.get_interested_agents(
            track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            obj_types=obj_types, scene_id=scene_id
        )  # (N_ia, 10), (N_ia)

        observe_objects, observe_frame_index = self.get_observed_agents(
            observe_frame_index=observe_frame_index,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            scene_id=scene_id
        )

        if center_objects is None or observe_objects is None: return None      # (N-obj, S)

        sample_num = observe_objects.shape[0]


        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state,
         obj_trajs_future_mask, center_gt_trajs,
         center_gt_trajs_mask, center_gt_final_valid_idx,
         track_index_to_predict_new) = self.get_agent_data_in_ego_frame(
            center_objects=observe_objects, obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict, sdc_track_index=sdc_track_index,
            timestamps=timestamps, obj_types=obj_types
        )

        ret_dict = {
            'scenario_id': np.array([scene_id] * len(track_index_to_predict)),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict_new,  # used to select center-features
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,

            'center_objects_world': observe_objects,
            'center_objects_id': np.array(track_infos['object_id'])[observe_frame_index],
            'center_objects_type': np.array(track_infos['object_type'])[observe_frame_index],
            'map_center': info['map_center'],

            'obj_trajs_future_state': obj_trajs_future_state,
            'obj_trajs_future_mask': obj_trajs_future_mask,
            'center_gt_trajs': center_gt_trajs,
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_src': obj_trajs_full[track_index_to_predict]
        }

        if info['map_infos']['all_polylines'].__len__() == 0:
            info['map_infos']['all_polylines'] = np.zeros((2, 7), dtype=np.float32)
            print(f'Warning: empty HDMap {scene_id}')

        if self.config.manually_split_lane:
            map_polylines_data, map_polylines_mask, map_polylines_center = self.get_manually_split_map_data(
                center_objects=observe_objects, map_infos=info['map_infos'])
        else:
            map_polylines_data, map_polylines_mask, map_polylines_center = self.get_map_data(
                center_objects=observe_objects, map_infos=info['map_infos'])

        ret_dict['map_polylines'] = map_polylines_data
        ret_dict['map_polylines_mask'] = map_polylines_mask.astype(bool)
        ret_dict['map_polylines_center'] = map_polylines_center

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

        object_onehot_mask = np.zeros((num_center_objects, num_objects, num_timestamps, 5)) # mark agenet type, track_to_predict, ego_index
        object_onehot_mask[:, obj_types == 1, :, 0] = 1
        object_onehot_mask[:, obj_types == 2, :, 1] = 1
        object_onehot_mask[:, obj_types == 3, :, 2] = 1
        object_onehot_mask[:, track_index_to_predict, :, 3] = 1
        object_onehot_mask[:, sdc_track_index, :, 4] = 1

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
            object_onehot_mask,         # 5
            object_time_embedding,      # 12 because T_h=10, this can change for different experiment setting
            object_heading_embedding,   # 2
            obj_trajs[:, :, :, 7:9],    # 2
            acce,                       # 2
        ], axis=-1)  # (N_na, N_a, T_h, 29)

        # print([col.nonzero()[0] for col in obj_trajs_data[:, :, -1, 9]])

        obj_trajs_mask = obj_trajs[:, :, :, -1]     # (N_sa, N_a, T_h)
        obj_trajs_data[obj_trajs_mask == 0] = 0

        obj_trajs_future = obj_trajs_future.astype(np.float32)
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )   # (N_selected_agents, N_a, T_f, S)
        obj_trajs_future_state = obj_trajs_future[:, :, :, [0, 1, 6, 7, 8, 3, 4, 5]]  # (x, y, heading, vx, vy, l, w, h)
        obj_trajs_future_mask = obj_trajs_future[:, :, :, -1]
        obj_trajs_future_state[obj_trajs_future_mask == 0] = 0

        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0) # (N_a,)

        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask] # (N_sa, Num_agents_valid_at_curr_t, T_h, )
        obj_trajs_data = obj_trajs_data[:, valid_past_mask] # (N_sa, Num_a_at_curr_t, T_h, 29)
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask] # (N_sa, Num_agents_valid_at_curr_t, T_f, 29)
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

        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)

        obj_trajs_data = np.take_along_axis(obj_trajs_data, topk_idxs, axis=1)
        obj_trajs_mask = np.take_along_axis(obj_trajs_mask, topk_idxs[..., 0], axis=1)
        obj_trajs_pos = np.take_along_axis(obj_trajs_pos, topk_idxs, axis=1)
        obj_trajs_last_pos = np.take_along_axis(obj_trajs_last_pos, topk_idxs[..., 0], axis=1)
        obj_trajs_future_state = np.take_along_axis(obj_trajs_future_state, topk_idxs, axis=1)
        obj_trajs_future_mask = np.take_along_axis(obj_trajs_future_mask, topk_idxs[..., 0], axis=1)
        # track_index_to_predict_new = np.zeros(len(track_index_to_predict), dtype=np.int64)
        track_index_to_predict_new = [col.nonzero()[0] for col in obj_trajs_data[:, :, -1, 9]]
        track_index_to_predict_new = np.asarray(
            [
                np.pad(
                    k, (0, max_num_agents-k.shape[0]), 'constant', constant_values=(0,-1)
                ) for k in track_index_to_predict_new
            ]
        ) #(N_max_agents, N_pa,)
        obj_trajs_data = np.pad(obj_trajs_data, ((0, 0), (0, max_num_agents - obj_trajs_data.shape[1]), (0, 0), (0, 0)))
        obj_trajs_mask = np.pad(obj_trajs_mask, ((0, 0), (0, max_num_agents - obj_trajs_mask.shape[1]), (0, 0)))
        obj_trajs_pos = np.pad(obj_trajs_pos, ((0, 0), (0, max_num_agents - obj_trajs_pos.shape[1]), (0, 0), (0, 0)))
        obj_trajs_last_pos = np.pad(obj_trajs_last_pos,
                                    ((0, 0), (0, max_num_agents - obj_trajs_last_pos.shape[1]), (0, 0)))
        obj_trajs_future_state = np.pad(obj_trajs_future_state,
                                        ((0, 0), (0, max_num_agents - obj_trajs_future_state.shape[1]), (0, 0), (0, 0)))
        obj_trajs_future_mask = np.pad(obj_trajs_future_mask,
                                       ((0, 0), (0, max_num_agents - obj_trajs_future_mask.shape[1]), (0, 0)))

        center_gt_trajs = []
        center_gt_trajs_mask = []
        for obj_id in range(obj_trajs_future_state.shape[0]):
            _center_gt_trajs = obj_trajs_future_state[obj_id, track_index_to_predict_new[obj_id]].copy()   # (N_sa, T_f, 8)
            _center_gt_trajs_mask = obj_trajs_future_mask[obj_id, track_index_to_predict_new[obj_id]].copy() # (N_sa, T_f)
            _center_gt_trajs_mask[track_index_to_predict_new[obj_id]==-1] = False
            _center_gt_trajs[_center_gt_trajs_mask == 0] = 0.
            center_gt_trajs.append(np.expand_dims(_center_gt_trajs, axis=0))
            center_gt_trajs_mask.append(np.expand_dims(_center_gt_trajs_mask, axis=0))
        center_gt_trajs = np.concatenate(center_gt_trajs, axis=0)   # (N_oa, N_pa, T_f, 8)
        center_gt_trajs_mask = np.concatenate(center_gt_trajs_mask, axis=0)
        
        center_gt_final_valid_idx = np.zeros((num_center_objects, max_num_agents), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[2]):
            cur_valid_mask = center_gt_trajs_mask[:, :, k] > 0
            center_gt_final_valid_idx[cur_valid_mask] = k

        return (obj_trajs_data, obj_trajs_mask.astype(bool), obj_trajs_pos, obj_trajs_last_pos,
                obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask,
                center_gt_final_valid_idx,
                track_index_to_predict_new)

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
