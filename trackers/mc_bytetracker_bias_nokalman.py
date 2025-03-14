from audioop import cross
import imghdr
from operator import truediv
from unicodedata import name
import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
from collections import defaultdict, deque
from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState,MCBaseTrack
import ipdb
import cv2
from trackers.ocsort_tracker import oc_kalmanfilter
from trackers.ocsort_tracker.association import associate, iou_batch, linear_assignment
from trackers.ocsort_tracker.ocsort import convert_bbox_to_z, convert_x_to_bbox
from trackers.ocsort_tracker.ocsort import k_previous_obs


# from mmflow.apis import inference_model, init_model
iou_function = matching.general_iou
iou_name = "iou"
#iou_name = "diou_01"
#iou_name = "center_distance"
#iou_name = "norm_center_distance"
#iou_name = "tiou2"

#lowscore_iouname = iou_name
#lowscore_iouname = "diou_01"
#lowscore_iouname = "center_distance"
lowscore_iouname = "iou"
#lowscore_iouname = "tiou2"
#lowscore_iouname = "norm_center_distance"

fuse_score = True
no_kf=False

match_thresh = 0.5
#match_thresh = 100
#iou_function = matching.diou_distance

kwargs = {"drange":2}


use_single_thresh = False
#use_single_thresh = True

debug = False
print("mc iou_function:",iou_function,iou_name,lowscore_iouname)
#print("fuse score:",fuse_score)
#print("no kf",no_kf)
print("mc match thresh:",match_thresh)
print("use single thresh:",use_single_thresh)
print("debug:",debug)
print(kwargs)

# debug_save_root = "/home/ymf/codes/bytetrack_debug"

# data_root = "/hdd/yaomf/datas/drone_vs_bird/raw_data/test"
# os.makedirs(debug_save_root,exist_ok=True)
# if debug:
#     exp_name = f"dvb_{iou_name}_debug"
#     debug_folder = os.path.join(debug_save_root,exp_name)
#     os.makedirs(debug_folder,exist_ok=True)


det_color = (255,0,0)

track_color = (0,0,255)






class MCByteTrackNK(MCBaseTrack):
    def __init__(self, tlwh, score, cls_id):
        """
        :param tlwh:
        :param score:
        """
        # object class id
        self.cls_id = cls_id

        # init tlwh
        self._tlwh = np.asarray(tlwh, dtype=np.float)

        # init tlbr
        self._tlbr = MCByteTrackNK.tlwh2tlbr(self._tlwh)

        ## ----- build and initiate the Kalman filter
        self.kf = oc_kalmanfilter.KalmanFilterNew(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])  # constant velocity model
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        # give high uncertainty to the unobservable initial velocities
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.R[2:, 2:] *= 10.0

        # states: z: center_x, center_y, s(area), r(aspect ratio)
        # and center_x, center_y, s, derivatives of time
        self.kf.x[:4] = convert_bbox_to_z(self._tlbr)

        ## ----- init is_activated to be False
        self.is_activated = False
        self.score = score
        self.track_len = 0
    @staticmethod
    def multi_bias(stracks, bias):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.kf.x.copy() for st in stracks])
            multi_covariance = np.asarray([st.kf.P for st in stracks])

            #R = H[:2, :2]
            #R8x8 = np.kron(np.eye(4, dtype=float), R)
            #t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                # print("mean before cmc", mean)
                # mean = R8x8.dot(np.insert(mean, 7, 0))
                # print("mean after *M", mean[:7])
                print(mean[:2].shape)
                #mean[:2] += t.reshape(2, 1)
                # cov = R8x8[:7, :7].dot(cov).dot(R8x8[:7, :7].transpose())

                stracks[i].kf.x = mean[:7]

    def reset_track_id(self):
        """
        :return:
        """
        self.reset_track_id(self.cls_id)

    def predict(self):
        """
        Advances the state vector and
        returns the predicted bounding box estimate.
        """

        # if (self.kf.x[6] + self.kf.x[2]) <= 0:
        #     self.kf.x[6] *= 0.0

        # ## ----- Kalman predict
        # self.kf.predict()

        # bbox = np.squeeze(convert_x_to_bbox(self.kf.x, score=None))
        # self._tlbr = bbox
        # return bbox

        return self._tlbr

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :return:
        """
        self.frame_id = frame_id
        self.track_len += 1
        self.score = new_track.score

        new_bbox = new_track._tlbr
        bbox_score = np.array([new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3], self.score])

        self._tlbr = new_bbox
        ## ----- Update motion model: update Kalman filter
        # self.kf.update(convert_bbox_to_z(bbox_score))

        ## ----- Update the states
        self.state = TrackState.Tracked
        self.is_activated = True
        ## -----

    def activate(self, frame_id):
        """
        Start a new track-let: the initial activation
        :param frame_id:
        :return:
        """
        # update track id for the object class
        self.track_id = self.next_id(self.cls_id)
        self.track_len = 0  # init track len
        self.state = TrackState.Tracked

        self.frame_id = frame_id
        self.start_frame = frame_id
        if self.frame_id == 1:
            self.is_activated = True

    def re_activate(self,
                    new_track,
                    frame_id,
                    new_id=False,
                    using_delta_t=False):
        """
        :param new_track:
        :param frame_id:
        :param new_id:
        :param using_delta_t:
        :return:
        """
        ## ----- Kalman filter update
        bbox = new_track._tlbr
        new_bbox_score = np.array([bbox[0], bbox[1], bbox[2], bbox[3], new_track.score])
        self._tlbr = bbox
        # self.kf.update(convert_bbox_to_z(new_bbox_score))

        ## ----- update track-let states
        self.track_len = 0
        self.frame_id = frame_id
        self.score = new_track.score

        ## ----- Update tracking states
        self.state = TrackState.Tracked
        self.is_activated = True
        ## -----

        if new_id:  # update track id for the object class
            self.track_id = self.next_id(self.cls_id)

    def get_bbox(self):
        """
        Returns the current bounding box estimate.
        x1y1x2y2
        """
        # state = np.squeeze(convert_x_to_bbox(self.kf.x))
        # self._tlbr = state[:4]  # x1y1x2y2
        return self._tlbr

    @property
    def tlbr(self):
        x1y1x2y2 = self.get_bbox()
        return x1y1x2y2

    @property
    def tlwh(self):
        tlbr = self.get_bbox()
        # self._tlwh = MCTrackOCByte.tlbr2tlwh(tlbr)
        self._tlwh = self.tlbr2tlwh(tlbr)
        return self._tlwh

    @staticmethod
    def tlwh2tlbr(tlwh):
        """
        :param tlwh:
        """
        ret = np.squeeze(tlwh.copy())
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlbr2tlwh(tlbr):
        """
        :param tlbr:
        :return:
        """
        ret = np.squeeze(tlbr.copy())
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh2xyah(tlwh):
        """
        Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.squeeze(np.asarray(tlwh).copy())
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        """
        :return:
        """
        return self.tlwh2xyah(self._tlwh)

    def __repr__(self):
        """
        :return:
        """
        return "TR_({}-{})_({}-{})" \
            .format(self.cls_id, self.track_id, self.start_frame, self.end_frame)


class MCByteTrackerBias(object):
    def __init__(self, opt, frame_rate=30, delta_t=3,show_parameters=False,resize= None):
        """
        :param opt:
        :param frame_rate:
        :param delta_t:
        """
        self.frame_id = 0
        self.opt = opt
        #print("opt:\n", self.opt)

        # self.det_thresh = args.track_thresh
        self.low_det_thresh = 0.1

        #self.low_det_thresh = 0.2
        self.high_det_thresh = self.opt.track_thresh  # 0.5
        self.high_match_thresh = self.opt.match_thresh  # 0.8
        self.low_match_thresh = 0.5
        self.unconfirmed_match_thresh = 0.7
        self.new_track_thresh = self.high_det_thresh
        # self.new_track_thresh = 0.2


        # self.low_det_thresh = 0.2
        # self.high_det_thresh = 0.7
        #self.high_match_thresh = 0.8
        #self.low_match_thresh = 0.4
        #self.unconfirmed_match_thresh = 0.5

        if iou_name == "tiou2":
            self.high_match_thresh /= 2
            self.low_match_thresh /= 2
            self.unconfirmed_match_thresh /= 2

        self.show_parameters = show_parameters
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        if self.show_parameters:
            print("Tracker's low det thresh: ", self.low_det_thresh)
            print("Tracker's high det thresh: ", self.high_det_thresh)
            print("Tracker's high match thresh: ", self.high_match_thresh)
            print("Tracker's low match thresh: ", self.low_match_thresh)
            print("Tracker's unconfirmed match thresh: ", self.unconfirmed_match_thresh)
            print("Tracker's new track thresh: ", self.new_track_thresh)
            print("Tracker's buffer size: ", self.buffer_size)

        ## ----- shared Kalman filter
        self.kalman_filter = KalmanFilter()

        # Get number of tracking object classes
        #self.class_names = opt.class_names
        self.n_classes = opt.num_classes

        # Define track lists for single object class
        self.tracked_tracks = []  # type: list[Track]
        self.lost_tracks = []  # type: list[Track]
        self.removed_tracks = []  # type: list[Track]

        # Define tracks dict for multi-class objects
        self.tracked_tracks_dict = defaultdict(list)  # value type: dict(int, list[Track])
        self.lost_tracks_dict = defaultdict(list)  # value type: dict(int, list[Track])
        self.removed_tracks_dict = defaultdict(list)  # value type: dict(int, list[Track])

        self.tracks = []
        self.tracked_tracks = []

        self.iou_threshold = 0.3
        self.vel_dir_weight = 0.2

        self.delta_t = delta_t
        self.max_age = self.buffer_size
        self.min_hits = 3
        self.using_delta_t = True

        self.resize = resize

    #def update(self, dets, img_size, net_size):
    def update(self, dets, img_info,img_size,flow,flow_ratios=(1,1)):
        """
        Original byte tracking
        :param dets:
        :param img_size:
        :param net_size:
        :return:
        """
        ## ----- update frame id
        self.frame_id += 1

        ## ----- reset the track ids for all object classes in the first frame
        if self.frame_id == 1:
            MCByteTrackNK.init_id_dict(self.n_classes)
        ## -----

        ## ----- image width, height and net width, height
        # img_h, img_w = img_size
        # net_h, net_w = net_size
        # scale = min(net_h / float(img_h), net_w / float(img_w))

        #img_info 是原始大小
        #img_size是输入网络的大小

        img_h, img_w = int(img_info[0]), int(img_info[1])
        # scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))

        scale = 1.0

        #box / imgsize * img_h
        #print(img_w,img_size[0])

        #print((img_h,img_w),(img_size[0],img_size[1]))

        ## ----- gpu ——> cpu
        with torch.no_grad():
            dets = dets.cpu().numpy()

        ## ----- The current frame 8 tracking states recording
        unconfirmed_tracks_dict = defaultdict(list)
        tracked_tracks_dict = defaultdict(list)
        track_pool_dict = defaultdict(list)
        activated_tracks_dict = defaultdict(list)
        retrieve_tracks_dict = defaultdict(list)  # re-find
        lost_tracks_dict = defaultdict(list)
        removed_tracks_dict = defaultdict(list)
        output_tracks_dict = defaultdict(list)

        #################### Even: Start MCMOT
        ## ----- Fill box dict and score dict
        boxes_dict = defaultdict(list)
        scores_dict = defaultdict(list)

        for det in dets:
            # print(dets.shape,det.shape)
            if det.size == 7:
                x1, y1, x2, y2, score1, score2, cls_id = det  # 7
                score = score1 * score2
            elif det.size == 6:
                x1, y1, x2, y2, score, cls_id = det  # 6

            # print(det.size)
            box = np.array([x1, y1, x2, y2])
            box /= scale  # convert box to image size

            boxes_dict[int(cls_id)].append(box)
            scores_dict[int(cls_id)].append(score)

        # print(boxes_dict)
        ## ---------- Process each object class
        for cls_id in range(self.n_classes):
            ## ----- class boxes
            bboxes = boxes_dict[cls_id]
            bboxes = np.array(bboxes)

            ## ----- class scores
            scores = scores_dict[cls_id]
            scores = np.array(scores)

            ## first group inds
            inds_high = scores > self.high_det_thresh

            ## second group inds
            inds_lb = scores > self.low_det_thresh
            inds_hb = scores < self.high_det_thresh
            inds_low = np.logical_and(inds_lb, inds_hb)

            bboxes_high = bboxes[inds_high]
            bboxes_low = bboxes[inds_low]

            scores_high = scores[inds_high]
            scores_low = scores[inds_low]

            if len(bboxes_high) > 0:
                '''Detections'''
                detections_high = [MCByteTrackNK(MCByteTrackNK.tlbr2tlwh(tlbr), s, cls_id) for
                                   (tlbr, s) in zip(bboxes_high, scores_high)]
            else:
                detections_high = []

            '''Add newly detected tracks(current frame) to tracked_tracks'''
            for track in self.tracked_tracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_tracks_dict[cls_id].append(track)  # record unconfirmed tracks in this frame
                else:
                    tracked_tracks_dict[cls_id].append(track)  # record tracked tracks of this frame

            ''' Step 2: First association, with high score detection boxes'''
            ## ----- build track pool for the current frame by joining tracked_tracks and lost tracks
            track_pool_dict[cls_id] = join_tracks(tracked_tracks_dict[cls_id],
                                                  self.lost_tracks_dict[cls_id])
            self.tracks = track_pool_dict[cls_id]
            self.tracked_tracks = tracked_tracks_dict[cls_id]

            # ---------- Predict the current location with KF
            # if flow is not None:
            #     #flow = flow.reshape([img_size[0],img_size[1],2])
            #     bottom_max,right_max,_ = flow.shape

            #     #print(flow.shape,img_size)
            # else:
            #     bottom_max,right_max = img_size[0],img_size[1]
            
            # bottom_max,right_max = img_size[0],img_size[1]
            bottom_max,right_max = img_h,img_w

            bottom_max -= 1
            right_max -= 1
            
            for track in self.tracked_tracks:

                #src_box = track.predict()
                #src_box = track.tlbr


                # src_box = track.predict()

                src_box = track.tlbr
                track.predict()

                #print(track.tlbr)
                #print("before:",src_box)
                left,top,right,bottom = src_box
                #print(flow.shape,src_box,type(src_box))
                top,left,bottom,right = int(top),int(left),int(bottom),int(right)

                top = 0 if top < 0 else top
                left = 0 if left < 0 else left
                bottom = bottom_max if bottom > bottom_max else bottom
                right = right_max if right > right_max else right
                #print(left,top,right,bottom,right_max,bottom_max)
                ##这个地方可能存在bug!

                #print(scale,flow_ratios)
                cx,cy = (right+left)/2,(bottom+top)/2
                cx,cy = int(cx / flow_ratios[0]) , int(cy / flow_ratios[1])
                
                #print(top,left,bottom,right)
                if flow is None or (top >= bottom or left >= right):
                    horizontal_offset,vertical_offset,flow_h,flow_w = 0,0,384,512
                else:
                    #flow_box = flow[top:bottom,left:right,:]
                    if len(flow.shape) == 4:
                        _,_,flow_h,flow_w = flow.shape
                        flow_box = flow[0,:,cy,cx]
                        horizontal_offset = flow_box[0,...].mean().cpu().numpy()
                        vertical_offset = flow_box[1,...].mean().cpu().numpy()
                    else:
                        flow_h,flow_w,_ = flow.shape
                        flow_box = flow[cy,cx,:]
                        horizontal_offset = flow_box[...,0].mean()
                        vertical_offset = flow_box[...,1].mean()
                    

                    #horizontal_offset = flow_box[]

                # flow_ratios[0] : w_ratios
                horizontal_offset *= img_w / flow_w
                vertical_offset *= img_h / flow_h

                # ipdb.set_trace()
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #horizontal_offset,vertical_offset = 0,0

                # left += horizontal_offset
                # right += horizontal_offset

                # top += vertical_offset
                # bottom += vertical_offset

                # cx,cy = (left + right)/2 , (top + bottom)/2
                new_left,new_top,new_right,new_bottom = src_box[0] + horizontal_offset,src_box[1] + vertical_offset,src_box[2] + horizontal_offset,src_box[3] + vertical_offset

                track._tlbr = np.array([new_left,new_top,new_right,new_bottom])

                #print(src_box,track._tlbr)
                # mean = track.kf.x.copy()
                # mean[0] += horizontal_offset
                # mean[1] += vertical_offset
                # track.kf.x = mean[:7]
                



            # Matching with Hungarian Algorithm

            dists = iou_function(self.tracks, detections_high,iou_name,kwargs)

            dists = matching.fuse_score(dists, detections_high)

            if use_single_thresh:
                matches, u_track_1st, u_detection_1st = matching.linear_assignment(dists,thresh=match_thresh)
            else:
                matches, u_track_1st, u_detection_1st = matching.linear_assignment(dists,thresh=self.high_match_thresh)

            # --- process matched pairs between track pool and current frame detection
            for i_track, i_det in matches:
                track = self.tracks[i_track]
                det = detections_high[i_det]

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)  # for multi-class
                else:  # re-activate the lost track
                    track.re_activate(det, self.frame_id, new_id=False)
                    retrieve_tracks_dict[cls_id].append(track)

            ''' Step 3: Second association, with low score detection boxes'''
            # association the un-track to the low score detections
            if len(bboxes_low) > 0:
                '''Detections'''
                detections_low = [MCByteTrackNK(MCByteTrackNK.tlbr2tlwh(tlbr), score, cls_id) for
                                  (tlbr, score) in zip(bboxes_low, scores_low)]
            else:
                detections_low = []

            ## ----- record un-matched tracks after the 1st round matching
            unmatched_tracks = [self.tracks[i]
                                for i in u_track_1st
                                if self.tracks[i].state == TrackState.Tracked]

            dists = iou_function(unmatched_tracks, detections_low,lowscore_iouname,kwargs)

            if use_single_thresh:
                matches, u_track_2nd, u_detection_2nd = matching.linear_assignment(dists,thresh=match_thresh)  # thresh=0.5
            else:
                if iou_name =="tiou2" and lowscore_iouname =="iou":
                    matches, u_track_2nd, u_detection_2nd = matching.linear_assignment(dists,thresh=self.low_match_thresh*2)  # thresh=0.5
                else:
                    matches, u_track_2nd, u_detection_2nd = matching.linear_assignment(dists,thresh=self.low_match_thresh)  # thresh=0.5

            for i_track, i_det in matches:
                track = unmatched_tracks[i_track]
                det = detections_low[i_det]

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_tracks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    retrieve_tracks_dict[cls_id].append(track)

            # process unmatched tracks for two rounds
            for i_track in u_track_2nd:
                track = unmatched_tracks[i_track]
                if not track.state == TrackState.Lost:
                    # mark unmatched track as lost track
                    track.mark_lost()
                    lost_tracks_dict[cls_id].append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            # current frame's unmatched detection
            dets_left = [detections_high[i] for i in u_detection_1st]  # high left
            dets_low_left = [detections_low[i] for i in u_detection_2nd]
            if len(dets_low_left) > 0:
                dets_left = join_tracks(dets_left, dets_low_left)

            # iou matching
            dists = iou_function(unconfirmed_tracks_dict[cls_id], dets_left,iou_name,kwargs)
            dists = matching.fuse_score(dists, dets_left)
            if use_single_thresh:
                matches, u_unconfirmed, u_detection_1st = matching.linear_assignment(dists,thresh=self.unconfirmed_match_thresh)  # 0.7
            else:
                if iou_name =="tiou2" and lowscore_iouname =="iou":
                    matches, u_unconfirmed, u_detection_1st = matching.linear_assignment(dists,thresh=match_thresh*2)  # 0.7
                else:
                    matches, u_unconfirmed, u_detection_1st = matching.linear_assignment(dists,thresh=match_thresh)  # 0.7
            for i_track, i_det in matches:
                track = unconfirmed_tracks_dict[cls_id][i_track]
                det = dets_left[i_det]
                track.update(det, self.frame_id)
                activated_tracks_dict[cls_id].append(track)

            for i_track in u_unconfirmed:  # process unconfirmed tracks
                track = unconfirmed_tracks_dict[cls_id][i_track]
                track.mark_removed()
                removed_tracks_dict[cls_id].append(track)

            """Step 4: Init new tracks"""
            for i_new in u_detection_1st:  # current frame's unmatched detection
                track = dets_left[i_new]
                if track.score < self.new_track_thresh:
                    continue

                # tracked but not activated: activate do not set 'is_activated' to be True
                # if fr_id > 1, tracked but not activated
                track.activate(self.frame_id)

                # activated_tracks_dict may contain track with 'is_activated' False
                activated_tracks_dict[cls_id].append(track)

            """Step 5: Update state"""
            # update removed tracks
            for track in self.lost_tracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_tracks_dict[cls_id].append(track)

            """Post processing"""
            self.tracked_tracks_dict[cls_id] = [t for t in self.tracked_tracks_dict[cls_id] if
                                                t.state == TrackState.Tracked]
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           activated_tracks_dict[cls_id])
            self.tracked_tracks_dict[cls_id] = join_tracks(self.tracked_tracks_dict[cls_id],
                                                           retrieve_tracks_dict[cls_id])

            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id], self.tracked_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id].extend(lost_tracks_dict[cls_id])
            self.lost_tracks_dict[cls_id] = sub_tracks(self.lost_tracks_dict[cls_id], self.removed_tracks_dict[cls_id])

            self.removed_tracks_dict[cls_id].extend(removed_tracks_dict[cls_id])

            self.tracked_tracks_dict[cls_id], self.lost_tracks_dict[cls_id] = remove_duplicate_tracks(
                self.tracked_tracks_dict[cls_id],
                self.lost_tracks_dict[cls_id])

            # get scores of lost tracks
            output_tracks_dict[cls_id] = [track for track in self.tracked_tracks_dict[cls_id]
                                          if track.is_activated]

        ## ---------- Return final online targets of the frame
        return output_tracks_dict


def remove_duplicate_tracks(tracks_a, tracks_b):
    """
    :param tracks_a:
    :param tracks_b:
    :return:
    """
    pdist = iou_function(tracks_a, tracks_b,iou_name,kwargs)
    #pdist = matching.iou_distance(tracks_a, tracks_b)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()

    for p, q in zip(*pairs):
        timep = tracks_a[p].frame_id - tracks_a[p].start_frame
        timeq = tracks_b[q].frame_id - tracks_b[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)

    resa = [t for i, t in enumerate(tracks_a) if not i in dupa]
    resb = [t for i, t in enumerate(tracks_b) if not i in dupb]

    return resa, resb

def join_tracks(list_1, list_2):
    """
    :param list_1:
    :param list_2:
    :return:
    """
    exists = {}
    res = []

    for t in list_1:
        exists[t.track_id] = 1
        res.append(t)
    for t in list_2:
        tr_id = t.track_id
        if not exists.get(tr_id, 0):
            exists[tr_id] = 1
            res.append(t)

    return res

def sub_tracks(t_list_a, t_list_b):
    """
    :param t_list_a:
    :param t_list_b:
    :return:
    """
    tracks = {}
    for t in t_list_a:
        tracks[t.track_id] = t
    for t in t_list_b:
        tid = t.track_id
        if tracks.get(tid, 0):
            del tracks[tid]
    return list(tracks.values())
