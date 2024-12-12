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

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState
import ipdb
import cv2
class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score,class_id,no_kf=False):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.no_kf = no_kf
        self.class_id = class_id

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        
        if not self.no_kf:
            self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        #if not self.no_kf:
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        if not self.no_kf:
            self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        if not self.no_kf:
            self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.class_id = new_track.class_id

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

iou_function = matching.general_iou
iou_name = "iou"
#iou_name = "diou"
#iou_name = "center_distance"
#iou_name = "norm_center_distance"

#lowscore_iouname = "norm_center_distance"
#lowscore_iouname = "diou"
#lowscore_iouname = "center_distance"
lowscore_iouname = "iou"
fuse_score = True
no_kf=False

match_thresh = 1.75
#match_thresh = 100
#iou_function = matching.diou_distance

kwargs = {"drange":4}

debug = False
print("iou_function:",iou_name,lowscore_iouname)
print("fuse score:",fuse_score)
print("no kf",no_kf)
print("match thresh:",match_thresh)
print("debug:",debug)
print(kwargs)


debug_save_root = "/home/ymf/codes/bytetrack_debug"

# data_root = "/hdd/yaomf/datas/drone_vs_bird/raw_data/test"
# os.makedirs(debug_save_root,exist_ok=True)
# if debug:
#     exp_name = f"dvb_{iou_name}_debug"
#     debug_folder = os.path.join(debug_save_root,exp_name)
#     os.makedirs(debug_folder,exist_ok=True)


det_color = (255,0,0)

track_color = (0,0,255)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2

            class_ids = output_results[:,6]

        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        img = None
        # if debug:
        #     img_file = img_info[4][0]
        #     img_path = os.path.join(data_root,img_file)
        #     splitted = img_file.split("/")
        #     seq_name,img_name = splitted[0],splitted[2]

        #     seq_debug_folder = os.path.join(debug_folder,seq_name)
        #     os.makedirs(seq_debug_folder,exist_ok=True)
        #     debug_img_path = os.path.join(seq_debug_folder,img_name)
        #     #print(img_path)
        #     img = cv2.imread(img_path)

        
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        class_ids_keep = class_ids[remain_inds]
        class_ids_second = class_ids[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s,ids,no_kf=no_kf) for
                          (tlbr, s,ids) in zip(dets, scores_keep,class_ids_keep)]
        else:
            detections = []

        # if debug:
        #     for t in detections:

        #         bboxes = t.tlbr
        #         #bboxes *= scale
        #         bboxes = bboxes.astype(np.int32).tolist()
        #         left,top,right,bottom = bboxes
        #         score = t.score
        #         track_id = t.track_id

        #         cv2.rectangle(img,(left,top),(right,bottom),det_color,2,2)
        #         #txt = f":{score:.3f}"
        #         #cv2.putText(img,txt,(left,top-2),cv2.FONT_HERSHEY_SIMPLEX, 0.75, det_color, 2)

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF

        if not no_kf:
            STrack.multi_predict(strack_pool)

        cross_len = (img_h*img_h + img_w*img_w).item()

        dists = iou_function(strack_pool, detections,iou_name,cross_len,kwargs)
        
        
        if not self.args.mot20 and fuse_score:
            dists = matching.fuse_score(dists, detections)
        #matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=match_thresh)
        #matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.6)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s,ids) for
                          (tlbr, s,ids) in zip(dets_second, scores_second,class_ids_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        dists = iou_function(r_tracked_stracks, detections_second,lowscore_iouname,cross_len,kwargs)
        #matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=match_thresh)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]


        #增加confirmed的难度

        dists = iou_function(unconfirmed, detections,iou_name,cross_len,kwargs)
        if not self.args.mot20 and fuse_score:
            dists = matching.fuse_score(dists, detections)

        #matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=match_thresh)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks,cross_len)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # if debug:
        #     for t in output_stracks:
        #         bboxes = t.tlbr
        #         #bboxes *= scale
        #         bboxes = bboxes.astype(np.int32).tolist()
        #         left,top,right,bottom = bboxes
        #         score = t.score
        #         track_id = t.track_id

        #         cv2.rectangle(img,(left,top),(right,bottom),track_color,1,1)
        #         txt = f"{track_id}:{score:.3f}"
        #         cv2.putText(img,txt,(left,top-2),cv2.FONT_HERSHEY_SIMPLEX, 0.75, track_color, 2)
            
        #     cv2.imwrite(debug_img_path,img)
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb,cross_len):
    
    pdist = iou_function(stracksa, stracksb,lowscore_iouname,cross_len,kwargs)

    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
