from re import I
import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from . import kalman_filter
import time
import ipdb
import math
sigma = 1e-8
def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []

    #thresh = 1
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)

    
    if ious.size == 0:
        return ious

    

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )

    #[[a1b1,a1b2,a1b3],
    # [a2b1,a2b2,a2b3]]
    return ious


def process_ab(atracks,btracks):
    
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    
    return atlbrs,btlbrs

def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    atlbrs,btlbrs = process_ab(atracks,btracks)
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix

def diou_distance(atracks,btracks):
    atlbrs0,btlbrs0 = process_ab(atracks,btracks)
    
    dious = np.zeros((len(atlbrs0), len(btlbrs0)), dtype=np.float64)
        
    if dious.size == 0:
        return dious

    atlbrs,btlbrs = np.array(atlbrs0),np.array(btlbrs0)
    
    
    top1,left1,bottom1,right1 = atlbrs[:,0],atlbrs[:,1],atlbrs[:,2],atlbrs[:,3]
    top2,left2,bottom2,right2 = btlbrs[:,0],btlbrs[:,1],btlbrs[:,2],btlbrs[:,3]

    top1,left1,bottom1,right1 = top1.reshape([top1.shape[0],1]),left1.reshape([left1.shape[0],1]),bottom1.reshape([bottom1.shape[0],1]),right1.reshape([right1.shape[0],1])
    top2,left2,bottom2,right2 = top2.reshape([top2.shape[0],1]),left2.reshape([left2.shape[0],1]),bottom2.reshape([bottom2.shape[0],1]),right2.reshape([right2.shape[0],1])
    top2,left2,bottom2,right2 = top2.T,left2.T,bottom2.T,right2.T


    top,left,bottom,right = np.minimum(top1,top2),np.minimum(left1,left2),np.maximum(bottom1,bottom2),np.maximum(right1,right2)
    cx1,cy1,cx2,cy2 = (left1 + right1)/2,(top1 + bottom1)/2,(left2 + right2)/2,(top2 + bottom2)/2

    p = (cx1 - cx2)**2 + (cy1 - cy2)**2
    cc = (top - bottom)**2 + (left - right)**2

    u = p/cc
    #u = u.reshape([u.shape[0],1])

    iou = ious(atlbrs,btlbrs)

    if iou.shape[0] != u.shape[0]:
        u = u.reshape([u.shape[1],u.shape[0]])

    dious = 1 - iou + u

    return dious

def diou_01(atracks,btracks):
    dious = diou_distance(atracks,btracks)
    dious = dious / 2
    return dious

def giou_distance(atracks,btracks):
    atlbrs0,btlbrs0 = process_ab(atracks,btracks)
    
    gious = np.zeros((len(atlbrs0), len(btlbrs0)), dtype=np.float64)
        
    if gious.size == 0:
        return gious

    atlbrs,btlbrs = np.array(atlbrs0),np.array(btlbrs0)
    
    
    top1,left1,bottom1,right1 = atlbrs[:,0],atlbrs[:,1],atlbrs[:,2],atlbrs[:,3]
    top2,left2,bottom2,right2 = btlbrs[:,0],btlbrs[:,1],btlbrs[:,2],btlbrs[:,3]

    top1,left1,bottom1,right1 = top1.reshape([top1.shape[0],1]),left1.reshape([left1.shape[0],1]),bottom1.reshape([bottom1.shape[0],1]),right1.reshape([right1.shape[0],1])
    top2,left2,bottom2,right2 = top2.reshape([top2.shape[0],1]),left2.reshape([left2.shape[0],1]),bottom2.reshape([bottom2.shape[0],1]),right2.reshape([right2.shape[0],1])
    top2,left2,bottom2,right2 = top2.T,left2.T,bottom2.T,right2.T

    top,left,bottom,right = np.minimum(top1,top2),np.minimum(left1,left2),np.maximum(bottom1,bottom2),np.maximum(right1,right2)
    
    C = (bottom - top)*(right - left)

    _top,_left,_bottom,_right = np.maximum(top1,top2),np.maximum(left1,left2),np.minimum(bottom1,bottom2),np.minimum(right1,right2)
    
    w0,h0 = (_bottom - _top),(_right - _left)
    w0[w0<0]=0
    h0[h0<0]=0
    inter = w0*h0
    #inter[inter < 0] = 0
    s1,s2 = (bottom1 - top1)*(right1 - left1),(bottom2 - top2)*(right2 - left2)
    union = s1 + s2 - inter

    true_iou = ious(atlbrs,btlbrs)
    gious = 1 - true_iou + (C-union)/C

    return gious


def ciou_distance(atracks,btracks):
    atlbrs0,btlbrs0 = process_ab(atracks,btracks)
    
    cious = np.zeros((len(atlbrs0), len(btlbrs0)), dtype=np.float64)
        
    if cious.size == 0:
        return cious

    atlbrs,btlbrs = np.array(atlbrs0),np.array(btlbrs0)
    
    
    top1,left1,bottom1,right1 = atlbrs[:,0],atlbrs[:,1],atlbrs[:,2],atlbrs[:,3]
    top2,left2,bottom2,right2 = btlbrs[:,0],btlbrs[:,1],btlbrs[:,2],btlbrs[:,3]

    top1,left1,bottom1,right1 = top1.reshape([top1.shape[0],1]),left1.reshape([left1.shape[0],1]),bottom1.reshape([bottom1.shape[0],1]),right1.reshape([right1.shape[0],1])
    top2,left2,bottom2,right2 = top2.reshape([top2.shape[0],1]),left2.reshape([left2.shape[0],1]),bottom2.reshape([bottom2.shape[0],1]),right2.reshape([right2.shape[0],1])
    top2,left2,bottom2,right2 = top2.T,left2.T,bottom2.T,right2.T

    top,left,bottom,right = np.minimum(top1,top2),np.minimum(left1,left2),np.maximum(bottom1,bottom2),np.maximum(right1,right2)
    

    cx1,cy1,cx2,cy2 = (left1 + right1)/2,(top1 + bottom1)/2,(left2 + right2)/2,(top2 + bottom2)/2

    p = (cx1 - cx2)**2 + (cy1 - cy2)**2
    cc = (top - bottom)**2 + (left - right)**2

    u = p/cc
    w1,h1,w2,h2 = (right1 - left1),(bottom1 - top1),(right2 - left2),(bottom2 - top2)

    true_iou = ious(atlbrs,btlbrs)
    arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v = (4 / (math.pi ** 2)) * np.power((np.arctan(w2 / h2) - np.arctan(w1 / h1)), 2)
    S = 1 - true_iou
    alpha = v / (S + v)
    w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    cious = 1 - true_iou + u + alpha*ar

    return cious


def eiou_distance(atracks,btracks):
    atlbrs0,btlbrs0 = process_ab(atracks,btracks)
    
    eious = np.zeros((len(atlbrs0), len(btlbrs0)), dtype=np.float64)
        
    if eious.size == 0:
        return eious

    atlbrs,btlbrs = np.array(atlbrs0),np.array(btlbrs0)
    
    
    top1,left1,bottom1,right1 = atlbrs[:,0],atlbrs[:,1],atlbrs[:,2],atlbrs[:,3]
    top2,left2,bottom2,right2 = btlbrs[:,0],btlbrs[:,1],btlbrs[:,2],btlbrs[:,3]

    

    top1,left1,bottom1,right1 = top1.reshape([top1.shape[0],1]),left1.reshape([left1.shape[0],1]),bottom1.reshape([bottom1.shape[0],1]),right1.reshape([right1.shape[0],1])
    top2,left2,bottom2,right2 = top2.reshape([top2.shape[0],1]),left2.reshape([left2.shape[0],1]),bottom2.reshape([bottom2.shape[0],1]),right2.reshape([right2.shape[0],1])
    top2,left2,bottom2,right2 = top2.T,left2.T,bottom2.T,right2.T


    top,left,bottom,right = np.minimum(top1,top2),np.minimum(left1,left2),np.maximum(bottom1,bottom2),np.maximum(right1,right2)
    cx1,cy1,cx2,cy2 = (left1 + right1)/2,(top1 + bottom1)/2,(left2 + right2)/2,(top2 + bottom2)/2


    w1,h1,w2,h2 = right1 - left1,bottom1 - top1,right2 - left2,bottom2 - top2

    cw,ch = (left - right)**2,(top - bottom)**2 
    pw0,ph0 = (w1 - w2)**2,(h1 - h2)**2

    uw,uh = pw0/cw , ph0 / ch

    p = (cx1 - cx2)**2 + (cy1 - cy2)**2
    cc = cw + ch

    u = p/cc
    #u = u.reshape([u.shape[0],1])

    iou = ious(atlbrs,btlbrs)

    if iou.shape[0] != u.shape[0]:
        u = u.reshape([u.shape[1],u.shape[0]])

    eious = 1 - iou + u + uw + uh

    return eious


def corner_iou_distance(atracks,btracks):
    atlbrs0,btlbrs0 = process_ab(atracks,btracks)
    
    corner_ious = np.zeros((len(atlbrs0), len(btlbrs0)), dtype=np.float64)
        
    if corner_ious.size == 0:
        return corner_ious

    atlbrs,btlbrs = np.array(atlbrs0),np.array(btlbrs0)
    
    
    top1,left1,bottom1,right1 = atlbrs[:,0],atlbrs[:,1],atlbrs[:,2],atlbrs[:,3]
    top2,left2,bottom2,right2 = btlbrs[:,0],btlbrs[:,1],btlbrs[:,2],btlbrs[:,3]

    top1,left1,bottom1,right1 = top1.reshape([top1.shape[0],1]),left1.reshape([left1.shape[0],1]),bottom1.reshape([bottom1.shape[0],1]),right1.reshape([right1.shape[0],1])
    top2,left2,bottom2,right2 = top2.reshape([top2.shape[0],1]),left2.reshape([left2.shape[0],1]),bottom2.reshape([bottom2.shape[0],1]),right2.reshape([right2.shape[0],1])
    top2,left2,bottom2,right2 = top2.T,left2.T,bottom2.T,right2.T


    top,left,bottom,right = np.minimum(top1,top2),np.minimum(left1,left2),np.maximum(bottom1,bottom2),np.maximum(right1,right2)
    cx1,cy1,cx2,cy2 = (left1 + right1)/2,(top1 + bottom1)/2,(left2 + right2)/2,(top2 + bottom2)/2

    
    d3 = (top1 - top2)**2 + (left1 - left2)**2
    d4 = (bottom1 - bottom2)**2 + (right1 - right2)**2
    #p = (cx1 - cx2)**2 + (cy1 - cy2)**2
    cc = (top - bottom)**2 + (left - right)**2

    u = (d3 + d4)/cc
    #u = u.reshape([u.shape[0],1])

    iou = ious(atlbrs,btlbrs)

    if iou.shape[0] != u.shape[0]:
        u = u.reshape([u.shape[1],u.shape[0]])

    corner_ious = 1 - iou + u

    return corner_ious




def center_distance(atracks,btracks,cross_len):
    atlbrs0,btlbrs0 = process_ab(atracks,btracks)
    
    center_dis = np.zeros((len(atlbrs0), len(btlbrs0)), dtype=np.float64)
        
    if center_dis.size == 0:
        return center_dis

    atlbrs,btlbrs = np.array(atlbrs0),np.array(btlbrs0)
    
    
    top1,left1,bottom1,right1 = atlbrs[:,0],atlbrs[:,1],atlbrs[:,2],atlbrs[:,3]
    top2,left2,bottom2,right2 = btlbrs[:,0],btlbrs[:,1],btlbrs[:,2],btlbrs[:,3]

    top1,left1,bottom1,right1 = top1.reshape([top1.shape[0],1]),left1.reshape([left1.shape[0],1]),bottom1.reshape([bottom1.shape[0],1]),right1.reshape([right1.shape[0],1])
    top2,left2,bottom2,right2 = top2.reshape([top2.shape[0],1]),left2.reshape([left2.shape[0],1]),bottom2.reshape([bottom2.shape[0],1]),right2.reshape([right2.shape[0],1])
    top2,left2,bottom2,right2 = top2.T,left2.T,bottom2.T,right2.T

    cx1,cy1,cx2,cy2 = (left1 + right1)/2,(top1 + bottom1)/2,(left2 + right2)/2,(top2 + bottom2)/2

    p = (cx1 - cx2)**2 + (cy1 - cy2)**2
    p = np.sqrt(p / cross_len)

    return p

def norm_center_distance(atracks,btracks,kwargs):
    
    drange = 10

    if kwargs is not None and "drange" in kwargs:
        drange = kwargs["drange"]
    #根据框的大小对中心点距离进行规范化
    
    atlbrs0,btlbrs0 = process_ab(atracks,btracks)
    
    center_dis = np.zeros((len(atlbrs0), len(btlbrs0)), dtype=np.float64)
        
    if center_dis.size == 0:
        return center_dis

    atlbrs,btlbrs = np.array(atlbrs0),np.array(btlbrs0)
    
    
    top1,left1,bottom1,right1 = atlbrs[:,0],atlbrs[:,1],atlbrs[:,2],atlbrs[:,3]
    top2,left2,bottom2,right2 = btlbrs[:,0],btlbrs[:,1],btlbrs[:,2],btlbrs[:,3]

    top1,left1,bottom1,right1 = top1.reshape([top1.shape[0],1]),left1.reshape([left1.shape[0],1]),bottom1.reshape([bottom1.shape[0],1]),right1.reshape([right1.shape[0],1])
    top2,left2,bottom2,right2 = top2.reshape([top2.shape[0],1]),left2.reshape([left2.shape[0],1]),bottom2.reshape([bottom2.shape[0],1]),right2.reshape([right2.shape[0],1])
    top2,left2,bottom2,right2 = top2.T,left2.T,bottom2.T,right2.T


    d1,d2 = (top1 - bottom1)**2 + (right1 - left1)**2 , (top2 - bottom2)**2 + (right2 - left2)**2
    d1,d2 = np.sqrt(d1),np.sqrt(d2)
    min_d = np.minimum(d1,d2)
    #min_d = d1

    cx1,cy1,cx2,cy2 = (left1 + right1)/2,(top1 + bottom1)/2,(left2 + right2)/2,(top2 + bottom2)/2

    p = (cx1 - cx2)**2 + (cy1 - cy2)**2
    p = np.sqrt(p)


    t_iou = p / (min_d*drange*2)
    return t_iou
    

def tiou2(atracks,btracks,kwargs):
    ious = iou_distance(atracks,btracks)
    tious = norm_center_distance(atracks,btracks,kwargs)

    return 0.5*ious + 0.5*tious




def general_iou(atracks,btracks,name="iou",cross_len=None,kwargs=None):

    if name == "iou":
        return iou_distance(atracks,btracks)
    elif name == "diou":
        return diou_distance(atracks,btracks)
    elif name == "giou":
        return giou_distance(atracks,btracks)
    elif name == "ciou":
        return ciou_distance(atracks,btracks)
    elif name == "center_distance":
        return center_distance(atracks,btracks,cross_len)
    elif name == "eiou":
        return eiou_distance(atracks,btracks)
    elif name == "corner_iou":
        return corner_iou_distance(atracks,btracks)
    elif name == "norm_center_distance":
        return norm_center_distance(atracks,btracks,kwargs)
    elif name == "diou_01":
        return diou_01(atracks,btracks)
    elif name == "tiou2":
        return tiou2(atracks,btracks,kwargs)
    
    raise Exception(f"iou {name} not found!")



def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost