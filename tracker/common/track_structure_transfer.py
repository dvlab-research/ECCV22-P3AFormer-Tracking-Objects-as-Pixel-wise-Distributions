from os import curdir
import pdb
import numpy as np

def frame_first_to_id_first(frame_first):
    """
    Frame first result: {Frame ID: a list of [x1, y1, x2, y2, score, ...]}
    Track first result: {Track ID: {Frame ID: [x1, y1, x2, y2, score, ...]}}
    """
    results = {}
    for frameid, bbs in frame_first.items():
        for one_bb in bbs:
            x1, y1, x2, y2, score, cur_id = one_bb[0], one_bb[1], one_bb[2], one_bb[3], one_bb[4], one_bb[5]
            if cur_id not in results:
                results[cur_id] = {}
            results[cur_id][frameid] = np.array([x1, y1, x2, y2, score])
    return results


def id_first_to_frame_first(id_first):
    """
    Frame first result: {Frame ID: a list of [x1, y1, x2, y2, score, ...]}
    Track first result: {Track ID: {Frame ID: [x1, y1, x2, y2, score, ...]}}
    """
    results = {}
    for i, track in id_first.items():
        for frame, bb in track.items():
            if frame not in results:
                results[frame] = []
            x1 = bb[0]
            y1 = bb[1]
            x2 = bb[2]
            y2 = bb[3]
            score = bb[4]
            results[frame].append([x1, y1, x2, y2, score, i+1])
    return results