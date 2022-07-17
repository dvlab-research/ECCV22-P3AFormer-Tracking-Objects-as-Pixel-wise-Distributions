import os
import os.path as osp
import csv
from shutil import copyfile


def write_results(all_tracks, out_dir, seq_name=None, frame_offset=0, verbose=False):
    output_dir = out_dir + "/txt/"
    """Write the tracks in the format for MOT16/MOT17 submission
    all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num  if frame_first=False,
    Each file contains these lines:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """
    # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"
    assert seq_name is not None, "[!] No seq_name, probably using combined database"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = osp.join(output_dir, seq_name + ".txt")
    with open(save_path, "w") as of:
        writer = csv.writer(of, delimiter=",")
        for i in sorted(all_tracks):
            track = all_tracks[i]
            for frame, bb in track.items():
                x1 = bb[0]
                y1 = bb[1]
                x2 = bb[2]
                y2 = bb[3]
                writer.writerow(
                    [
                        frame + frame_offset,
                        i + 1,
                        x1 + 1,
                        y1 + 1,
                        x2 - x1 + 1,
                        y2 - y1 + 1,
                        -1,
                        -1,
                        -1,
                        -1,
                    ]
                )
    # TODO: validate this in MOT15
    # copy to FRCNN, DPM.txt, private setting
    copyfile(save_path, save_path[:-7] + "FRCNN.txt")
    copyfile(save_path, save_path[:-7] + "DPM.txt")
    if verbose:
        print("Write txt results at: ", save_path, ".")
    return save_path
