from loguru import logger

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},{ids},-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores,class_ids in results:
            #print(results)
            for tlwh, track_id, score,class_id in zip(tlwhs, track_ids, scores,class_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2),ids=class_id)
                f.write(line)
    logger.info('save results to {}'.format(filename))