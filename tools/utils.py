import cv2


def annotate(frame, tracks, detections):
    """Annotate the current frame from the video with tracks and detections.

    If empty lists are given, annotate nothing.

    Parameters
    ----------
    frame : ndarray
        The current frame in format (H, W, C) #BGR
    tracks : List[deep_sort.track.Track]
        A track is the current tracked position for a person.
    detections : List[deep_sort.detection.Detection]
        A detection is a bbox for a person, either tracked or not.
    """
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
            bbox[2]), int(bbox[3])), (255, 255, 255), 2)
        cv2.putText(frame, str(track.track_id), (int(
            bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

    for det in detections:
        bbox = det.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
            bbox[2]), int(bbox[3])), (255, 0, 0), 2)


def crop_tracks(image, tracks):
    """Crop the tracks at this time point to a dictionary.

    Parameters
    ----------
    image : ndarray
        The current image in format (H, W, C) #RGB
    tracks : List[deep_sort.track.Track]
        A track is the current tracked position for a person.

    Returns
    ----------
    crops : dict[track_id -> image]
    """
    crops = {}
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_tlbr()
        crops[track.track_id] = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    return crops
