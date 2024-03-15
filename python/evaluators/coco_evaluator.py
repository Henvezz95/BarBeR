""" version ported from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    Notes:
        1) The default area thresholds here follows the values defined in COCO, that is,
        small:           area <= 32**2
        medium: 32**2 <= area <= 96**2
        large:  96**2 <= area.
        If area is not specified, all areas are considered.

        2) COCO's ground truths contain an 'area' attribute that is associated with the segmented area if
        segmentation-level information exists. While coco uses this 'area' attribute to distinguish between
        'small', 'medium', and 'large' objects, this implementation simply uses the associated bounding box
        area to filter the ground truths.

        3) COCO uses floating point bounding boxes, thus, the calculation of the box area
        for IoU purposes is the simple open-ended delta (x2 - x1) * (y2 - y1).
        PASCALVOC uses integer-based bounding boxes, and the area includes the outer edge,
        that is, (x2 - x1 + 1) * (y2 - y1 + 1). This implementation assumes the open-ended (former)
        convention for area calculation.
"""

from collections import defaultdict

import numpy as np
from bounding_box import BBFormat

def get_coco_summary(groundtruth_bbs, detected_bbs):
    """Calculate the 12 standard metrics used in COCOEval,
        AP, AP50, AP75,
        AR1, AR10, AR100,
        APsmall, APmedium, APlarge,
        ARsmall, ARmedium, ARlarge.

        When no ground-truth can be associated with a particular class (NPOS == 0),
        that class is removed from the average calculation.
        If for a given calculation, no metrics whatsoever are available, returns NaN.

    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
    Returns:
            A dictionary with one entry for each metric.
    """

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    def _evaluate(iou_threshold, max_dets, area_range):
        # accumulate evaluations on a per-class basis
        _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})
        for img_id, class_id in _bbs:
            ev = _evaluate_image(
                _bbs[img_id, class_id]["dt"],
                _bbs[img_id, class_id]["gt"],
                _ious[img_id, class_id],
                iou_threshold,
                max_dets,
                area_range,
            )
            acc = _evals[class_id]
            acc["scores"].append(ev["scores"])
            acc["matched"].append(ev["matched"])
            acc["NP"].append(ev["NP"])

        # now reduce accumulations
        for class_id in _evals:
            acc = _evals[class_id]
            acc["scores"] = np.concatenate(acc["scores"])
            acc["matched"] = np.concatenate(acc["matched"]).astype('bool')
            acc["NP"] = np.sum(acc["NP"])

        res = []
        # run ap calculation per-class
        for class_id in _evals:
            ev = _evals[class_id]
            res.append({
                "class": class_id,
                **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"]),
            })
        return res

    iou_thresholds = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)

    # compute simple AP with all thresholds, using up to 100 dets, and all areas
    full = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(0, np.inf))
        for i in iou_thresholds
    }

    AP50 = np.mean([x['AP'] for x in full[0.50] if x['AP'] is not None])
    AP75 = np.mean([x['AP'] for x in full[0.75] if x['AP'] is not None])
    AP = np.mean([x['AP'] for k in full for x in full[k] if x['AP'] is not None])

    TP50 = np.sum([x['TP'] for x in full[0.50] if x['AP'] is not None])
    FP50 = np.sum([x['FP'] for x in full[0.50] if x['AP'] is not None])
    FN50 = np.sum([x['total positives']-x['TP'] for x in full[0.50] if x['AP'] is not None])
    TP75 = np.sum([x['TP'] for x in full[0.75] if x['AP'] is not None])
    FP75 = np.sum([x['FP'] for x in full[0.75] if x['AP'] is not None])
    FN75 = np.sum([x['total positives']-x['TP'] for x in full[0.75] if x['AP'] is not None])

    P = np.sum([x['total positives'] for x in full[0.75] if x['AP'] is not None])

    # max recall for 100 dets can also be calculated here
    AR100 = np.mean(
        [x['TP'] / x['total positives'] for k in full for x in full[k] if x['TP'] is not None])

    small = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(0, 32**2))
        for i in iou_thresholds
    }
    APsmall = [x['AP'] for k in small for x in small[k] if x['AP'] is not None]
    APsmall = np.nan if APsmall == [] else np.mean(APsmall)
    ARsmall = [
        x['TP'] / x['total positives'] for k in small for x in small[k] if x['TP'] is not None
    ]
    ARsmall = np.nan if ARsmall == [] else np.mean(ARsmall)

    medium = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(32**2, 96**2))
        for i in iou_thresholds
    }
    APmedium = [x['AP'] for k in medium for x in medium[k] if x['AP'] is not None]
    APmedium = np.nan if APmedium == [] else np.mean(APmedium)
    ARmedium = [
        x['TP'] / x['total positives'] for k in medium for x in medium[k] if x['TP'] is not None
    ]
    ARmedium = np.nan if ARmedium == [] else np.mean(ARmedium)

    large = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(96**2, np.inf))
        for i in iou_thresholds
    }
    APlarge = [x['AP'] for k in large for x in large[k] if x['AP'] is not None]
    APlarge = np.nan if APlarge == [] else np.mean(APlarge)
    ARlarge = [
        x['TP'] / x['total positives'] for k in large for x in large[k] if x['TP'] is not None
    ]
    ARlarge = np.nan if ARlarge == [] else np.mean(ARlarge)

    max_det1 = {
        i: _evaluate(iou_threshold=i, max_dets=1, area_range=(0, np.inf))
        for i in iou_thresholds
    }
    AR1 = np.mean([
        x['TP'] / x['total positives'] for k in max_det1 for x in max_det1[k] if x['TP'] is not None
    ])

    max_det10 = {
        i: _evaluate(iou_threshold=i, max_dets=10, area_range=(0, np.inf))
        for i in iou_thresholds
    }
    AR10 = np.mean([
        x['TP'] / x['total positives'] for k in max_det10 for x in max_det10[k]
        if x['TP'] is not None
    ])

    return {
        "AP": AP,
        "AP50": AP50,
        "AP75": AP75,
        "TP50": TP50,
        "FP50": FP50,
        "FN50": FN50,
        "TP75": TP75,
        "FP75": FP75,
        "FN75": FN75,
        "P":P,
        "APsmall": APsmall,
        "APmedium": APmedium,
        "APlarge": APlarge,
        "AR1": AR1,
        "AR10": AR10,
        "AR100": AR100,
        "ARsmall": ARsmall,
        "ARmedium": ARmedium,
        "ARlarge": ARlarge
    }

def get_pixel_density_summary(groundtruth_bbs, detected_bbs, bins):

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    def _evaluate(iou_threshold, max_dets, area_range, ppe_range):
        # accumulate evaluations on a per-class basis
        _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})
        for img_id, class_id in _bbs:
            ev = _evaluate_image(
                _bbs[img_id, class_id]["dt"],
                _bbs[img_id, class_id]["gt"],
                _ious[img_id, class_id],
                iou_threshold,
                max_dets,
                area_range,
                ppe_range
            )
            acc = _evals[class_id]
            acc["scores"].append(ev["scores"])
            acc["matched"].append(ev["matched"])
            acc["NP"].append(ev["NP"])

        # now reduce accumulations
        for class_id in _evals:
            acc = _evals[class_id]
            acc["scores"] = np.concatenate(acc["scores"])
            acc["matched"] = np.concatenate(acc["matched"]).astype('bool')
            acc["NP"] = np.sum(acc["NP"])

        res = []
        # run ap calculation per-class
        for class_id in _evals:
            ev = _evals[class_id]
            res.append({
                "class": class_id,
                **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"]),
            })
        return res

    iou_thresholds = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
    results = []
    for j in range(len(bins)-1):
        scores = {
            i: _evaluate(iou_threshold=i, max_dets=100, ppe_range=(bins[j], bins[j+1]), area_range=(0, np.inf))
            for i in iou_thresholds
        }

        AP50 = np.mean([x['AP'] for x in scores[0.50] if x['AP'] is not None])
        AP75 = np.mean([x['AP'] for x in scores[0.75] if x['AP'] is not None])
        AP = np.mean([x['AP'] for k in scores for x in scores[k] if x['AP'] is not None])
        AR100 = np.mean(
            [x['TP'] / x['total positives'] for k in scores for x in scores[k] if x['TP'] is not None])

        TP = np.zeros(len(scores))
        FP = np.zeros(len(scores))
        P = 0
        for category in scores[0.5]:
            class_id = category['class']
            if scores[0.5][class_id]['total positives'] is not None:
                TP += np.array([int(scores[IOU_t][class_id]['TP']) for IOU_t in scores])
                FP += np.array([int(scores[IOU_t][class_id]['FP']) for IOU_t in scores])
                P += int(scores[0.5][class_id]['total positives'])
            else:
                continue
                

        results.append({
            "AP": AP.item() if not np.isnan(AP) else 'nan',
            "AP50": AP50.item() if not np.isnan(AP50) else 'nan',
            "AP75": AP75.item() if not np.isnan(AP75) else 'nan',
            "AR100": AR100.item() if not np.isnan(AR100) else 'nan',
            "P": P,
            "TP": TP.tolist(),
            "FP": FP.tolist()
        })

    return results



def get_coco_summary2(groundtruth_bbs, detected_bbs):
    """Calculate the 12 standard metrics used in COCOEval,
        AP, AP50, AP75,
        AR1, AR10, AR100,
        APsmall, APmedium, APlarge,
        ARsmall, ARmedium, ARlarge.

        When no ground-truth can be associated with a particular class (NPOS == 0),
        that class is removed from the average calculation.
        If for a given calculation, no metrics whatsoever are available, returns NaN.

    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
    Returns:
            A dictionary with one entry for each metric.
    """

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    def _evaluate(iou_threshold, max_dets, area_range):
        # accumulate evaluations on a per-class basis
        _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})
        for img_id, class_id in _bbs:
            ev = _evaluate_image(
                _bbs[img_id, class_id]["dt"],
                _bbs[img_id, class_id]["gt"],
                _ious[img_id, class_id],
                iou_threshold,
                max_dets,
                area_range,
            )
            acc = _evals[class_id]
            acc["scores"].append(ev["scores"])
            acc["matched"].append(ev["matched"])
            acc["NP"].append(ev["NP"])

        # now reduce accumulations
        for class_id in _evals:
            acc = _evals[class_id]
            acc["scores"] = np.concatenate(acc["scores"])
            acc["matched"] = np.concatenate(acc["matched"]).astype('bool')
            acc["NP"] = np.sum(acc["NP"])

        res = []
        # run ap calculation per-class
        for class_id in _evals:
            ev = _evals[class_id]
            res.append({
                "class": class_id,
                **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"]),
            })
        return res

    iou_thresholds = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)

    # compute simple AP with all thresholds, using up to 100 dets, and all areas
    full = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(0, np.inf))
        for i in iou_thresholds
    }
    full50 = full[0.50]
    full75 = full[0.75]
    class_types = [x['class'] for x in full50]
    result = {}
    
    for class_id in range(len(class_types)):
        AP50 = full50[class_id]['AP']
        AP75 = full75[class_id]['AP']
        AP = [full[k][class_id]['AP'] for k in full if full[k][class_id] is not None]
        if AP is not None and len(AP)>0:
            if AP[0] is not None:
                AP = np.mean(AP)
            else:
                AP = None
        else:
            AP = None

        TP = [int(full[IOU_t][class_id]['TP']) for IOU_t in full if full[IOU_t][class_id]['TP'] is not None]
        FP = [int(full[IOU_t][class_id]['FP']) for IOU_t in full if full[IOU_t][class_id]['TP'] is not None]

        P = full50[class_id]['total positives']
        if P is not None:
            P = int(P)
            FN = [int(P-TP_i) for TP_i in TP if TP_i is not None]
            Recall = [TP_i/P for TP_i in TP if TP_i is not None]
            Precision = []
            for i, TP_i in enumerate(TP):
                if TP_i is not None:
                    if TP_i + FP[i] > 0:
                        Precision.append(TP_i/(TP_i+FP[i]))
                    else:
                        Precision.append(-1)
            F1 = [(2*Precision[i]*Recall[i]/(Precision[i]+Recall[i])) for i in range(len(TP)) if (Precision[i]+Recall[i])>0]
        else:
            P = None
            FN = []
            Recall = []
            Precision = []
            F1 = []
        
        # max recall for 100 dets can also be calculated here
        if P is None:
            AR100 = None
        else:
            AR100 = [full[k][class_id]['TP'] / full[k][class_id]['total positives'] for k in full]
            AR100 = np.mean(AR100)
            
        small = {
            i: _evaluate(iou_threshold=i, max_dets=100, area_range=(0, 32**2))
            for i in iou_thresholds
        }
        APsmall  = [small[k][class_id]['AP'] for k in small if small[k][class_id] is not None]
        if APsmall:
            if APsmall[0] is not None:
                APsmall = np.mean(APsmall)
                ARsmall = [small[k][class_id]['TP'] / small[k][class_id]['total positives'] for k in small]
                ARsmall = np.mean(ARsmall) if ARsmall[0] is not None else None
            else:
                ARsmall = None
        else:
            APsmall = None
            ARsmall =None
        
        medium = {
            i: _evaluate(iou_threshold=i, max_dets=100, area_range=(32**2, 96**2))
            for i in iou_thresholds
        }

        APmedium  = [medium[k][class_id]['AP'] for k in medium if medium[k][class_id] is not None]
        if APmedium:
            if APmedium[0] is not None:
                APmedium = np.mean(APmedium)
                ARmedium = [medium[k][class_id]['TP'] / medium[k][class_id]['total positives'] for k in medium]
                ARmedium = np.mean(ARmedium) if ARmedium[0] is not None else None
            else:
                ARmedium = None
        else:
            APmedium = None
            ARmedium =None

        large = {
            i: _evaluate(iou_threshold=i, max_dets=100, area_range=(96**2, np.inf))
            for i in iou_thresholds
        }
        APlarge = [large[k][class_id]['AP'] for k in large if large[k][class_id] is not None]
        if APlarge:
            if APlarge[0] is not None:
                APlarge = np.mean(APlarge)
                ARlarge = [large[k][class_id]['TP'] / large[k][class_id]['total positives'] for k in large]
                ARlarge = np.mean(ARlarge) if ARlarge[0] is not None else None
            else:
                ARlarge = None
        else:
            APlarge = None
            ARlarge =None
       
        max_det1 = {
            i: _evaluate(iou_threshold=i, max_dets=1, area_range=(0, np.inf))
            for i in iou_thresholds
        }
        if max_det1[0.50][class_id]['total positives'] is not None:
            AR1 = np.mean([max_det1[k][class_id]['TP'] / max_det1[k][class_id]['total positives'] for k in max_det1])
        else:
            AR1 = None

        result[class_types[class_id]] = {
            "AP": AP,
            "AP50": AP50,
            "AP75": AP75,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "P": P,
            "Precision": Precision,
            "Recall": Recall,
            "F1": F1,
            "APsmall": APsmall,
            "APmedium": APmedium,
            "APlarge": APlarge,
            "AR1": AR1,
            "AR100": AR100,
            "ARsmall": ARsmall,
            "ARmedium": ARmedium,
            "ARlarge": ARlarge
        }
    result['All_classes'] = {}
    for key in result[class_types[0]]:
        tmp = [result[class_id][key] for class_id in class_types]
        tmp = list(filter(lambda x: x is not None, tmp))
        tmp = list(filter(lambda x: not isinstance(x, list), tmp))
        if key in ['TP','FP','FN','TP50', 'TP75', 'FP50', 'FP75', 'FN50', 'FN75', 'P']:
            result['All_classes'][key] = np.sum(tmp)
        else:
            result['All_classes'][key] = np.mean(tmp)
    return result

def get_coco_metrics(
        groundtruth_bbs,
        detected_bbs,
        iou_threshold=0.5,
        area_range=(0, np.inf),
        ppe_range=(-np.inf, np.inf),
        max_dets=100,
):
    """ Calculate the Average Precision and Recall metrics as in COCO's official implementation
        given an IOU threshold, area range and maximum number of detections.
    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
            iou_threshold : float
                Intersection Over Union (IOU) value used to consider a TP detection.
            area_range : (numerical x numerical)
                Lower and upper bounds on annotation areas that should be considered.
            max_dets : int
                Upper bound on the number of detections to be considered for each class in an image.

    Returns:
            A list of dictionaries. One dictionary for each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['TP']: total number of True Positive detections;
            dict['FP']: total number of False Positive detections;

            if there was no valid ground truth for a specific class (total positives == 0),
            all the associated keys default to None
    """

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    # accumulate evaluations on a per-class basis
    _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})

    for img_id, class_id in _bbs:
        ev = _evaluate_image(
            _bbs[img_id, class_id]["dt"],
            _bbs[img_id, class_id]["gt"],
            _ious[img_id, class_id],
            iou_threshold,
            max_dets,
            area_range,
            ppe_range
        )
        acc = _evals[class_id]
        acc["scores"].append(ev["scores"])
        acc["matched"].append(ev["matched"])
        acc["NP"].append(ev["NP"])

    # now reduce accumulations
    for class_id in _evals:
        acc = _evals[class_id]
        acc["scores"] = np.concatenate(acc["scores"])
        acc["matched"] = np.concatenate(acc["matched"]).astype('bool')
        acc["NP"] = np.sum(acc["NP"])

    res = {}
    # run ap calculation per-class
    for class_id in _evals:
        ev = _evals[class_id]
        res[class_id] = {
            "class": class_id,
            **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"])
        }
    return res


def _group_detections(dt, gt):
    """ simply group gts and dts on a imageXclass basis """
    bb_info = defaultdict(lambda: {"dt": [], "gt": []})
    for d in dt:
        i_id = d.get_image_name()
        c_id = d.get_class_id()
        bb_info[i_id, c_id]["dt"].append(d)
    for g in gt:
        i_id = g.get_image_name()
        c_id = g.get_class_id()
        bb_info[i_id, c_id]["gt"].append(g)
    return bb_info


def _get_area(a):
    """ COCO does not consider the outer edge as included in the bbox """
    x, y, x2, y2 = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    return (x2 - x) * (y2 - y)


def _jaccard(a, b):
    xa, ya, x2a, y2a = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    xb, yb, x2b, y2b = b.get_absolute_bounding_box(format=BBFormat.XYX2Y2)

    # innermost left x
    xi = max(xa, xb)
    # innermost right x
    x2i = min(x2a, x2b)
    # same for y
    yi = max(ya, yb)
    y2i = min(y2a, y2b)

    # calculate areas
    Aa = max(x2a - xa, 0) * max(y2a - ya, 0)
    Ab = max(x2b - xb, 0) * max(y2b - yb, 0)
    Ai = max(x2i - xi, 0) * max(y2i - yi, 0)
    return Ai / (Aa + Ab - Ai)


def _compute_ious(dt, gt):
    """ compute pairwise ious """

    ious = np.zeros((len(dt), len(gt)))
    for g_idx, g in enumerate(gt):
        for d_idx, d in enumerate(dt):
            ious[d_idx, g_idx] = _jaccard(d, g)
    return ious


def _evaluate_image(dt, gt, ious, iou_threshold, max_dets=None, area_range=None, ppe_range=None):
    """ use COCO's method to associate detections to ground truths """
    # sort dts by increasing confidence
    dt_sort = np.argsort([-d.get_confidence() for d in dt], kind="stable")

    # sort list of dts and chop by max dets
    dt = [dt[idx] for idx in dt_sort[:max_dets]]
    ious = ious[dt_sort[:max_dets]]

    # generate ignored gt list by area_range
    def _is_ignore(bb, area_range = None, ppe_range = None):
        if area_range is None:
            area_range = [0, np.inf]
        if ppe_range is None:
            ppe_range = [-np.inf, np.inf]

        return not (area_range[0] <= _get_area(bb) <= area_range[1] and ppe_range[0]<=bb.get_ppe()<=ppe_range[1])

    gt_ignore = [_is_ignore(g, area_range, ppe_range) for g in gt]

    # sort gts by ignore last
    gt_sort = np.argsort(gt_ignore, kind="stable")
    gt = [gt[idx] for idx in gt_sort]
    gt_ignore = [gt_ignore[idx] for idx in gt_sort]
    ious = ious[:, gt_sort]

    gtm = {}
    dtm = {}

    for d_idx, d in enumerate(dt):
        # information about best match so far (m=-1 -> unmatched)
        iou = min(iou_threshold, 1 - 1e-10)
        m = -1
        for g_idx, g in enumerate(gt):
            # if this gt already matched, and not a crowd, continue
            if g_idx in gtm:
                continue
            # if dt matched to reg gt, and on ignore gt, stop
            if m > -1 and gt_ignore[m] == False and gt_ignore[g_idx] == True:
                break
            # continue to next gt unless better match made
            if ious[d_idx, g_idx] < iou:
                continue
            # if match successful and best so far, store appropriately
            iou = ious[d_idx, g_idx]
            m = g_idx
        # if match made store id of match for both dt and gt
        if m == -1:
            continue
        dtm[d_idx] = m
        gtm[m] = d_idx

    # generate ignore list for dts
    dt_ignore = [
        gt_ignore[dtm[d_idx]] if d_idx in dtm else _is_ignore(d) for d_idx, d in enumerate(dt)
    ]

    # get score for non-ignored dts
    scores = [dt[d_idx].get_confidence() for d_idx in range(len(dt)) if not dt_ignore[d_idx]]
    matched = [d_idx in dtm for d_idx in range(len(dt)) if not dt_ignore[d_idx]]

    n_gts = len([g_idx for g_idx in range(len(gt)) if not gt_ignore[g_idx]])
    return {"scores": scores, "matched": matched, "NP": n_gts}


def _compute_ap_recall(scores, matched, NP, recall_thresholds=None):
    """ This curve tracing method has some quirks that do not appear when only unique confidence thresholds
    are used (i.e. Scikit-learn's implementation), however, in order to be consistent, the COCO's method is reproduced. """
    if NP == 0:
        return {
            "precision": None,
            "recall": None,
            "AP": None,
            "interpolated precision": None,
            "interpolated recall": None,
            "total positives": None,
            "TP": None,
            "FP": None
        }

    # by default evaluate on 101 recall levels
    if recall_thresholds is None:
        recall_thresholds = np.linspace(0.0,
                                        1.00,
                                        int(np.round((1.00 - 0.0) / 0.01)) + 1,
                                        endpoint=True)

    # sort in descending score order
    inds = np.argsort(-scores, kind="stable")

    scores = scores[inds]
    matched = matched[inds]

    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)

    rc = tp / NP
    pr = tp / (tp + fp)

    # make precision monotonically decreasing
    i_pr = np.maximum.accumulate(pr[::-1])[::-1]

    rec_idx = np.searchsorted(rc, recall_thresholds, side="left")
    n_recalls = len(recall_thresholds)

    # get interpolated precision values at the evaluation thresholds
    i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

    return {
        "precision": pr,
        "recall": rc,
        "AP": np.mean(i_pr),
        "interpolated precision": i_pr,
        "interpolated recall": recall_thresholds,
        "total positives": NP,
        "TP": tp[-1] if len(tp) != 0 else 0,
        "FP": fp[-1] if len(fp) != 0 else 0
    }