import numpy as np
import os
import logging
import glob
import cv2

# ------------------ 基础计算函数 ------------------

def cal_global_acc(pred, gt):
    h, w = gt.shape
    return [np.sum(pred == gt), float(h * w)]

def get_statistics_seg(pred, gt, num_cls=2):
    h, w = gt.shape
    statistics = []
    for i in range(num_cls):
        tp = np.sum((pred == i) & (gt == i))
        fp = np.sum((pred == i) & (gt != i))
        fn = np.sum((pred != i) & (gt == i))
        statistics.append([tp, fp, fn])
    return statistics

def get_statistics_prf(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return [tp, fp, fn]

def get_statistics(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return [tp, fp, fn]

# ------------------ 语义分割指标（全局准确率/均值IoU） ------------------

def segment_metrics(pred_list, gt_list, num_cls=2):
    global_accuracy_cur = []
    statistics = []

    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        pred_img = ((pred / 255) > 0.5).astype('uint8')
        global_accuracy_cur.append(cal_global_acc(pred_img, gt_img))
        statistics.append(get_statistics_seg(pred_img, gt_img, num_cls))

    global_acc = np.sum([v[0] for v in global_accuracy_cur]) / np.sum([v[1] for v in global_accuracy_cur])
    counts = []
    for i in range(num_cls):
        tp = np.sum([v[i][0] for v in statistics])
        fp = np.sum([v[i][1] for v in statistics])
        fn = np.sum([v[i][2] for v in statistics])
        counts.append([tp, fp, fn])

    mean_acc = np.sum([v[0] / (v[0] + v[2] + 1e-8) for v in counts]) / num_cls
    mean_iou_acc = np.sum([v[0] / (np.sum(v) + 1e-8) for v in counts]) / num_cls
    return global_acc, mean_acc, mean_iou_acc

# ------------------ Precision / Recall / F1 ------------------

def prf_metrics(pred_list, gt_list):
    statistics = []
    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        pred_norm = pred.astype(np.float32)
        if pred_norm.max() > 1:
            pred_norm = pred_norm / 255.0
        pred_img = (pred_norm > 0.5).astype('uint8')
        statistics.append(get_statistics_prf(pred_img, gt_img))

    tp = np.sum([v[0] for v in statistics])
    fp = np.sum([v[1] for v in statistics])
    fn = np.sum([v[2] for v in statistics])

    p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp + 1e-8)
    r_acc = tp / (tp + fn + 1e-8)
    f_acc = 0 if (p_acc + r_acc) == 0 else 2 * p_acc * r_acc / (p_acc + r_acc)
    return p_acc, r_acc, f_acc

# ------------------ 逐阈值评估（PRF随阈值变化） ------------------

def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        statistics = []
        for pred, gt in zip(pred_list, gt_list):
            pred_norm = pred.astype(np.float32)
            if pred_norm.max() > 1:
                pred_norm = pred_norm / 255.0
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred_norm > thresh).astype('uint8')
            statistics.append(get_statistics(pred_img, gt_img))

        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])
        p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp + 1e-8)
        r_acc = tp / (tp + fn + 1e-8)
        f_acc = 0 if (p_acc + r_acc) == 0 else 2 * p_acc * r_acc / (p_acc + r_acc)
        final_accuracy_all.append([thresh, p_acc, r_acc, f_acc])
    return final_accuracy_all

# ------------------ OIS、ODS、mIoU ------------------

def cal_OIS_metrics(pred_list, gt_list, thresh_step=0.01):
    final_F1_list = []
    for pred, gt in zip(pred_list, gt_list):
        F1_list = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            pred_norm = pred.astype(np.float32)
            if pred_norm.max() > 1:
                pred_norm = pred_norm / 255.0
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred_norm > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp + 1e-8)
            r_acc = tp / (tp + fn + 1e-8)
            F1 = 0 if (p_acc + r_acc) == 0 else 2 * p_acc * r_acc / (p_acc + r_acc)
            F1_list.append(F1)
        max_F1 = np.max(F1_list)
        final_F1_list.append(max_F1)
    final_F1 = np.mean(final_F1_list)
    return final_F1

def cal_ODS_metrics(pred_list, gt_list, thresh_step=0.01):
    final_ODS = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        ODS_list = []
        for pred, gt in zip(pred_list, gt_list):
            pred_norm = pred.astype(np.float32)
            if pred_norm.max() > 1:
                pred_norm = pred_norm / 255.0
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred_norm > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp + 1e-8)
            r_acc = tp / (tp + fn + 1e-8)
            F1 = 0 if (p_acc + r_acc) == 0 else 2 * p_acc * r_acc / (p_acc + r_acc)
            ODS_list.append(F1)
        ave_F1 = np.mean(np.array(ODS_list))
        final_ODS.append(ave_F1)
    ODS = np.max(np.array(final_ODS))
    return ODS

def cal_mIoU_metrics(pred_list, gt_list, thresh_step=0.01, pred_imgs_names=None, gt_imgs_names=None):
    final_iou = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_list = []
        for pred, gt in zip(pred_list, gt_list):
            pred_norm = pred.astype(np.float32)
            if pred_norm.max() > 1:
                pred_norm = pred_norm / 255.0
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred_norm > thresh).astype('uint8')
            TP = np.sum((pred_img == 1) & (gt_img == 1))
            TN = np.sum((pred_img == 0) & (gt_img == 0))
            FP = np.sum((pred_img == 1) & (gt_img == 0))
            FN = np.sum((pred_img == 0) & (gt_img == 1))
            if (FN + FP + TP) <= 0:
                iou = 0
            else:
                iou_1 = TP / (FN + FP + TP + 1e-8)
                iou_0 = TN / (FN + FP + TN + 1e-8)
                iou = (iou_1 + iou_0) / 2
            iou_list.append(iou)
        ave_iou = np.mean(iou_list)
        final_iou.append(ave_iou)
    mIoU = np.max(np.array(final_iou))
    return mIoU

# ------------------ 图像加载与配对 ------------------

def imread(path, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
    im = cv2.imread(path, load_mode)
    if im is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if convert_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if load_size > 0:
        im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    if thresh > 0:
        _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    return im

def get_image_pairs(data_dir, suffix_gt='real_B', suffix_pred='fake_B'):
    gt_list = sorted(glob.glob(os.path.join(data_dir, f'*{suffix_gt}.png')))
    pred_list = [ll.replace(suffix_gt, suffix_pred) for ll in gt_list]
    assert len(gt_list) == len(pred_list), f"GT and pred images count mismatch: {len(gt_list)} vs {len(pred_list)}"
    pred_imgs, gt_imgs = [], []
    pred_imgs_names, gt_imgs_names = [], []
    for pred_path, gt_path in zip(pred_list, gt_list):
        pred_imgs.append(imread(pred_path))
        gt_imgs.append(imread(gt_path, thresh=127))
        pred_imgs_names.append(pred_path)
        gt_imgs_names.append(gt_path)
    return pred_imgs, gt_imgs, pred_imgs_names, gt_imgs_names

# ------------------ 评估入口 ------------------

def eval(log_eval, results_dir, epoch):
    suffix_gt = "lab"
    suffix_pred = "pre"
    log_eval.info(results_dir)
    log_eval.info("checkpoints -> " + results_dir)
    src_img_list, tgt_img_list, pred_imgs_names, gt_imgs_names = get_image_pairs(results_dir, suffix_gt, suffix_pred)
    assert len(src_img_list) == len(tgt_img_list)

    final_accuracy_all = np.array(cal_prf_metrics(src_img_list, tgt_img_list))
    Precision_list, Recall_list, F_list = final_accuracy_all[:, 1], final_accuracy_all[:, 2], final_accuracy_all[:, 3]

    best_idx = np.argmax(F_list)
    best_thresh = final_accuracy_all[best_idx, 0]

    mIoU = cal_mIoU_metrics(src_img_list, tgt_img_list, pred_imgs_names=pred_imgs_names, gt_imgs_names=gt_imgs_names)
    ODS = cal_ODS_metrics(src_img_list, tgt_img_list)
    OIS = cal_OIS_metrics(src_img_list, tgt_img_list)

    log_eval.info(f"Best threshold -> {best_thresh:.2f}")
    log_eval.info(f"mIoU -> {mIoU:.4f}")
    log_eval.info(f"ODS -> {ODS:.4f}")
    log_eval.info(f"OIS -> {OIS:.4f}")
    log_eval.info(f"F1 -> {F_list[best_idx]:.4f}")
    log_eval.info(f"Precision -> {Precision_list[best_idx]:.4f}")
    log_eval.info(f"Recall -> {Recall_list[best_idx]:.4f}")
    log_eval.info("eval finish!")

    return {
        'epoch': epoch,
        'mIoU': mIoU,
        'ODS': ODS,
        'OIS': OIS,
        'F1': F_list[best_idx],
        'Precision': Precision_list[best_idx],
        'Recall': Recall_list[best_idx],
        'Best_thresh': best_thresh
    }

# ------------------ 主函数执行 ------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    suffix_gt = "lab"
    suffix_pred = "pre"
    results_dir = "../results/results_test/TUT_results"

    src_img_list, tgt_img_list, pred_imgs_names, gt_imgs_names = get_image_pairs(results_dir, suffix_gt, suffix_pred)
    assert len(src_img_list) == len(tgt_img_list)

    final_accuracy_all = np.array(cal_prf_metrics(src_img_list, tgt_img_list))
    Precision_list, Recall_list, F_list = final_accuracy_all[:, 1], final_accuracy_all[:, 2], final_accuracy_all[:, 3]

    best_idx = np.argmax(F_list)
    best_thresh = final_accuracy_all[best_idx, 0]

    mIoU = cal_mIoU_metrics(src_img_list, tgt_img_list, pred_imgs_names=pred_imgs_names, gt_imgs_names=gt_imgs_names)
    ODS = cal_ODS_metrics(src_img_list, tgt_img_list)
    OIS = cal_OIS_metrics(src_img_list, tgt_img_list)

    print(f"Best threshold: {best_thresh:.2f}")
    print(f"mIoU -> {mIoU:.4f}")
    print(f"ODS -> {ODS:.4f}")
    print(f"OIS -> {OIS:.4f}")
    print(f"F1 -> {F_list[best_idx]:.4f}")
    print(f"P -> {Precision_list[best_idx]:.4f}")
    print(f"R -> {Recall_list[best_idx]:.4f}")
    print("✅ Evaluation finished successfully.")