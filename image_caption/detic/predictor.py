# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import copy
import multiprocessing as mp
from collections import deque
import cv2
import torch
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

from .modeling.utils import reset_cls_test


def get_clip_embeddings(vocabulary, prompt='a '):
    from .modeling.text.text_encoder import build_text_encoder
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

BUILDIN_CLASSIFIER = {
    'lvis': '/home/shuijing/Documents/Detic/datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': '/home/shuijing/Documents/Detic/datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': '/home/shuijing/Documents/Detic/datasets/metadata/oid_clip_a+cname.npy',
    'coco': '/home/shuijing/Documents/Detic/datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}

class VisualizationDemo(object):
    def __init__(self, cfg, args, 
        instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        if args.vocabulary == 'custom':
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = args.custom_vocabulary.split(',')
            classifier = get_clip_embeddings(self.metadata.thing_classes)
        else:
            self.metadata = MetadataCatalog.get(
                BUILDIN_METADATA_PATH[args.vocabulary])
            classifier = BUILDIN_CLASSIFIER[args.vocabulary]

        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)
        reset_cls_test(self.predictor.model, classifier, num_classes)

    def run_on_image(self, image, args):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
            args: the arguments from demo.py
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        if args.keep_top_k_instances:
            end_idx = min(predictions['instances']._fields['scores'].size(dim=0), args.instance_k_value)
            for key in ['pred_boxes', 'scores', 'pred_classes', 'pred_masks']:
                predictions['instances']._fields[key] = predictions['instances']._fields[key][:end_idx]
        elif args.keep_top_k_classes:
            bbox_list_np = predictions['instances']._fields['pred_boxes'].tensor.cpu().numpy().astype(int)
            bbox_sizes = [(y1-y0)*(x1-x0) for y0, x0, y1, x1 in bbox_list_np]
            bbox_size_avg_class = {}
            pred_classes_np = predictions['instances']._fields['pred_classes'].cpu().numpy().astype(int)
            for class_id in np.unique(pred_classes_np):
                idx = np.where(pred_classes_np == class_id)[0]
                bbox_size_avg_class[int(class_id)] = np.mean([bbox_sizes[i] for i in idx])
            sorted_bbox_size_avg_class = sorted(bbox_size_avg_class.items(), key=lambda x:x[1], reverse=True)
            sorted_bbox_size_avg_class = sorted_bbox_size_avg_class[:args.class_k_value]
            sorted_bbox_size_avg_class = dict(sorted_bbox_size_avg_class)

            remain_idx = [i for i in range(len(bbox_list_np)) if predictions['instances']._fields['pred_classes'][i].item() in sorted_bbox_size_avg_class.keys()]
            for key in ['pred_boxes', 'scores', 'pred_classes', 'pred_masks']:
                predictions['instances']._fields[key] = predictions['instances']._fields[key][remain_idx]

        # todo: add non-maximum suppression for each class here
        #  if two instances belong to the same class AND their IOU > threshold, then remove the smaller one
        if args.non_max_suppress:
            # # [num obj, 4]
            # bbox_list_np = predictions['instances']._fields['pred_boxes'].tensor.cpu().numpy().astype(int)
            # # [num obj,]
            # bbox_sizes = np.array([(y1 - y0) * (x1 - x0) for y0, x0, y1, x1 in bbox_list_np])
            # # sort predictions by area size, not by score
            # sorted_idx = [i[0] for i in sorted(enumerate(bbox_sizes), key=lambda x:x[1], reverse=True)]
            # for key in ['pred_boxes', 'scores', 'pred_classes', 'pred_masks']:
            #     predictions['instances']._fields[key] = predictions['instances']._fields[key][sorted_idx]

            # calculate all bboxs and their sizes again after sorting
            # [num obj, 4]
            bbox_list_np = predictions['instances']._fields['pred_boxes'].tensor.cpu().numpy().astype(int)
            # [num obj,]
            bbox_sizes = np.array([(y1 - y0) * (x1 - x0) for y0, x0, y1, x1 in bbox_list_np])
            # [num obj,]
            pred_classes_np = predictions['instances']._fields['pred_classes'].cpu().numpy().astype(int)
            all_remain_idx = []
            # for each class of objects
            for class_id in np.unique(pred_classes_np):
                idx = np.where(pred_classes_np == class_id)[0]
                print(idx)
                # if there are more than one instances, do non-maximum suppression
                if len(idx) > 1:
                    remain_idx = self.non_maximum_suppression(bbox_list_np, idx, bbox_sizes, 0.9)
                else: # else the only instance will always be kept
                    remain_idx = idx
                all_remain_idx.extend(copy.deepcopy(remain_idx))
            for key in ['pred_boxes', 'scores', 'pred_classes', 'pred_masks']:
                predictions['instances']._fields[key] = predictions['instances']._fields[key][all_remain_idx]

        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        self.visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = self.visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = self.visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = self.visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def non_maximum_suppression(self, boxes, indices, areas, threshold):
        return_indices = copy.deepcopy(indices)
        for idx in indices:
            # Create temporary indices
            temp_indices = indices[indices != idx]
            # Find out the coordinates of the intersection box
            xx1 = np.maximum(boxes[idx, 1], boxes[temp_indices, 1])
            yy1 = np.maximum(boxes[idx, 0], boxes[temp_indices, 0])
            xx2 = np.minimum(boxes[idx, 3], boxes[temp_indices, 3])
            yy2 = np.minimum(boxes[idx, 2], boxes[temp_indices, 2])
            # Find out the width and the height of the intersection box
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            # # compute the ratio of overlap
            # overlap = (w * h) / areas[temp_indices]
            # # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index
            # if np.any(overlap > threshold):
            #     return_indices = return_indices[return_indices != idx]
            overlap = (w * h) / areas[temp_indices]
            for i in range(len(overlap)):
                if overlap[i] > threshold:
                    return_indices = return_indices[return_indices != temp_indices[i]]
            # return only the boxes at the remaining indices
        return return_indices

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
