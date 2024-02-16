from timeit import default_timer
import cv2
import py_trees
import robokudo.annotators
import robokudo.types.scene
import robokudo.utils.cv_helper
from robokudo.cas import CASViews
from ultralytics import SAM
from copy import deepcopy
import numpy as np
import torch


class SAMTrackObject(robokudo.annotators.core.BaseAnnotator):
    """Tracks the object by creating a bounding box when YOLO doesn't give one"""

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
                self.slice_x = slice(70, 400)
                self.slice_y = slice(40, 600)

                self.sam_model = "mobile_sam.pt"
                self.precision_mode = True

                self.classname = None


        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="SAMTrackObject", descriptor=Descriptor()):
        """
        Default construction. Minimal one-time init!
        """
        super(SAMTrackObject, self).__init__(name, descriptor)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

        # self.tracker = cv2.TrackerCSRT_create()
        self.tracker = cv2.legacy.TrackerCSRT_create()

        if descriptor.parameters.precision_mode:
            self.sam = SAM(self.descriptor.parameters.sam_model)


    def update(self):
        cam_to_world_transform = self.get_cas().get(robokudo.cas.CASViews.VIEWPOINT_CAM_TO_WORLD)


        start_timer = default_timer()

        # Read color image from the cas
        color = deepcopy(self.get_cas().get(CASViews.COLOR_IMAGE))

        self.color = self.get_cas().get(CASViews.COLOR_IMAGE)

        object_hypothesis_list = self.get_cas().filter_annotations_by_type(robokudo.types.scene.ObjectHypothesis)

        for hypothesis in object_hypothesis_list:
            assert isinstance(hypothesis, robokudo.types.scene.ObjectHypothesis)
            class_name = hypothesis.classification.classname

            # if class_name == 'Crackerbox':
            if class_name == self.descriptor.parameters.classname:
                roi = hypothesis.roi.roi
                self.tracker.init(color, (roi.pos.x, roi.pos.y, roi.width, roi.height))
                return py_trees.Status.SUCCESS



        # ok is a boolean indicating if the update was successful
        # roi is being overwritten and it is now the box coming from the tracker not the box coming from yolo
        # roi is a tuple containing (x,y,width,height) of the selected bounding box
        ok, roi = self.tracker.update(color)

        # Draw bounding box on the frame  using the coordinates of the updated bbox
        if ok:
            p1 = (int(roi[0]), int(roi[1]))
            p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
            cv2.rectangle(color, p1, p2, (255, 0, 0), 2, 1)



            object_hypothesis = robokudo.types.scene.ObjectHypothesis()
            object_hypothesis.type = self.descriptor.parameters.classname
            # object_hypothesis.type = 'Crackerbox'

            object_hypothesis.roi.roi.pos.x = roi[0]
            object_hypothesis.roi.roi.pos.y = roi[1]
            object_hypothesis.roi.roi.width = roi[2]
            object_hypothesis.roi.roi.height = roi[3]

            x1 = int(object_hypothesis.roi.roi.pos.x)
            y1 = int(object_hypothesis.roi.roi.pos.y)
            x2 = int(object_hypothesis.roi.roi.width + x1)
            y2 = int(object_hypothesis.roi.roi.height + y1)

            # x1 = object_hypothesis.roi.roi.pos.x
            # y1 = object_hypothesis.roi.roi.pos.y
            # x2 = object_hypothesis.roi.roi.width + x1
            # y2 = object_hypothesis.roi.roi.height + y1

            w = int(object_hypothesis.roi.roi.width)
            h = int(object_hypothesis.roi.roi.height)

            # w = object_hypothesis.roi.roi.width
            # h = object_hypothesis.roi.roi.height

            # object_hypothesis.bbox = [x1, y1, x2, y2]



            if not self.descriptor.parameters.precision_mode:
                mask = np.zeros_like(self.color, dtype=np.uint8)
                mask[int(y1): int(y2), int(x1): int(x2)] = 1
            else:
                masks = self.sam.predict(self.color, bboxes=[x1, x2, y1, y2], labels=[1])[0].masks.data.cpu().numpy()
                mask = masks[0].astype(np.uint8)
            torch.cuda.empty_cache()

            object_hypothesis.roi.mask = mask

            object_hypothesis.roi.mask *= 255  # SAM outputs masks with 1. Scale to 255.

            # the try except was added
            object_hypothesis.roi.mask = \
                robokudo.utils.cv_helper.crop_image(object_hypothesis.roi.mask, (int(x1), int(y1)), (int(w), int(h)))
                # robokudo.utils.cv_helper.crop_image(object_hypothesis.roi.mask, (x1, y1), (w, h))



            classification = robokudo.types.annotation.Classification()
            classification.classname = self.descriptor.parameters.classname
            # classification.classname = 'Crackerbox'
            object_hypothesis.classification = classification
            self.get_cas().annotations.append(object_hypothesis)

            full_mask = np.zeros((color.shape[0], color.shape[1]), dtype="uint8")
            full_mask[y1: y1 + object_hypothesis.roi.mask.shape[0], x1: x1 + object_hypothesis.roi.mask.shape[1]] = object_hypothesis.roi.mask

            color[full_mask == 255] = [0, 255, 0]

            # self.tracked = False



        # visualize it in the robokudi gui
        self.get_annotator_output_struct().set_image(color)
        #
        # # update the cas with the rotated image
        # self.get_cas().set(CASViews.COLOR_IMAGE, rotated)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.Status.SUCCESS
