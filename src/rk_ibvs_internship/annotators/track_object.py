import copy
from timeit import default_timer

import cv2
import py_trees
import robokudo.annotators
import robokudo.types.scene
import robokudo.utils.cv_helper
from robokudo.cas import CASViews
from ultralytics import YOLO, SAM
from copy import deepcopy


class TrackObject(robokudo.annotators.core.BaseAnnotator):
    """Crops and rotates the color image"""

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
                self.slice_x = slice(70, 400)
                self.slice_y = slice(40, 600)

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="TrackObject", descriptor=Descriptor()):
        """
        Default construction. Minimal one-time init!
        """
        super().__init__(name, descriptor)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

        self.tracker = cv2.TrackerCSRT_create()


    def update(self):
        start_timer = default_timer()

        # Read color image from the cas
        color = deepcopy(self.get_cas().get(CASViews.COLOR_IMAGE))

        object_hypothesis = self.get_cas().filter_annotations_by_type(robokudo.types.scene.ObjectHypothesis)

        tracked = False

        for hypothesis in object_hypothesis:
            assert isinstance(hypothesis, robokudo.types.scene.ObjectHypothesis)
            class_name = hypothesis.classification.classname

            if class_name == 'Crackerbox' and not tracked:
                roi = hypothesis.roi.roi
                self.tracker.init(color, (roi.pos.x, roi.pos.y, roi.width, roi.height))
                tracked = True
                continue
            # ok is a boolean indicating if the update was successful
        ok, roi = self.tracker.update(color)

                # Draw bounding box on the frame  using the coordinates of the updated bbox
        if ok:
            p1 = (int(roi[0]), int(roi[1]))
            p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
            cv2.rectangle(color, p1, p2, (255, 0, 0), 2, 1)



        # # crop it
        # cropped = color[self.descriptor.parameters.slice_x, self.descriptor.parameters.slice_y]
        #
        # # rotate it
        # rotated = cv2.rotate(cropped, self.descriptor.parameters.rotate)
        #
        # visualize it in the robokudi gui
        self.get_annotator_output_struct().set_image(color)
        #
        # # update the cas with the rotated image
        # self.get_cas().set(CASViews.COLOR_IMAGE, rotated)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.Status.SUCCESS
