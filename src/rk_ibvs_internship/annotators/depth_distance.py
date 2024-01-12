import copy
from timeit import default_timer

import cv2
import py_trees
import robokudo.types.scene
import robokudo.utils.cv_helper
from robokudo.cas import CASViews
import numpy as np
from copy import deepcopy


# This annotator works for the box


class DepthDistance(robokudo.annotators.core.BaseAnnotator):
    """Crops and rotates the color image"""

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
                self.slice_x = slice(70, 400)
                self.slice_y = slice(40, 600)

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="DepthDistance", descriptor=Descriptor()):
        """
        Default construction. Minimal one-time init!
        """
        super().__init__(name, descriptor)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self):
        start_timer = default_timer()

        # Read color and depth image from the cas
        color = deepcopy(self.get_cas().get(CASViews.COLOR_IMAGE))
        depth = self.get_cas().get(CASViews.DEPTH_IMAGE)
        if len(self.get_cas().annotations) > 0:
            rot_angle = self.get_cas().annotations[-1]
        else:
            return py_trees.Status.SUCCESS
        # self.get_cas().filter_annotations_by_type(robokudo.types.scene.ObjectHypothesis)

        if isinstance(self.get_cas().annotations[-1], float):
            if rot_angle > -4 and rot_angle < 4:


                object_hypothesis = self.get_cas().filter_annotations_by_type(robokudo.types.scene.ObjectHypothesis)

                for hypothesis in object_hypothesis:
                    assert isinstance(hypothesis, robokudo.types.scene.ObjectHypothesis)
                    class_name = hypothesis.classification.classname

                    # if class_name == 'Fork':
                    if class_name == 'Crackerbox':
                        roi = hypothesis.roi.roi
                        roi_depth = depth[roi.pos.y : (roi.pos.y + roi.height) , roi.pos.x : (roi.pos.x + roi.width)]
                        # roi_depth = depth[roi.height:roi.pos.y , roi.pos.x: roi.width]
                    # average pixel value
                        average_depth_value = np.mean(roi_depth)
                        # compare to white pixels (scale 0 to 1)
                        normalized = average_depth_value / 65535

                        # Define the bounding box for the gripper area
                        gripper_bbox = (0, 323, 33, 423)  # (x, y, x1, y1)
                        x, y, x1, y1 = gripper_bbox

                        gripper_roi = depth[y:y1, x:x1]

                        average_gripper_value = np.mean(gripper_roi)
                        # Compare to white pixels (for grayscale image assumption
                        gripper_normalized = average_gripper_value / 65535

                        if gripper_normalized >= normalized:
                            normalized_distance = gripper_normalized - normalized

                            # Define text to be written on the image
                            box_text = f"Box to gripper distance: {normalized_distance:.2f}"
                            # Position for the text
                            position_box = (10, 30)
                            # Font settings
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            font_color = (255, 255, 255)
                            line_type = 2
                            # Write text on the image
                            depth_distance = cv2.putText(color, box_text, position_box, font, font_scale, font_color, line_type)
                            # self.get_annotator_output_struct().set_image(depth_distance)
                        else:
                            # Define text to be written on the image
                            box_text = f"The object is between the grippers"
                            # Position for the text
                            position_box = (10, 30)
                            # Font settings
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            font_color = (255, 255, 255)
                            line_type = 2
                            # Write text on the image
                            depth_distance = cv2.putText(color, box_text, position_box, font, font_scale, font_color, line_type)


                        # visualize it in the robokudi gui
                        self.get_annotator_output_struct().set_image(depth_distance)


        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.Status.SUCCESS
