import copy
from timeit import default_timer

import cv2
import numpy as np
import py_trees
import math

import rospy

import robokudo.types.scene
import robokudo.utils.cv_helper

from robokudo.cas import CASViews
from copy import deepcopy
from geometry_msgs.msg import Vector3Stamped



class KeepInCenter(robokudo.annotators.core.BaseAnnotator):
    """Crops and rotates the color image"""

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
                self.slice_x = slice(70, 400)
                self.slice_y = slice(40, 600)

                self.vector = ()

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="KeepInCenter", descriptor=Descriptor()):
        """
        Default construction. Minimal one-time init!
        """
        super().__init__(name, descriptor)
        self.pub = rospy.Publisher('Vector', Vector3Stamped)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self):
        start_timer = default_timer()

        # Read color image from the cas
        color = deepcopy(self.get_cas().get(CASViews.COLOR_IMAGE))

        self.vector = np.zeros((3, 1))

        object_hypothesis_list = self.get_cas().filter_annotations_by_type(robokudo.types.scene.ObjectHypothesis)
        for hypothesis in object_hypothesis_list:
            assert isinstance(hypothesis, robokudo.types.scene.ObjectHypothesis)
            class_name = hypothesis.classification.classname

            if class_name == 'Crackerbox':
                roi = hypothesis.roi.roi
                box_width = hypothesis.roi.roi.width
                box_height = hypothesis.roi.roi.height
                box_x1 = hypothesis.roi.roi.pos.x
                box_y1 = hypothesis.roi.roi.pos.y

                box_center_x = (box_x1 + box_x1 + box_width) / 2
                box_center_y = (box_y1 + box_y1 + box_height) / 2

                image_center_x = color.shape[1] / 2
                image_center_y = color.shape[0] / 2

                offset_x = image_center_x - box_center_x
                offset_y = image_center_y - box_center_y

                # create the normalized vector
                self.vector[0] = offset_x / math.sqrt((offset_x * offset_x) + (offset_y * offset_y))
                self.vector[1] = offset_y / math.sqrt((offset_x * offset_x) + (offset_y * offset_y))
                self.vector[2] = 0




        # # visualize it in the robokudi gui
        # Define text to be written on the image
        box_text = f"Offset x: {offset_x:.5f} Offset y: {offset_y:.5f}"
        # Position for the text
        position_box = (10, 30)
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        line_type = 2
        # Write text on the image
        text = cv2.putText(color, box_text, position_box, font, font_scale, font_color, line_type)
        self.get_annotator_output_struct().set_image(text)



        # visualize it in the robokudi gui
        self.get_annotator_output_struct().set_image(text)

        message = Vector3Stamped()
        message.vector.x = self.vector[0]
        message.vector.y = self.vector[1]
        message.vector.z = self.vector[2]
        self.pub.publish(message)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.Status.SUCCESS
