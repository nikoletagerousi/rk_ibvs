import copy
from timeit import default_timer

import cv2
import rospy
import py_trees
import robokudo.annotators
import robokudo.types.scene
import robokudo.utils.cv_helper
from robokudo.cas import CASViews
from copy import deepcopy
import numpy as np
from std_msgs.msg import Float64


class Distance(robokudo.annotators.core.BaseAnnotator):
    """Crops and rotates the color image"""

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
                self.slice_x = slice(70, 400)
                self.slice_y = slice(40, 600)

                self.classname = None
                self.real_width = None
                self.real_height = None

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="Distance", descriptor=Descriptor()):
        """
        Default construction. Minimal one-time init!
        """
        super().__init__(name, descriptor)
        self.pub = rospy.Publisher('hand_cam/distance', Float64)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

        self.list_length = 5
        self.width = [None] * 5
        self.height = [None] * 5

        self.camera_position = [None] * 5
        self.counter = 0

        self.A = []
        self.b = []
        self.x = []

    def update(self):
        start_timer = default_timer()
        real_depth = -100

        # Read color image from the cas
        color = deepcopy(self.get_cas().get(CASViews.COLOR_IMAGE))

        relative_camera_position = self.get_cas().get(robokudo.cas.CASViews.VIEWPOINT_CAM_TO_WORLD)

        # Use separate focal lengths for x and y
        focal_length_x = 226.4069213087372  # focal length in pixels-fx
        focal_length_y = 226.92131144188522  # focal length in pixels-fx

        # Real dimensions of the object
        real_width = self.descriptor.parameters.real_width
        # real_width = 60  # width of the object (mm)
        real_height = self.descriptor.parameters.real_height
        # real_height = 210  # height of the object (mm)

        object_hypothesis_list = self.get_cas().filter_annotations_by_type(robokudo.types.scene.ObjectHypothesis)
        for hypothesis in object_hypothesis_list:
            assert isinstance(hypothesis, robokudo.types.scene.ObjectHypothesis)
            class_name = hypothesis.classification.classname
            cv2.undistort

            # if class_name == 'Crackerbox':
            if class_name == self.descriptor.parameters.classname:
                roi = hypothesis.roi.roi
                box_width = hypothesis.roi.roi.width
                box_height = hypothesis.roi.roi.height

                camera_position = relative_camera_position.translation[2]

                # read bb dim
                # write to list
                self.width[self.counter % self.list_length] = box_width
                self.height[self.counter % self.list_length] = box_height

                self.camera_position[self.counter % self.list_length] = camera_position

                # if counter > list_length
                # calculate depth
                if self.counter > self.list_length:
                    self.A = np.zeros((10, 1))
                    self.b = np.zeros((10, 1))
                    for i in range(self.list_length):
                        self.A[2*i, 0] = self.width[i]
                        self.A[2*i+1, 0] = self.height[i]

                        # Assign values from lists a and b to specific positions in array A
                        self.b[2*i, 0] = (self.width[i] * ((self.camera_position[i] - self.camera_position[4])) + (focal_length_x * real_width))
                        self.b[2*i+1, 0] = (self.height[i] * ((self.camera_position[i] - self.camera_position[4])) + (focal_length_y * real_height))

                        # solve least squares
                    depth, residuals, rank, s = np.linalg.lstsq(self.A, self.b, rcond=None)

                    real_depth = depth[0][0]

                    # turning the distance from mm to m
                    real_depth = real_depth * 0.001

                self.counter += 1

        # # visualize it in the robokudi gui
        # Define text to be written on the image
        box_text = f"Box to gripper distance: {real_depth:.5f}"
        # Position for the text
        position_box = (10, 30)
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        line_type = 2
        # Write text on the image
        text = cv2.putText(color, box_text, position_box, font, font_scale, font_color, line_type)

        # get the rotation angle from the CAS
        if len(self.get_cas().annotations) > 0:
            rot_angle = self.get_cas().annotations[-1]
        else:
            return py_trees.Status.SUCCESS

        # if class_name == 'Crackerbox':
        if class_name == self.descriptor.parameters.classname:
            if isinstance(self.get_cas().annotations[-1], float):
                if rot_angle < 2 and rot_angle > -2:
                    message = Float64()
                    message.data = real_depth
                else:
                    message = Float64()
                    message.data = -100
            else:
                message = Float64()
                message.data = -100
        else:
            message = Float64()
            message.data = -100
        self.pub.publish(message)

        self.get_annotator_output_struct().set_image(text)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.Status.SUCCESS
