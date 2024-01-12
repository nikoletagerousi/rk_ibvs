import copy
from timeit import default_timer

import cv2
import py_trees
import numpy as np
import rospy
import math

import robokudo.utils.cv_helper
from robokudo.cas import CASViews
from std_msgs.msg import Float64


class PinholeDistance(robokudo.annotators.core.BaseAnnotator):
    """Crops and rotates the color image"""


    def calculate_distance(self, focal_length_x, focal_length_y, real_width, real_height, x_dimension_in_image, y_dimension_in_image):
    # Calculate distances in x and y dimensions
        distance_x = (focal_length_x * real_width) / x_dimension_in_image
        distance_y = (focal_length_y * real_height) / y_dimension_in_image

        # Calculate overall distance using Euclidean distance
        distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

        return distance

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
                self.slice_x = slice(70, 400)
                self.slice_y = slice(40, 600)

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="PinholeDistance", descriptor=Descriptor()):
        """
        Default construction. Minimal one-time init!
        """
        super().__init__(name, descriptor)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

        # #i am publishing this topic to be able to see what is usually the difference between the gripper and the area right next to it
        # self.pub = rospy.Publisher('normalized_gripper', Float64)


    def update(self):
        start_timer = default_timer()

        # Read color image from the cas
        color = self.get_cas().get(CASViews.COLOR_IMAGE) #takes the updated image from CAS which is already rotated
        depth = self.get_cas().get(CASViews.DEPTH_IMAGE)


        # Use separate focal lengths for x and y
        focal_length_x = 226.4069213087372  # focal length in pixels-fx
        focal_length_y = 226.92131144188522 # focal length in pixels-fx

        # Real dimensions of the object
        real_width = 60 # width of the object (mm)
        real_height = 210  # height of the object (mm)

        if len(self.get_cas().annotations) > 0:
            rot_angle = self.get_cas().annotations[-1]
        else:
            return py_trees.Status.SUCCESS


        if isinstance(self.get_cas().annotations[-1], float):
            if rot_angle > -7 and rot_angle < 4:
            # if rot_angle > -4 and rot_angle < 4:

                object_hypothesis = self.get_cas().filter_annotations_by_type(robokudo.types.scene.ObjectHypothesis)

                for hypothesis in object_hypothesis:
                    assert isinstance(hypothesis, robokudo.types.scene.ObjectHypothesis)
                    class_name = hypothesis.classification.classname

                    if class_name == 'Crackerbox':
                    # if class_name == 'Fork':
                        roi = hypothesis.roi.roi

                        # checking if the gripper has already grasped the object
                        gripper_bbox = (0, 323, 33, 423)  # (x, y, x1, y1)
                        x, y, x1, y1 = gripper_bbox
                        gripper_roi = depth[y:y1, x:x1]
                        average_gripper_value = np.mean(gripper_roi)
                        # Compare to white pixels - the max vaue on giskard is 65535 and not 255
                        gripper_normalized = average_gripper_value / 65535

                        gripper_limits = (34, 323, 53, 423)  # (x, y, x1, y1)
                        x, y, x1, y1 = gripper_limits
                        gripper_limits_roi = depth[y:y1, x:x1]
                        average_gripper_limits_value = np.mean(gripper_limits_roi)
                        gripper_limits_normalized = average_gripper_limits_value / 65535

                        # find their difference and if it is bigger than a threshold then the rest of the code should work
                        normalized = gripper_normalized - gripper_limits_normalized

                        if normalized > 0.05:
                            x_dimension_in_image = roi.width
                            y_dimension_in_image = roi.height

                            # Calculate distances using both x and y dimensions
                            distance = self.calculate_distance(focal_length_x, focal_length_y, real_width, real_height, x_dimension_in_image, y_dimension_in_image)


                        # text on screen settings
                            # Define text to be written on the image
                            box_text = f"Box to gripper distance: {distance:.2f} mm"
                            # Position for the text
                            position_box = (10, 30)
                            # Font settings
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            font_color = (255, 255, 255)
                            line_type = 2

                            # Put the text on the original color image
                            pinhole_distance = cv2.putText(color, box_text, position_box, font, font_scale, font_color, line_type)

                            # visualize it in the robokudo gui
                            self.get_annotator_output_struct().set_image(pinhole_distance)

                            # message = Float64()
                            # message.data = normalized
                            # self.pub.publish(message)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.Status.SUCCESS
