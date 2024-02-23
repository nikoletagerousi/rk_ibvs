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
from std_msgs.msg import Float64



class KeepInCenter(robokudo.annotators.core.BaseAnnotator):
    """Publishes normalized vectors for moving the gripper on the x and y axes"""

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
                self.slice_x = slice(70, 400)
                self.slice_y = slice(40, 600)

                self.classname = None

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="KeepInCenter", descriptor=Descriptor()):
        """
        Default construction. Minimal one-time init!
        """
        super().__init__(name, descriptor)
        # self.pub = rospy.Publisher('hand_cam/movement', Vector3Stamped)
        # self.pub = rospy.Publisher('offset', Float64)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

        # defining these thresholds to avoid oscillation around the centralization of the camera to object
        self.small_thres = 10
        self.big_thres = 20
        self.threshold = self.small_thres

    def update(self):
        start_timer = default_timer()

        # Read color image from the cas
        color = deepcopy(self.get_cas().get(CASViews.COLOR_IMAGE))

        self.vector = np.zeros((3, 1))

        offset_x = 0
        offset_y = 0



        object_hypothesis_list = self.get_cas().filter_annotations_by_type(robokudo.types.scene.ObjectHypothesis)
        for hypothesis in object_hypothesis_list:
            assert isinstance(hypothesis, robokudo.types.scene.ObjectHypothesis)
            class_name = hypothesis.classification.classname

            movement = robokudo.types.annotation.Movement()
            movement.vector = self.vector
            movement.threshold = self.threshold
            movement.offset = offset_x

            image_center_x = color.shape[1] / 2
            image_center_y = color.shape[0] / 2

            # if class_name == 'Crackerbox':
            if class_name == self.descriptor.parameters.classname:
                roi = hypothesis.roi.roi
                box_width = hypothesis.roi.roi.width
                box_height = hypothesis.roi.roi.height
                box_x1 = hypothesis.roi.roi.pos.x
                box_y1 = hypothesis.roi.roi.pos.y

                box_center_x = (box_x1 + box_x1 + box_width) / 2
                box_center_y = (box_y1 + box_y1 + box_height) / 2

                #image_center_x = color.shape[1] / 2
                #image_center_y = color.shape[0] / 2

                offset_x = image_center_x - box_center_x
                offset_y = image_center_y - box_center_y

                movement.offset = abs(offset_x)

                # create the normalized vector
                self.vector[0] = - offset_x / math.sqrt((offset_x * offset_x) + (offset_y * offset_y))
                self.vector[1] = - offset_y / math.sqrt((offset_x * offset_x) + (offset_y * offset_y))
                self.vector[2] = 0
                movement.vector = self.vector


            # # visualize it in the robokudi gui
            start = (image_center_x, image_center_y)
            # end = (start[0] + self.vector[0], start[0] + self.vector[0])
            end = (start[0] + offset_x, start[0] + offset_y)

            vector = cv2.arrowedLine(color, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (0,255,0), 2)

            self.get_annotator_output_struct().set_image(vector)





        # # if class_name == 'Crackerbox':
        if class_name == self.descriptor.parameters.classname:
            if abs(offset_x) <= self.threshold:
                self.threshold = self.big_thres
            else:
                self.threshold = self.small_thres
            movement.threshold = self.threshold

            # message = Float64()
            # message = offset_x
            # self.pub.publish(message)
            self.get_cas().annotations.append(movement)

        #     if offset_x > self.threshold:
        #         # move close to the target
        #         message = Vector3Stamped()
        #         message.vector.x = self.vector[0]
        #         message.vector.y = self.vector[1]
        #         message.vector.z = self.vector[2]
        #
        #     else:
        #         message = Vector3Stamped()
        #         message.vector.x = 0
        #         message.vector.y = 0
        #         message.vector.z = 0
        #
        # else:
        #     message = Vector3Stamped()
        #     message.vector.x = 0
        #     message.vector.y = 0
        #     message.vector.z = 0
        # self.pub.publish(message)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.Status.SUCCESS
