from timeit import default_timer

import cv2
import py_trees
import rospy

import robokudo.utils.cv_helper
from robokudo.cas import CASViews
import robokudo.types.scene
import robokudo.types.annotation

from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3Stamped


class HandleMessages(robokudo.annotators.core.BaseAnnotator):
    """Handles publishing the angle, movement and distance messages"""

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
                self.slice_x = slice(70, 400)
                self.slice_y = slice(40, 600)

                self.classname = None

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="HandleMessages", descriptor=Descriptor()):
        """
        Default construction. Minimal one-time init!
        """
        super().__init__(name, descriptor)
        self.pub_angle = rospy.Publisher('hand_cam/angle', Float64)
        self.pub_vector = rospy.Publisher('hand_cam/movement', Vector3Stamped)
        self.pub_distance = rospy.Publisher('hand_cam/distance', Float64)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self):
        start_timer = default_timer()

        object_hypothesis = self.get_cas().filter_annotations_by_type(robokudo.types.scene.ObjectHypothesis)

        angle_message = Float64()
        vector_message = Vector3Stamped()
        distance_message = Float64()

        for hypothesis in object_hypothesis:
            assert isinstance(hypothesis, robokudo.types.scene.ObjectHypothesis)
            class_name = hypothesis.classification.classname

            if class_name == self.descriptor.parameters.classname:

                angle = self.get_cas().filter_annotations_by_type(robokudo.types.annotation.Angle)

                movement = self.get_cas().filter_annotations_by_type(robokudo.types.annotation.Movement)
                vector = movement[0].vector
                offset = movement[0].offset
                threshold = movement[0].threshold

                dist = self.get_cas().filter_annotations_by_type(robokudo.types.annotation.Distance)

                if len(angle) > 0:
                    # provide with angle in radians
                    angle_message.data = angle[0].rot_angle
                    self.pub_angle.publish(angle_message)
                    rot_angle = angle[0].rot_angle
                else:
                    angle_message.data = 0
                    self.pub_angle.publish(angle_message)
                    rot_angle = -100

                if len(vector) > 0:
                    if offset > threshold:
                        # move close to the target
                        vector_message.vector.x = vector[0]
                        vector_message.vector.y = vector[1]
                        vector_message.vector.z = vector[2]
                        self.pub_vector.publish(vector_message)
                    else:
                        vector_message.vector.x = 0
                        vector_message.vector.y = 0
                        vector_message.vector.z = 0
                        self.pub_vector.publish(vector_message)

                else:
                    vector_message.vector.x = 0
                    vector_message.vector.y = 0
                    vector_message.vector.z = 0
                    self.pub_vector.publish(vector_message)

                # if 0.174532925 > rot_angle > -0.174532925:
                if 0.035 > rot_angle > -0.035:
                    # provide with distance in meters
                    if len(dist) > 0:
                        distance_message.data = dist[0].distance
                        self.pub_distance.publish(distance_message)
                    else:
                        distance_message.data = -100
                        self.pub_distance.publish(distance_message)
                else:
                    distance_message.data = -100
                    self.pub_distance.publish(distance_message)

            else:
                angle_message.data = 0

                vector_message.vector.x = 0
                vector_message.vector.y = 0
                vector_message.vector.z = 0

                distance_message.data = -100

                self.pub_angle.publish(angle_message)
                self.pub_vector.publish(vector_message)
                self.pub_distance.publish(distance_message)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.Status.SUCCESS
