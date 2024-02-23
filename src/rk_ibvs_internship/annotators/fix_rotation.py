import copy
from timeit import default_timer

import cv2
import py_trees
import rospy
import std_msgs.msg

import robokudo.utils.cv_helper
from robokudo.cas import CASViews
import robokudo.types.scene
import numpy as np
from math import atan2, cos, sin, sqrt, pi
from scipy import ndimage
from std_msgs.msg import Float64

class FixRotation(robokudo.annotators.core.BaseAnnotator):
    """Gives the gripper angle rotation in radians"""

    def rotate_image(self, source, angle):
        (h, w) = source.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated = cv2.warpAffine(source, M, (w, h))
        return rotated

    def drawAxis(self, img, p_, q_, color, scale):
        p = list(p_)
        q = list(q_)
        angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
        hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
        # lengthen the arrow by a factor of scale
        q[0] = p[0] - scale * hypotenuse * cos(angle)
        q[1] = p[1] - scale * hypotenuse * sin(angle)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
        # create the arrow hooks
        p[0] = q[0] + 9 * cos(angle + pi / 4)
        p[1] = q[1] + 9 * sin(angle + pi / 4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
        p[0] = q[0] + 9 * cos(angle - pi / 4)
        p[1] = q[1] + 9 * sin(angle - pi / 4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    def getOrientation(self, pts, img):
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i, 0] = pts[i, 0, 0]
            data_pts[i, 1] = pts[i, 0, 1]

        # pca analysis
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, np.empty((0)))


        # store the center of the object
        cntr = (int(mean[0, 0]), int(mean[0, 1]))

        cv2.circle(img, cntr, 3, (255, 0, 255), 2)
        p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
              cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
        p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
              cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
        self.drawAxis(img, cntr, p1, (255, 255, 0), 1)  # this one is the blue axis
        self.drawAxis(img, cntr, p2, (0, 0, 255), 5)  # red axis

        angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians. Angle between the first eigenvector and the positive x axis
        return angle, eigenvectors
    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
                self.slice_x = slice(70, 400)
                self.slice_y = slice(40, 600)

                self.classname = None

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="FixRotation", descriptor=Descriptor()):
        """
        Default construction. Minimal one-time init!
        """
        super().__init__(name, descriptor)
        # self.pub = rospy.Publisher('hand_cam/angle', Float64)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self):
        start_timer = default_timer()

        # Read color image from the cas
        color = self.get_cas().get(CASViews.COLOR_IMAGE)

        # If the class of my object exists the SAM mask should be extracted
        object_hypothesis = self.get_cas().filter_annotations_by_type(robokudo.types.scene.ObjectHypothesis)


        for hypothesis in object_hypothesis:
            assert isinstance(hypothesis, robokudo.types.scene.ObjectHypothesis)
            class_name = hypothesis.classification.classname

            rot_angle = 0

            rotation = robokudo.types.annotation.Angle()
            rotation.rot_angle = - rot_angle


            # if class_name == 'Crackerbox':
            if class_name == self.descriptor.parameters.classname:
                # Perform some action if the specific class is detected
                mask = hypothesis.roi.mask
                _, thresh = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                if len(contours) > 1:
                    largest_contour = max(contours, key=cv2.contourArea)
                    angle, eigenvectors = self.getOrientation(largest_contour, mask)
                else:
                    angle, eigenvectors = self.getOrientation(contours[0], mask)

                # this should be used in cae we want the angle to be calculated in degrees
                # angle_degrees = np.rad2deg(angle)
                # angle1 = abs(angle_degrees)
                # angle2 = 180 - angle1
                # new_angle = min(angle1, angle2)

                angle1 = angle
                angle2 = 3.14159265 - angle1
                new_angle = min(angle1, angle2)

                if (eigenvectors[0, 0] > 0 and eigenvectors[0, 1] > 0) or (
                        eigenvectors[0, 0] < 0 and eigenvectors[0, 1] < 0):
                    # rot_angle = 90 - new_angle
                    rot_angle = 1.57079633 - new_angle
                else:
                    # rot_angle = new_angle - 90
                    rot_angle = new_angle - 1.57079633

                fixed_rotated = ndimage.rotate(color, -rot_angle, reshape=True)




                # visualize it in the robokudi gui
                self.get_annotator_output_struct().set_image(fixed_rotated)

                rotation.rot_angle = - rot_angle

                # the desired angle for rotation is being written on the CAS
                self.get_cas().annotations.append(rotation)
                self.get_cas().annotations.append(rot_angle)
                # self.get_cas().set(CASViews.COLOR_IMAGE, fixed_rotated)


            # # if class_name == 'Crackerbox':
            # if class_name == self.descriptor.parameters.classname:
            #
            #     message = Float64()
            #     message.data = - rot_angle
            #     self.pub.publish(message)
            # else:
            #     message = Float64()
            #     message.data = 0
            #     self.pub.publish(message)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.Status.SUCCESS
