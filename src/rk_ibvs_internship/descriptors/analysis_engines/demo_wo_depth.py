import robokudo.analysis_engine

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo_depth_estimation.annotators.monodepth import Monodepth
from robokudo_yolo.annotators.YoloAnnotator import YoloAnnotator
from robokudo.annotators.crop_and_rotate import CropAndRotate
from rk_ibvs_internship.annotators.fix_rotation import FixRotation
from rk_ibvs_internship.annotators.depth_distance import DepthDistance
from rk_ibvs_internship.annotators.pinhole_distance import PinholeDistance
from rk_ibvs_internship.annotators.track_object import TrackObject
from rk_ibvs_internship.annotators.sam_track_object import SAMTrackObject
from rk_ibvs_internship.annotators.distance import Distance
from rk_ibvs_internship.annotators.keep_in_center import KeepInCenter


import robokudo.descriptors.camera_configs.config_hsr_wo_depth

import robokudo.io.camera_interface
import robokudo.idioms


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    def name(self):
        return "demo_wo_depth"

    def implementation(self):
        """
        Create a basic pipeline that does tabletop segmentation
        """
        camera_config = robokudo.descriptors.camera_configs.config_hsr_wo_depth.CameraConfig()
        reader_config = CollectionReaderAnnotator.Descriptor(
            camera_config=camera_config,
            camera_interface=robokudo.io.camera_interface.ROSCameraWoDepthInterface(camera_config))

        yolo_descriptor = YoloAnnotator.Descriptor()
        yolo_descriptor.parameters.ros_pkg_path = "robokudo"
        yolo_descriptor.parameters.weights_path = "weights/ycb_weights.pt"
        yolo_descriptor.parameters.threshold = 0.5
        yolo_descriptor.parameters.precision_mode = True
        yolo_descriptor.parameters.id2name_json_path = "weights/id2name.json"

        # SAMTrackObject.Descriptor().parameters.classname = "Crackerbox"
        #
        # FixRotation.Descriptor().parameters.classname = SAMTrackObject.Descriptor.parameters.classname
        # Distance.Descriptor().parameters.classname = SAMTrackObject.Descriptor.parameters.classname
        # KeepInCenter.Descriptor().parameters.classname = SAMTrackObject.Descriptor.parameters.classname
        # DepthDistance.Descriptor().parameters.classname = SAMTrackObject.Descriptor.parameters.classname
        # PinholeDistance.Descriptor().parameters.classname = SAMTrackObject.Descriptor.parameters.classname



        seq = robokudo.pipeline.Pipeline("RWPipeline")
        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),
                CollectionReaderAnnotator(descriptor=reader_config),
                CropAndRotate(),
                Monodepth(),
                YoloAnnotator(descriptor=yolo_descriptor),
                SAMTrackObject(),
                # TrackObject(),
                FixRotation(),
                Distance(),
                KeepInCenter(),
                DepthDistance(),
                PinholeDistance()


            ])
        return seq
