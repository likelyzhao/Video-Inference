import numpy as np
import caffe
from video import Video
from utils import center_crop_images
import skimage





class FeatureExtraction(object):
    """
    extract features for video frames.
    
    Parameters
    -----------------
    video: Video
    modelPrototxt: models architecture file
    modelFile: models snapshot
    featureLayer: which layer to be extracted as feature
    gpu_id: which gpu to use
    """

    def __init__(self, modelPrototxt, modelFile, featureLayer, gpu_id=0):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)

        self.net = caffe.Net(modelPrototxt, modelFile, caffe.TEST)
        data_shape = self.net.blobs['data'].data.shape
        self.batchsize = data_shape[0]
        self.height = data_shape[2]
        self.width = data_shape[3]

        self.featureLayer = featureLayer
        featureDim = self.net.blobs[featureLayer].data.shape
        print "featureDim:", featureDim

        transformer = caffe.io.Transformer({'data': data_shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))  # mean pixel
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))
        self.transformer = transformer

    def ext_process(self,img_list):
        import cv2
        im_group = np.empty((self.batchsize, 3, self.height, self.width), dtype=np.float32)
        for ix, img in enumerate(img_list):
            img = cv2.resize(img, (225, 225))
            img = img.astype(np.float32, copy=True)
            img -= np.array([[[103.94, 116.78, 123.68]]])
            img = img * 0.017
            img = img.transpose((2, 0, 1))
            im_group[ix] = img

        self.net.blobs['data'].data[...] = im_group
        out = self.net.forward()
        feature = np.squeeze(out[self.featureLayer])
        print np.shape(feature)
        return feature

    def __call__(self, video):
        if video.frame_group_len != self.batchsize:
            raise IOError(
                    ("FeatureExtraction error: video frame group len (%d) is not equal to prototxt batchsize (%d)"
                     % (self.video.frame_group_len, self.batchsize)))
            """
            warnings.warn(
                "FeatureExtractionWarning: "
                "video frame group len (%d) " % (self.video.frame_group_len) +
                "is not equal to prototxt batchsize (%d)" % (data_shape[0]) +
                "Change prototxt batchsize to video frame group len.",
                UserWarning)
            data_shape[0] = self.video.frame_group_len
            """
        for timestamps, frames in video: # frames are rgb channel-ordered
            features = self.ext_process(frames)

            yield timestamps, frames, features


if __name__ == "__main__":
    from config import config

    filename = "test.avi"
    video = Video(filename, frame_group_len=1)
    features = FeatureExtraction(modelPrototxt=config.FEATURE_EXTRACTION.MODEL_PROTOTXT,
	                            modelFile=config.FEATURE_EXTRACTION.MODEL_FILE,
	                            featureLayer=config.FEATURE_EXTRACTION.FEATURE_LAYER, gpu_id=0)
    for timestamps, frames, fea in features(video):
        print fea.shape
        print fea
        break
