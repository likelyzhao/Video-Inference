import time
from utils import Composite_Video
from featureExtract import FeatureExtraction
from featureCoding import FeatureCoding
from postProcessing import PostProcessing
from config import config
from video import Video

class VideoInfer(object):
	def __init__(self, args):

		feature_extract = FeatureExtraction(modelPrototxt=config.FEATURE_EXTRACTION.MODEL_PROTOTXT,
											modelFile=config.FEATURE_EXTRACTION.MODEL_FILE,
											featureLayer=config.FEATURE_EXTRACTION.FEATURE_LAYER, gpu_id=args.gpu_id)
		feature_coding = FeatureCoding(featureDim=config.FEATURE_CODING.FEATURE_DIM,
		                               batchsize=args.frame_group,
		                               modelPrefix=config.FEATURE_CODING.MODEL_PREFIX,
										modelEpoch=config.FEATURE_CODING.MODEL_EPOCH,
										synset=config.FEATURE_CODING.SYNSET, gpu_id=args.gpu_id)
		post_processing = PostProcessing(score_thresh=args.display_score_thresh)

		self.feature_extraction = feature_extract
		self.feature_coding = feature_coding
		self.post_processing = post_processing
		self.step = args.step
		self.skip = args.skip
		self.frame_group = args.frame_group

	def infer(self, video_path):
		video = Video(video_path, step=self.step, skip=self.skip, frame_group_len=self.frame_group)
		video_timestamps = []
		video_classification_result = []
		t1 = time.time()
		for batch_timestamps, _, batch_classification_result in self.feature_coding(self.feature_extraction, video, topN=1):
			t2 = time.time()
			print "time cost: %f, results in timeduration:(%f~%f)s\n" % (t2 - t1, batch_timestamps[0], batch_timestamps[-1])
			for label, prob in batch_classification_result.items():
				print "{0}:{1:0.2f}%".format(label, prob * 100)
			print "--" * 10
                        import copy
			video_timestamps.append(copy.deepcopy(batch_timestamps))
			video_classification_result.append(batch_classification_result)
			t1 = time.time()

		label_durations, video_labels, label_probs = self.post_processing(video_timestamps, video_classification_result)
		return label_durations, video_labels, label_probs

	def composite_video(self, video, composite_video_name, display_score_thresh):
		newvideo = Composite_Video(videoname=composite_video_name, fps=1. / video.step, framesize=video._size)
		t1 = time.time()
		for batch_timestamps, batch_frames, batch_classification_result in self.feature_coding(self.feature_extraction, video, topN=5):
			t2 = time.time()
			print "time cost: %f, results in timeduration:(%f~%f)s\n" % (t2 - t1, batch_timestamps[0], batch_timestamps[-1])
			texts = []
			for label, prob in batch_classification_result.items():
				text = "{0}:{1:0.2f}%".format(label, prob * 100)
				if prob > display_score_thresh:
					texts.append(text)
				print text
			print "--" * 10
			t1 = time.time()
			newvideo._add_frame(batch_frames, texts)

		newvideo._composite_video()

		return composite_video_name
