from easydict import EasyDict as edict

config = edict()

config.FEATURE_EXTRACTION = edict()
config.FEATURE_EXTRACTION.MODEL_PROTOTXT = 'models/extract_feature.prototxt'
config.FEATURE_EXTRACTION.MODEL_FILE = 'models/weight.caffemodel'
config.FEATURE_EXTRACTION.FEATURE_LAYER = 'pool5'

config.FEATURE_CODING = edict()
config.FEATURE_CODING.MODEL_PREFIX = 'models/netvlad'
config.FEATURE_CODING.MODEL_EPOCH = 50
config.FEATURE_CODING.FEATURE_DIM = 2048
config.FEATURE_CODING.SYNSET='lsvc_class_index.txt'