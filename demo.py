import argparse
from videoInfer import VideoInfer

def parse_args():
	parser = argparse.ArgumentParser(description='Video Inference Demo')
	parser.add_argument('--video_path', help='video to be classified', default='test.avi', type=str)
	parser.add_argument('--step', help='Iterate frames every `step` seconds. Defaults to iterating every frame.', default=None, type=float)
	parser.add_argument('--frame_group', help='number of frames to be grouped as one classification input', default=1, type=int)
	parser.add_argument('--gpu_id', help='which gpu to use', default=0, type=int)
	parser.add_argument('--composite_video', help='composite a new video with video inference result.', action='store_true')
	parser.add_argument('--composite_video_name', help='new video name', default='newvideo.mp4', type=str)
	parser.add_argument('--display_score_thresh', help='label prob higher than the thresh can be displayed', default=0.1, type=float)

	args = parser.parse_args()
	return args


if __name__ == '__main__':

	args = parse_args()
	print('Called with argument:', args)
	video_infer_handler = VideoInfer(args)

	if args.composite_video:
		demo_video = video_infer_handler.composite_video(args.video_path, args.composite_video_name, args.display_score_thresh)
		print demo_video + " is generated."
	else:
		video_labels, label_duration, label_prob = video_infer_handler.infer(args.video_path)
