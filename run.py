import torch
from model import MattingNetwork

model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))

from inference import convert_video

convert_video(
    model,                           # The model, can be on any device (cpu or cuda).
    input_source='video.webm',        # A video file or an image sequence directory.
    output_type='video',             # Choose "video" or "png_sequence"
    output_composition='output/com.mp4',    # File path if video; directory path if png sequence.
    output_alpha="output/mask.mp4",          # [Optional] Output the raw alpha prediction.
    output_foreground="output/fgr.mp4",     # [Optional] Output the raw foreground prediction.
    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
    downsample_ratio=None,           # A hyperparameter to adjust or use None for auto.
    seq_chunk=12,                    # Process n frames at once for better parallelism.
)

# # python inference.py \
# #     --variant mobilenetv3 \
# #     --checkpoint "CHECKPOINT" \
# #     --device cuda \
# #     --input-source "input.mp4" \
# #     --output-type video \
# #     --output-composition "composition.mp4" \
# #     --output-alpha "alpha.mp4" \
# #     --output-foreground "foreground.mp4" \
# #     --output-video-mbps 4 \
# #     --seq-chunk 1

# import torch

# model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").cuda() # or "resnet50"
# # convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")

# # convert_video(
# #     model,                           # The loaded model, can be on any device (cpu or cuda).
# #     input_source='video.mp4',        # A video file or an image sequence directory.
# #     # downsample_ratio=None,           # [Optional] If None, make downsampled max size be 512px.
# #     output_type='video',             # Choose "video" or "png_sequence"
# #     # output_composition='com.mp4',    # File path if video; directory path if png sequence.
# #     output_alpha="pha.mp4",          # [Optional] Output the raw alpha prediction.
# #     # output_foreground="fgr.mp4",     # [Optional] Output the raw foreground prediction.
# #     output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
# #     seq_chunk=12,                    # Process n frames at once for better parallelism.
# #     num_workers=1,                   # Only for image sequence input. Reader threads.
# #     progress=True                    # Print conversion progress.
# # )


# from inference import convert_video

# # import torch
# # from model import MattingNetwork

# # model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
# # model.load_state_dict(torch.load('rvm_mobilenetv3.pth')

# convert_video(
#     model,                           # The loaded model, can be on any device (cpu or cuda).
#     input_source='video.mp4',        # A video file or an image sequence directory.
#     # downsample_ratio=None,           # [Optional] If None, make downsampled max size be 512px.
#     output_type='video',             # Choose "video" or "png_sequence"
#     # output_composition='com.mp4',    # File path if video; directory path if png sequence.
#     output_alpha="pha.mp4",          # [Optional] Output the raw alpha prediction.
#     # output_foreground="fgr.mp4",     # [Optional] Output the raw foreground prediction.
#     output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
#     seq_chunk=12,                    # Process n frames at once for better parallelism.
#     num_workers=1,                   # Only for image sequence input. Reader threads.
#     progress=True                    # Print conversion progress.
# )
