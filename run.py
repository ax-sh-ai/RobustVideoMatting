import torch
from model import MattingNetwork
from pathlib import Path

from inference import convert_video

path = Path(__file__).parent
model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
output_path = path/'output'

batch_size = 62 # change if out of memory error
batch_size = 32 # change if out of memory error
batch_size = 16 # change if out of memory error
batch_size = 12 # change if out of memory error
batch_size = 10 # change if out of memory error best for 8gb free gpu

def process_vido(input_video_path):
    input_video_path  = Path(str(input_video_path.encode('utf-8').decode('utf-8')))
    print(input_video_path, str(input_video_path))
    if not  input_video_path.is_file():
        print('input video does not exist')
        return
    file_name = input_video_path.stem
    ext = input_video_path.suffix
    print('Processing', file_name, ext)
   
 
    input_source = str(input_video_path)
    output_file = output_path.joinpath(f'{file_name}-output.mp4')
    
    if(output_file.is_file()):
        print('previous file exists exiting')
        return
    
    
    comp = str(output_file)
    mask = comp+'-mask.mp4'
    print('Generating', comp, mask)
    
    # extracted = "output/fgr.mp4"

    try:
        convert_video(
            model,                           # The model, can be on any device (cpu or cuda).
            input_source=input_source,        # A video file or an image sequence directory.
            output_type='video',             # Choose "video" or "png_sequence"
            output_composition=comp,    # File path if video; directory path if png sequence.
            output_alpha=mask,          # [Optional] Output the raw alpha prediction.
            # output_foreground=extracted,     # [Optional] Output the raw foreground prediction.
            output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
            downsample_ratio=None,           # A hyperparameter to adjust or use None for auto.
            seq_chunk=batch_size,                    # Process n frames at once for better parallelism.
        )
    except RuntimeError as e:
        print('\n\n')
        print('CUDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',e.args[0], "<<<<")

# print(torch.cuda.get_device_properties(0).total_memory,"<<<<<<<<")
def main():
    # output_path.iterdir()
    input_video_path  = r'foo.webm'
    process_vido(input_video_path)
    

main()
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
