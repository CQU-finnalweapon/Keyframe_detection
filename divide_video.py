import os
import sys
import time
import cv2
import pathlib
import json
import click
import requests
from tqdm import tqdm
from dashscope import MultiModalConversation

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, ROOT_DIR)

from task.robomimic import get_task_config
from config import DASHSCOPE_API_KEY



def video_understanding(video_path, prompt, model, fps):
    if isinstance(video_path, str):
        # print("loading a video:", video_path)
        # 处理视频时使用fps参数，表示每隔1/fps 秒抽取一帧
        content = [{'video': video_path, "fps": fps}, {'text': prompt}]
    else:
        # print("loading a list of images")
        # 处理图像列表时不使用fps参数，让API分析所有图像，to 避免二次降维
        content = [{'video': video_path, "fps": 1}, {'text': prompt}]

    models1 = ['qwen2.5-vl-72b-instruct', 'qwen2.5-vl-32b-instruct', 'qwen3-vl-235b-a22b-instruct',
                 'qwen-vl-plus', 'qwen-vl-plus-latest', 'qwen-vl-max', 'qwen-vl-max-latest',
                 ]
    models2 = ['qvq-plus', 'qvq-plus-latest', 'qvq-max', 'qvq-max-latest', 
               'qwen3-vl-plus','qwen3-vl-235b-a22b-thinking',]

    if model in models1:
        messages = [
            {
                'role': 'system', 
                # 'content': [{'text': 'You are a helpful assistant.'}]
                'content': [{'text': 'You are a robot engineering helping divide a demo into subtask pieces.'}]
            },
            {
                'role':'user',
                'content': content
            }
        ]
        stream=False
    elif model in models2:
        messages = [
            {
                'role': 'user',
                'content': content
            }
        ]
        stream=True
    else:
        raise NotImplementedError(f"Model {model} not implemented.")

    # 添加重试机制
    max_retries = 3
    retry_delay = 5  # 秒
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} of {max_retries} to call the API...")
            response = MultiModalConversation.call(
                # API key configured in config.py
                api_key=DASHSCOPE_API_KEY,
                model=model,
                # model='qwen2.5-vl-72b-instruct',
                # model='qwen2.5-vl-32b-instruct',
                messages=messages,
                stream=stream,
                )
            # 如果调用成功，尝试解析响应
            if model in models1:
                json_output = response["output"]["choices"][0]["message"].content[0]["text"] # type: ignore
            elif model in models2:
                json_output = ""
                # 判断是否结束思考过程并开始回复
                is_answering = False
                for chunk in response:
                    # 如果思考过程与回复皆为空，则忽略
                    message = chunk.output.choices[0].message
                    reasoning_content_chunk = message.get("reasoning_content", None)

                    if (chunk.output.choices[0].message.content == [] and
                        reasoning_content_chunk == ""):
                        pass
                    else:
                        # 如果当前为思考过程
                        # if reasoning_content_chunk != None and chunk.output.choices[0].message.content == []:
                        #     print(chunk.output.choices[0].message.reasoning_content, end="")
                        #     reasoning_content += chunk.output.choices[0].message.reasoning_content
                        # 如果当前为回复
                        if chunk.output.choices[0].message.content != []:
                            if not is_answering:
                                print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                                is_answering = True
                            print(chunk.output.choices[0].message.content[0]["text"], end="")
                            json_output += chunk.output.choices[0].message.content[0]["text"]
            else:
                raise NotImplementedError(f"Model {model} not implemented.")
            return json_output
            
        except requests.exceptions.SSLError as e:
            print(f"SSL错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                print("所有重试都失败了, 返回None")
                return None
                
        except Exception as e:
            print(f"其他错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("所有重试都失败了, 返回None")
                return None
    
    return None




def save_video_frames(video_path, output_folder, target_fps=None, frame_exist=False, scale_factor=2):
    """
    Save frames from a video file as images with a specific FPS.
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Folder to save the extracted frames
        target_fps (float): Desired frames per second (if None, saves all frames)
    """
    output_folder += f"_fps_{target_fps}_scale_{scale_factor}" if target_fps is not None else f"_fps_all_scale_{scale_factor}"
    output_folder = os.path.expanduser(output_folder)
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    else:
        if frame_exist:
            image_path_list = [f"file://{os.path.join(output_folder, f)}" for f in sorted(os.listdir(output_folder)) if f.endswith('.jpg')]
            if len(image_path_list) > 0:
                print(f"Frame images already exist in {output_folder}, skipping video to image step.")
                return image_path_list
        # delete all files in the folder
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video file {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS) ## input_video 的原始帧率，即每秒帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    
    print(f"Original FPS: {original_fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    
    # If target_fps is None, save all frames
    if target_fps is None: ## output 的目标帧率
        target_fps = original_fps
    
    # Calculate frame interval based on target FPS
    frame_interval = max(1, int(round(original_fps / target_fps)))
    # print(f"Saving 1 frame every {frame_interval} frames")
    
    frame_count = 0
    saved_count = 0
    
    image_path_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame if it's the right interval
        if frame_count % frame_interval == 0:
            # Dynamically adjust font size and position based on image size
            height, width, _ = frame.shape
            width = int(width * scale_factor)
            height = int(height * scale_factor)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            
            font_scale = min(width, height) / (250)  # Scale font size based on image dimensions
            thickness = max(1, int(font_scale * 2))  # Adjust thickness
            position = (int(width * 0.02), int(height * 0.1))  # Position at 2% width, 10% height

            # Add the frame index to the image
            text = f"index {saved_count:02d}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (0, 255, 0)  # Green color
            
            # Add text to the frame
            cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
            
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            frame_filename = os.path.abspath(frame_filename)
            cv2.imwrite(frame_filename, frame)
            # image_path_list.append(frame_filename)
            image_path_list.append(f"file://{frame_filename}")
            # print(f"Saved frame {saved_count} from original frame {frame_count}")
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    # print(f"Finished pre-processing.\nSaved {saved_count} frames from original {frame_count} frames.")

    return image_path_list




def save_text_to_json(data, output_file, indent=4):
    """I
    Save the data as a JSON file.
    
    Args:
    - data (dict): The data structure to be saved.
    - output_file (str): The path of the output JSON file.
    - indent (int): The number of spaces for JSON indentation, with a default value of 4.
    """
    try:
        # 写入 JSON 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        
        # print(f"\nSuccessfully saved the data to {output_file}")
    except Exception as e:
        print(f"\nError saving file: {e}")




def frameidx_to_timestep(frame_idx, fps, video_path):
    """
    Convert frame index to time step in seconds.
    
    Args:
        frame_idx (int): The index of the frame.
        fps (float): Frames per second of the video.
    
    Returns:
        float: Time step in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    original_fps = cap.get(cv2.CAP_PROP_FPS) ## input_video 的原始帧率，即每秒帧数

    frame_interval = fps / original_fps
    time_step = frame_idx * frame_interval

    return time_step


"""
# NOTE: Paths can be configured in config.py (ROBOMIMIC_DATA_ROOT, etc.)
# Examples below use default configuration: paths resolved from config.py

# Using config helper (recommended):
from config import get_task_paths
paths = get_task_paths('can')
# paths['video_dir'] will give you the video directory

# Or use demo scripts which handle paths automatically:
# See demo/01_prepare_videos.sh

# Manual examples with default paths:
python divide_video.py "--dataset_folder=./data/robomimic/datasets/square/ph/video/" \
    --out_dir='./data/divided_events/' --task=square --model=qvq-max --frame_exist=True --target_fps=10 --use_image_list=False

TASK=can
TASK=square
TASK=tool_hang
python divide_video.py "--dataset_folder=./data/robomimic/datasets/${TASK}/ph/video/"  \
    --out_dir='./data/divided_events/' --task="${TASK}" --model=3-vl-235b-t  --frame_exist=True --target_fps=5 --scale=1 --version=1
"""

@click.command()
@click.option('--dataset_folder', type=str, help='Path to the input video file or a list of images')
@click.option('--task', type=str, help='task name')
@click.option('--model', type=str, default='max', help='Model to use for video understanding (default: qwen-vl-max)')
@click.option('--num_video', type=int, default=1, help='Number of videos to process (default: 1)')
@click.option('--out_dir', type=str, help='Directory to save the output JSON file')
@click.option('--target_fps', type=int, default=10, help='Target frames per second for video processing (default: 2)')
@click.option('--use_image_list', type=bool, default=True, help='Use a list of images instead of a video file')
@click.option('--frame_exist', type=bool, default=True, help='Whether the frame images already exist, if True, skip the video to image step')
@click.option('--version', type=int, default=0, help='version of the prompt')
@click.option('--scale',type=int, default=1, help='to scale up the frames')
def main(dataset_folder, task, model, num_video, out_dir, target_fps, frame_exist, use_image_list, version, scale):
    '''
    Code logic:
    1. Convert the video into a list of images, or directly use the video path.
    2. Define the task name, object list, robot list, surface list, and video duration.
    3. Define the prompt, including task description, object, robot, surface information, and video duration.
    4. Call the video_understanding function for video understanding.
    5. Save the results as a JSON file.
    '''
    args = {'dataset_folder': dataset_folder, 'task': task, 'model': model, 'num_video': num_video, 
            'out_dir': out_dir, 'target_fps': target_fps, 'frame_exist': frame_exist, 'use_image_list': use_image_list, 
            'version': version, 'scale': scale}
    # dataset_root = '~/github/diff/diffusion_policy_guidance/data/robomimic/datasets/can/ph/video/'
    dataset_folder = os.path.expanduser(dataset_folder)
    assert os.path.exists(dataset_folder), f"Input path {dataset_folder} does not exist."

    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir, exist_ok=True)

    (
        TASK_NAME,
        OBJECT_LIST,
        ROBOT_LIST,
        SURFACE_LIST,
        DEMO_ID,
        TIME_LENGTH,
        FOLDER_NAME,
    ) = get_task_config(task)
    PROMPT = f"""
        Task: {TASK_NAME}
        Objects in the scene: {OBJECT_LIST}
        Surfaces in the scene: {SURFACE_LIST}
        Robots in the scene: {ROBOT_LIST}
        Video duration: {TIME_LENGTH} seconds
        
        You will analyze a sequence of video frames to divide it into consecutive subtasks.
        Each subtask represents ONE primary robot action (e.g., "grasp object", "place object", "hang object").
        
        FRAME INDEXING GUIDE:
        - If analyzing a video: frames are sampled at the specified fps, with indices naming along with the sequence order (0, 1, 2, ...).
        - If analyzing image files: the files are named as frame_index.jpg. The <frame_index> is also labelled using green words at the top left side of each image.
        
        SUBTASK DIVISION PRINCIPLE:
        Divide the video into subtasks based on the GOAL of each action sequence:
        - **PICK subtask**: Robot picks up an object (grasp + detach from surface)
        - **PLACE/DROP subtask**: Robot places object on surface and releases (release → attach_drop)
        - **HANG/INSERT subtask**: Robot hangs/inserts object then releases (attach_hang + release)
        
        For simple pick-and-place: separate PICK and PLACE subtasks
        For pick-and-hang (continuous): can combine grasp, detach, attach_hang, release in ONE subtask if continuous
        
        SUBTASK PHASE DEFINITIONS:
        Each <subtask> consists of three phases: targeting (required), interaction (required), and result (optional).
        Each phase has the form {{'start_time': <t1>, 'end_time': <t2>, 'connections': <con>}}.
        
        1. **TARGETING phase**: Robot moves toward target object/location.
           - Typically has empty connections []
           - Ends when robot is positioned to begin interaction
        
        2. **INTERACTION phase**: Robot ACTIVELY performs actions.
           - 'grasp': robot gripper closes to hold object
           - 'release': robot gripper opens to let go
           - 'attach_hang': robot actively places/inserts object while holding
           - 'detach': object separates from surface (happens with grasp, as robot lifts object)
        
        3. **RESULT phase** (optional): PASSIVE consequence that happens AFTER robot action.
           - ONLY 'attach_drop': object lands on surface after being released
           - This has a clear visual signal: object visibly falls and lands
           - Only used for DROP subtasks
        
        WORKFLOW EXAMPLES:
        - **GRASP subtask**: targeting → interaction[grasp, detach]
        - **DROP subtask**: targeting → interaction[release] → result[attach_drop]
        - **HANG/INSERT subtask**: targeting → interaction[attach_hang, release]
        - **INSERT without release**: targeting → interaction[attach_hang]
        
        CONNECTION FORMAT:
        Each connection element is [<edge>, <primitive>, <h_time>], where:
            - <edge>: [A, B] indicates interaction between A and B
            - <primitive>: 'grasp', 'release', 'attach_hang', 'attach_drop', or 'detach'
            - <h_time>: the frame index when the interaction occurs

        CRITICAL RULES: 
        1) No explanation needed.
        2) For simple pick-and-place: use separate subtasks (PICK subtask, then PLACE subtask).
        3) For pick-and-hang (continuous motion): can combine in ONE subtask with grasp, detach, attach_hang, release.
        4) 'grasp' and 'detach' go together (robot grabs and lifts object from surface).
        5) 'release' goes in INTERACTION, 'attach_drop' goes in RESULT (object falls after release).
        6) 'attach_hang': object attaches to another object/surface, edge should be [object, target], NOT [robot, object].
        7) All subtasks must be consecutive and non-overlapping, covering the ENTIRE video duration.

        OUTPUT FORMAT:
            ```json
            {{
                "subtask1": {{
                    "targeting": {{'start_time': 0, 'end_time': 8, 'connections': []}},
                    "interaction": {{'start_time': 8, 'end_time': 12, 'connections': [[["robot", "can"], "grasp", 8], [["can", "table"], "detach", 10]]}}
                }},
                "subtask2": {{
                    "targeting": {{'start_time': 12, 'end_time': 20, 'connections': []}},
                    "interaction": {{'start_time': 20, 'end_time': 24, 'connections': [[["robot", "can"], "release", 23]]}},
                    "result": {{'start_time': 24, 'end_time': 28, 'connections': [[["can", "tray"], "attach_drop", 24]]}}
                }}
            }}
            ```
        """

    ## output path
    json_dir = os.path.join(out_dir, f"{TASK_NAME}", 'json_sg')
    if not os.path.exists(json_dir):
        os.makedirs(json_dir, exist_ok=True)

    print(f"Processing {num_video} videos in {dataset_folder}...")

    ## understand each video in the dataset_folder
    for idx in tqdm(range(num_video), desc="Processing videos"):
        local_path = os.path.join(dataset_folder, f'demo_{idx}_video.mp4')

        ## pre-process video 
        if use_image_list:
            ''' convert video to frame_images based on target_fps '''
            frame_images_folder = os.path.join(out_dir, f"{TASK_NAME}", f"image_frame_{idx}")
            input_path = save_video_frames(
                local_path, 
                frame_images_folder, 
                target_fps=target_fps,
                frame_exist=frame_exist,
                scale_factor=scale
                ) 
            # print(input_path[0:5], '...', input_path[-5:])
            # print(f"Saved frame images to {frame_images_folder}\n")

        else:
            input_path = local_path

        ## *** process the video or image list ***
        if model == '2.5-32b':
            model_name = 'qwen2.5-vl-32b-instruct'
        elif model == '2.5-72b':
            model_name = 'qwen2.5-vl-72b-instruct'
        elif model == 'qwen-plus':
            model_name = 'qwen-vl-plus'
        elif model == 'qwen-plus-latest':
            model_name = 'qwen-vl-plus-latest'
        elif model == 'qwen-max':
            model_name = 'qwen-vl-max'
        elif model == 'qwen-max-latest':
            model_name = 'qwen-vl-max-latest'
        elif model == 'qvq-plus':
            model_name = 'qvq-plus'
        elif model == 'qvq-plus-latest':
            model_name = 'qvq-plus-latest'
        elif model == 'qvq-max':
            model_name = 'qvq-max'
        elif model == 'qvq-max-latest':
            model_name = 'qvq-max-latest' # 2025-06-16
        elif model == '3-vl-plus':
            model_name = 'qwen3-vl-plus'
        elif model == '3-vl-235b-t':
            model_name = 'qwen3-vl-235b-a22b-thinking'
        elif model == '3-vl-235b-i':
            model_name = 'qwen3-vl-235b-a22b-instruct'
        else:
            raise NotImplementedError(f"Model {model} not implemented.")
        result = video_understanding(
            input_path, 
            PROMPT, 
            model=model_name,    
            fps=target_fps
        )
        # print(result.split("```"))

        # Save the result to a JSON file
        if result is not None:
            json_str = "\n".join(result.split("```")[1].split("\n")[1:]) if "```json" in result else result
            json_output = json.loads(json_str)
            json_output['prompt'] = PROMPT
            json_output['args'] = args

            ## convert frame_idx to timestep, saving to json_output
            if 'ModeChangeDetection' in json_output:
                for change in json_output['ModeChangeDetection']:
                    frame_idx = change['frame_idx']
                    change['timestep_orig'] = frameidx_to_timestep(
                        frame_idx, 
                        target_fps,
                        video_path=local_path
                    )

            input_type = 'imglist' if use_image_list else 'video'
            json_path = os.path.join(json_dir, f"{input_type}_episode_{idx}_{model_name}_fps_{target_fps}_scale_{scale}_v{version:02d}.json")
            save_text_to_json(json_output, json_path, indent=4)

            print("Done!")
            print(f"Results saved to {json_path}")      
        else:
            print(f"Failed to process video {local_path}.")


if __name__ == "__main__":
    main()