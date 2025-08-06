import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_add = os.path.join(current_dir, 'sam2_opt', 'sam2')
if path_to_add not in sys.path:
    sys.path.insert(0, path_to_add)

import shutil
import random
import json
import copy
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

class GroundedSAM2Pipeline:
    def __init__(self, sam2_checkpoint, model_cfg, gdino_model_id, text_query="person.", device="cuda"):
        """
        Initializes the pipeline by loading all necessary models.
        """
        self.device = device
        self.text_query = text_query
        self.step = 30  # Process video in chunks of 30 frames
        self.prompt_type = "mask"
        self.frame_rate = 30

        print(f"[INFO] Initializing models on device: {self.device}")
        torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()
        # Enable TF32 for better performance on Ampere GPUs and newer
        if self.device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load SAM2 models for video and image prediction
        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)

        # Apply speed-up optimizations
        self.video_predictor.speedup()
        self.image_predictor.speedup()

        # Load Grounding DINO model for object detection from text prompts
        self.processor = AutoProcessor.from_pretrained(gdino_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_model_id).to(self.device)
        print("[INFO] Models initialized successfully.")

    def extract_frames(self, video_path, frame_dir):
        """
        Extracts frames from a video file into a specified directory.
        """
        print(f"[INFO] Extracting frames from {video_path} to {frame_dir}...")
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        CommonUtils.extract_frames_ffmpeg(video_path, frame_dir)
        print("[INFO] Frame extraction complete.")

    def cleanup(self, *dirs):
        """
        Removes specified temporary directories.
        """
        for d in dirs:
            if os.path.exists(d):
                shutil.rmtree(d)
                # print(f"[DEBUG] Cleaned up directory: {d}")

    def save_track_images_with_frame_info(self, mask_dir, json_dir, frame_dir, save_root, video_name):
        """
        For each tracked object, extracts its image crops from the video 
        and saves them using the original frame number as the filename.
        """
        # Create a video-specific root directory for saving
        save_root = os.path.join(save_root, video_name)

        # A dictionary to organize data: {track_id -> [(frame_path, binary_mask), ...]}
        track_dict = {}  

        # 1. Collect and organize data by track ID
        for json_file in sorted(os.listdir(json_dir)):
            if not json_file.endswith(".json"):
                continue

            # Parse frame index from the filename
            frame_idx_str = json_file.replace("mask_", "").replace(".json", "")
            frame_name = f"{frame_idx_str}.jpg"
            frame_path = os.path.join(frame_dir, frame_name)

            if not os.path.exists(frame_path):
                print(f"[WARN] Frame {frame_path} not found, skipping...")
                continue
            
            mask_path = os.path.join(mask_dir, json_file.replace(".json", ".npy"))
            json_path = os.path.join(json_dir, json_file)

            with open(json_path, "r") as f:
                data = json.load(f)
            mask_np = np.load(mask_path)

            for obj in data["labels"]:
                tid = int(obj)
                binary_mask = (mask_np == tid)

                if tid not in track_dict:
                    track_dict[tid] = []
                # Append (frame_path, mask) tuple to the corresponding track ID list
                track_dict[tid].append((frame_path, binary_mask))

        # 2. Iterate through the organized track_dict and save images for each object
        for tid, samples in track_dict.items():
            # Create a separate save directory for each track ID
            save_dir = os.path.join(save_root, str(tid))
            os.makedirs(save_dir, exist_ok=True)

            # 3. Process and save all samples for the track
            for img_path, binary_mask in samples:
                # Extract the original frame number from the image path
                frame_number_str = os.path.splitext(os.path.basename(img_path))[0]
                
                # Construct the output filename, e.g., "5.png", "88.png"
                output_filename = f"{frame_number_str}.png"
                output_path = os.path.join(save_dir, output_filename)

                # Load image and apply the mask
                img = Image.open(img_path).convert("RGB")
                img_np = np.array(img)
                
                mask = binary_mask.astype(bool)
                masked_img = np.ones_like(img_np) * 255  # White background
                masked_img[mask] = img_np[mask]

                # Save the cropped image with the frame number as its name
                Image.fromarray(masked_img).save(output_path)

    # def save_track_images(self, mask_dir, json_dir, frame_dir, save_root, video_name, max_per_track=30):
    #     """
    #     Saves a random sample of cropped images for each tracked object.
    #     """
    #     save_root = os.path.join(save_root, video_name)
    #     track_dict = {}  # {track_id -> list of (frame_path, binary_mask)}

    #     for json_file in sorted(os.listdir(json_dir)):
    #         if not json_file.endswith(".json"):
    #             continue

    #         frame_idx = json_file.replace("mask_", "").replace(".json", "")
    #         frame_name = f"{frame_idx}.jpg"
    #         frame_path = os.path.join(frame_dir, frame_name)

    #         if not os.path.exists(frame_path):
    #             print(f"[WARN] Frame {frame_path} not found, skipping...")
    #             continue
            
    #         mask_path = os.path.join(mask_dir, json_file.replace(".json", ".npy"))
    #         json_path = os.path.join(json_dir, json_file)

    #         with open(json_path, "r") as f:
    #             data = json.load(f)
    #         mask_np = np.load(mask_path)

    #         for obj in data["labels"]:
    #             tid = int(obj)
    #             binary_mask = (mask_np == tid)
    #             if tid not in track_dict:
    #                 track_dict[tid] = []
    #             track_dict[tid].append((frame_path, binary_mask))

    #     for tid, samples in track_dict.items():
    #         save_dir = os.path.join(save_root, str(tid))
    #         os.makedirs(save_dir, exist_ok=True)

    #         # Choose a random sample of images for each track
    #         chosen = random.sample(samples, min(len(samples), max_per_track))
    #         for idx, (img_path, binary_mask) in enumerate(chosen):
    #             img = Image.open(img_path).convert("RGB")
    #             img_np = np.array(img)
    #             # Create a cropped image with a white background
    #             mask = binary_mask.astype(bool)
    #             masked_img = np.ones_like(img_np) * 255
    #             masked_img[mask] = img_np[mask]
    #             Image.fromarray(masked_img).save(os.path.join(save_dir, f"{idx:03d}.png"))


    def process_video(self, video_path, save_path):
        """
        Main processing function for a single video.
        """
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}. Skipping.")
            return

        print(f"\n--- Processing video: {video_path} ---")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cache_img_dir = f"./cache/{video_name}/frames"
        output_dir = f"./cache/{video_name}/outputs"

        self.extract_frames(video_path, cache_img_dir)
        
        frame_names = sorted(os.listdir(cache_img_dir), key=lambda x: int(os.path.splitext(x)[0]))
        if not frame_names:
            print(f"[ERROR] No frames extracted from {video_path}. Skipping.")
            self.cleanup(cache_img_dir, output_dir)
            return

        # Create directories for saving mask and metadata
        mask_data_dir = os.path.join(output_dir, "mask_data")
        json_data_dir = os.path.join(output_dir, "json_data")
        for d in [mask_data_dir, json_data_dir]:
            os.makedirs(d, exist_ok=True)

        inference_state = self.video_predictor.init_state(video_path=cache_img_dir, offload_video_to_cpu=True, async_loading_frames=True)
        sam2_masks = MaskDictionaryModel() # Stores masks from the most recently processed frame
        objects_count = 0 # Global counter for unique object IDs
        video_segments = {} # Stores results for all frames: {frame_idx: MaskDictionaryModel}

        print(f"[INFO] Starting tracking with prompt: '{self.text_query}'")
        
        # Process video in chunks (steps)
        for start_idx in tqdm(range(0, len(frame_names), self.step), desc=f"Processing {video_name}"):
            # STEP 1: Key-frame detection using Grounding DINO
            # At the beginning of each chunk, detect objects in the first frame (key-frame).
            img_path = os.path.join(cache_img_dir, frame_names[start_idx])
            image = Image.open(img_path)
            image_base_name = os.path.splitext(frame_names[start_idx])[0]
            mask_dict = MaskDictionaryModel(promote_type=self.prompt_type, mask_name=f"mask_{image_base_name}.npy")

            inputs = self.processor(images=image, text=self.text_query, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)
            results = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=0.25, text_threshold=0.25, target_sizes=[image.size[::-1]])

            # STEP 2: Mask generation with SAM2 and ID assignment
            self.image_predictor.set_image(np.array(image.convert("RGB")))
            input_boxes = results[0]["boxes"]
            OBJECTS = results[0]["labels"] 
            if input_boxes.shape[0] != 0:
                # Generate high-quality masks from bounding boxes
                masks, _, _ = self.image_predictor.predict(box=input_boxes, multimask_output=False)
                if masks.ndim == 4: masks = masks.squeeze(1)
                
                # Match new detections with existing tracks or assign new unique IDs
                mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(self.device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
                objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
            else:
                # If no objects are detected, use the masks from the previous frame as a reference
                mask_dict = sam2_masks

            if len(mask_dict.labels) == 0:
                # If no objects are being tracked at all, save empty files for this chunk
                mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list=frame_names[start_idx:start_idx+self.step])
                continue

            # STEP 3: Video propagation (tracking)
            self.video_predictor.reset_state(inference_state)
            # Add the masks from the key-frame as seeds for tracking
            for object_id, object_info in mask_dict.labels.items():
                self.video_predictor.add_new_mask(inference_state, start_idx, object_id, object_info.mask)

            # Propagate masks frame-by-frame for the current chunk
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=self.step, start_frame_idx=start_idx):
                frame_masks = MaskDictionaryModel()
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0)
                    object_info = ObjectInfo(instance_id=out_obj_id, mask=out_mask[0], class_name=mask_dict.get_target_class_name(out_obj_id))
                    object_info.update_box()
                    frame_masks.labels[out_obj_id] = object_info
                
                if frame_masks.labels:
                    # Store the tracking result for this specific frame
                    image_base_name = frame_names[out_frame_idx].split(".")[0]
                    frame_masks.mask_name = f"mask_{image_base_name}.npy"
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]
                    video_segments[out_frame_idx] = frame_masks
                    # Update the short-term memory with the latest tracking result for the next chunk's matching
                    sam2_masks = copy.deepcopy(frame_masks)

        # STEP 4: Save all tracking results to files
        print("[INFO] Saving tracking results...")
        for frame_idx, frame_masks_info in video_segments.items():
            mask = frame_masks_info.labels
            # Create a single mask image where pixel value corresponds to object ID
            mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
            for obj_id, obj_info in mask.items():
                mask_img[obj_info.mask == True] = obj_id

            # Save the mask image as a .npy file
            mask_img = mask_img.numpy().astype(np.uint16)
            np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

            # Save metadata (bboxes, labels) as a .json file
            json_data = frame_masks_info.to_dict()
            json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
            with open(json_data_path, "w") as f:
                json.dump(json_data, f)
        
        print("[INFO] Saving tracked object images...")
        # Save cropped images of each tracked object
        self.save_track_images_with_frame_info(mask_data_dir, json_data_dir, cache_img_dir, save_path, video_name)
        
        # Clean up temporary files
        self.cleanup(cache_img_dir, output_dir)
        print(f"--- Finished processing {video_path} ---")


    def run_from_csv(self, csv_path, save_path):
        """
        Reads a CSV file with video paths and processes each video.
        """
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"[FATAL ERROR] CSV file not found at: {csv_path}")
            print("Please make sure the file exists and the path is correct.")
            return

        if "vid_path" not in df.columns:
            print(f"[FATAL ERROR] CSV file must contain a 'vid_path' column.")
            return

        for video_path in df["vid_path"]:
            # Handle relative paths: assume they are relative to the CSV file's location
            if not os.path.isabs(video_path):
                csv_dir = os.path.dirname(csv_path)
                video_path = os.path.join(csv_dir, video_path)

            try:
                self.process_video(video_path, save_path)
            except Exception as e:
                # Catch errors during a single video's processing to allow the batch to continue
                print(f"[CRITICAL ERROR] An unexpected error occurred while processing {video_path}:")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    # Path to the SAM2 model checkpoint
    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
    # Path to the SAM2 model configuration file
    MODEL_CFG = "sam2.1_hiera_l.yaml"
    # Path to the Grounding DINO model
    GDINO_MODEL_ID = "./checkpoints/grounding-dino-tiny"
    
    # Text prompt for object detection. Use " . " to separate multiple classes.
    TEXT_PROMPT = "child ." 
    
    # Path to the CSV file containing video paths
    CSV_FILE_PATH = "test.csv"
    
    # Path to the directory where final results will be saved
    SAVE_PATH = "./outputs"

    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"[INFO] Results will be saved to: {SAVE_PATH}")

    pipeline = GroundedSAM2Pipeline(
        sam2_checkpoint=SAM2_CHECKPOINT,
        model_cfg=MODEL_CFG,
        gdino_model_id=GDINO_MODEL_ID,
        text_query=TEXT_PROMPT
    )

    print(f"[INFO] Starting processing from CSV file: {CSV_FILE_PATH}")
    pipeline.run_from_csv(CSV_FILE_PATH, save_path=SAVE_PATH)
    print("\n[INFO] All videos processed.")