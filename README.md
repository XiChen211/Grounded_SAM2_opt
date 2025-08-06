# grounded_sam2_opt
optimize grounded_sam2 with tensorrt


# Download models

```bash
cd checkpoints
bash download_ckpts.sh 
```

# Download onnx models

```bash
cd sam2_opt/sam2/checkpoints
bash download_opt.sh
```
# demo: track_id_demo.py

## how to speedup
```python
    class GroundedSAM2Pipeline:
        def __init__(self, sam2_checkpoint, model_cfg, gdino_model_id, text_query="person.", device="cuda"):
        ...
        # Load SAM2 models for video and image prediction
        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)

        # Apply speed-up optimizations
        self.video_predictor.speedup()
        self.image_predictor.speedup()
        # use predictor like raw version
        # self.predictor.speedup("torch")   # reset to raw version, which support other model version, such as tiny
```

## notice
```python
    # if you download Grounding DINO model to local, use your own path here.
    if __name__ == "__main__":
        ...
        # Path to the Grounding DINO model
        GDINO_MODEL_ID = "./checkpoints/grounding-dino-tiny"
        # if you want to download model from huggingface,
        # set GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
```