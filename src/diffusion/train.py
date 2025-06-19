from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

def main():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    emotions = {
        0: "angry",
        1: "disgusted",
        2: "fearful",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprised",
    }

    out_root = Path("synthetic_fer2013")
    out_root.mkdir(exist_ok=True)

    num_per_class = 1
    for label, emo in emotions.items():
        folder = out_root / str(label)
        folder.mkdir(exist_ok=True)
        prompt = f"A close-up portrait of a person looking {emo}, realistic photo"
        for i in range(num_per_class):
            img = pipe(prompt, guidance_scale=7.5, num_inference_steps=30).images[0]
            img.save(folder / f"{emo}_{i:03d}.png")

if __name__ == "__main__":
    main()
