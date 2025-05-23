from datasets import load_dataset
import os


os.environ["HF_HOME"] = "D:/belajar/audio/vad/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/belajar/audio/vad/hf_cache/models"
os.environ["HF_DATASETS_CACHE"] = "D:/belajar/audio/vad/hf_cache/datasets"

try:
    ds = load_dataset(
        "google/fleurs-r",
        "default",
        split="test",
        trust_remote_code=True,
        cache_dir="D:/belajar/audio/vad/hf_cache"
    )
    print("Success, My Lord! google/fleurs-r loaded without remote code!")
    print(ds)
except Exception as e:
    print(f"Oh dear, My Lord, an error occurred: {e}")
    print("This might indicate that 'google/fleurs-r' does require remote code,")
    print("or there might be another issue (like network or config name).")
    print("If the error message specifically mentions remote code or a script, then we have our answer!")