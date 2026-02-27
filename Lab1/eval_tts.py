import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")

# model = Qwen3TTSModel.from_pretrained(
#     "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
#     device_map="cuda:0",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
# )

model = Qwen3TTSModel.from_pretrained(
    "Qwen3-TTS-12Hz-0.6B-Base",
    dtype=dtype,
    device_map=device
)

ref_audio = "resources/clone.wav"
ref_text  = "When someone comes up and says something like, I am a God, everybody says, who does he think he is? I just told you who I thought I was. A God! I just told you, that's who I think I am."

wavs, sr = model.generate_voice_clone(
    text="I lost the only girl in the world that know me best. I got the money and the fame, man, that don't mean shit. I got the Jesus on the chain, man, that don't mean shit. Cause when the Jesus pieces can't bring me peace",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write("output_voice_clone.wav", wavs[0], sr)