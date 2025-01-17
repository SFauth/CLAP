import sys
sys.path.append("../.")
import os
import torch
import librosa
import glob
from open_clip import create_model
from training.data import get_audio_features
from training.data import int16_to_float32, float32_to_int16
from transformers import RobertaTokenizer

tokenize = RobertaTokenizer.from_pretrained('roberta-base')
def tokenizer(text):
    result = tokenize(
        text,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in result.items()}

def infer_text():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    precision = 'fp32'
    amodel = 'HTSAT-tiny' # or 'PANN-14'
    tmodel = 'roberta' # the best text encoder in our training
    enable_fusion = False # False if you do not want to use the fusion model
    fusion_type = 'aff_2d'
    pretrained = "/home/sfauth/code/CLAP/assets/checkpoints/epoch_top_0.pt" # the checkpoint name, the unfusion model can also be loaded.

    model, model_cfg = create_model(
        amodel,
        tmodel,
        pretrained,
        precision=precision,
        device=device,
        enable_fusion=enable_fusion,
        fusion_type=fusion_type
    )
    # load the text, can be a list (i.e. batch size)
    text_data = ["I love the contrastive learning", "I love the pretrain model"] 
    # tokenize for roberta, if you want to tokenize for another text encoder, please refer to data.py#L43-90 
    text_data = tokenizer(text_data)
    
    text_embed = model.get_text_embedding(text_data)
    print(text_embed.size())

    return text_embed

def infer_audio():
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    precision = 'fp32'
    amodel = 'HTSAT-tiny' # or 'PANN-14'
    tmodel = 'roberta' # the best text encoder in our training
    enable_fusion = False # False if you do not want to use the fusion model
    fusion_type = 'aff_2d'
    pretrained = "/home/sfauth/code/CLAP/assets/checkpoints/epoch_top_0.pt" # the checkpoint name, the unfusion model can also be loaded.

    model, model_cfg = create_model(
        amodel,
        tmodel,
        pretrained,
        precision=precision,
        device=device,
        enable_fusion=enable_fusion,
        fusion_type=fusion_type
    )

    # load the waveform of the shape (T,), should resample to 48000
    audio_waveform, sr = librosa.load("/home/sfauth/code/CLAP/data/clotho_test_data/080809_05_FontanaKoblerov.wav", sr=48000) 
    # quantize
    audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
    audio_waveform = torch.from_numpy(audio_waveform).float()
    audio_dict = {}

    # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
    audio_dict = get_audio_features(
        audio_dict, audio_waveform, 480000, 
        data_truncating='rand_trunc', 
        data_filling='repeatpad',
        audio_cfg=model_cfg['audio_cfg']
    )
    # can send a list to the model, to process many audio tracks in one time (i.e. batch size)
    audio_embed = model.get_audio_embedding([audio_dict])
    print(audio_embed.size())

    return audio_embed
    


if __name__ == "__main__":
    
    audio = infer_audio()
    text = infer_text()

    print(audio.norm())
    print(text.norm())
