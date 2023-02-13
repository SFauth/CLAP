#%%
import sys
sys.path.append("../.")
import os
import torch
import librosa
import glob
import pandas as pd
from open_clip import create_model
from training.data import get_audio_features
from training.data import int16_to_float32, float32_to_int16
from transformers import RobertaTokenizer
#%%
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

def infer_text(text_data):
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
        enable_fusion=False,
        fusion_type=fusion_type
    )
    # load the text, can be a list (i.e. batch size)
    #text_data = ["I love the contrastive learning", "I love the pretrain model"] 
    # tokenize for roberta, if you want to tokenize for another text encoder, please refer to data.py#L43-90 
    text_data = tokenizer(text_data)
    
    text_embed = model.get_text_embedding(text_data)

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

    logit_scale_a, logit_scale_t = model(None, None, device)
    logit_scale_a = logit_scale_a.cpu()

    # load the waveform of the shape (T,), should resample to 48000
    file_list = glob.glob("/home/sfauth/code/CLAP/data/clotho_test_data/*.wav")

    #file = file_list[232]

    file = "/home/sfauth/code/CLAP/data/clotho_test_data/080809_05_FontanaKoblerov.wav"

    audio_waveform, sr = librosa.load(file, sr=48000) 

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

    return audio_embed, file, logit_scale_a
    


if __name__ == "__main__":

    # 5 experiments (HTML files)
    # Automatize getting GT caption
 
    
    audio_embed, file, logit_scale_a = infer_audio()
    
    wav_only = os.path.split(file)[1]
    path_to_wav = os.path.join("clotho_test_data", wav_only) # place close to html to avoid errors with not finding the file

    # get GT captions

    gt_captions = pd.read_csv("sim_tests/clotho_test_data/clotho_captions_evaluation.csv")
    captions = gt_captions[gt_captions["file_name"] == wav_only].iloc[:, 3:5].stack().tolist() #take 3rd and 4th GT caption
    
    captions.insert(0, "I love the contrastive learning")
    
    caption_list = ["water", "this is water", "this is an audio clip of water", "this is an audio clip of water running"]

    for caption in caption_list:

        captions.insert(len(captions), caption)

    captions = [' the', ' a', ' this', ' an', ' "', "water", "this is water", "this is an audio clip of water", "this is an audio clip of water running"]
    # " gets highest score, which shouldn't be!!!!!!

    text_embed = infer_text(captions) 

    #similarities = text_embed @ audio_embed.T
    #similarities = similarities.cpu().data.numpy()


    logits_per_text = logit_scale_a * torch.cosine_similarity(audio_embed, text_embed) 
    logits_per_image = torch.unsqueeze(logits_per_text.T, 0)

    similarities = logits_per_image.softmax(dim=-1).cpu().detach().numpy().T

    
    
    #only_wav = os.path.split(file)[1]
    wav_col = pd.Series([file] * len(captions))

    
    sim_text = pd.concat([pd.DataFrame(similarities), pd.Series(captions)], axis=1)
    sim_text = pd.concat([sim_text, wav_col], axis=1)
    sim_text.columns = ["dot_product(text, audio.T)", "Text", "Audio"]

    #sim_text["Audio"] = pd.Series([f"""<audio controls> <source src="clotho_test_data/105bpm.wav" type="audio/wav"> </audio>"""] * len(text))

    sim_text["Audio"] = sim_text["Audio"].apply(lambda audio_path: f"""<audio controls> <source src="{path_to_wav}" type="audio/wav"> </audio>""")
    
    # save table as html

    html_filename =  wav_only + ".html"
    root_path = os.getcwd()
    html_path = os.path.join(root_path, "sim_tests", html_filename)
    sim_text.to_html(html_path, escape=False)
    

    """
    - fluctuates somewhat, but the winner stays the winner, which is what matters to us
    """
    