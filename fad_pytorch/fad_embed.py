# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_fad_embed.ipynb.

# %% auto 0
__all__ = ['OPENL3_VERSION', 'GUDGUD_LICENSE', 'download_file', 'download_if_needed', 'get_ckpt', 'setup_embedder', 'embed',
           'main']

# %% ../nbs/02_fad_embed.ipynb 5
import os
import numpy as np
import argparse
import laion_clap 
from laion_clap.training.data import get_audio_features
from accelerate import Accelerator
import warnings
import torch

from aeiou.core import get_device, load_audio, get_audio_filenames, makedir
from aeiou.datasets import AudioDataset
from aeiou.hpc import HostPrinter
from torch.utils.data import DataLoader
from pathlib import Path
import requests 
from tqdm import tqdm
import site 
from einops import rearrange

try:
    from fad_pytorch.pann import Cnn14_16k
except: 
    from pann import Cnn14_16k
    
# there are TWO 'torchopenl3' repos!  they operate differently.
OPENL3_VERSION = "turian" #  #  "hugo" | "turian". set to which version you've installed
import torchopenl3

# %% ../nbs/02_fad_embed.ipynb 7
def download_file(url, local_filename):
    "Includes a progress bar.  from https://stackoverflow.com/a/37573701/4259243"
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kilobye
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(local_filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    return local_filename

def download_if_needed(url, local_filename, accelerator=None):
    "wrapper for download file"
    if accelerator is None or accelerator.is_local_main_process:  # Only do this on one process instead of all
        if not os.path.isfile(local_filename):
            print(f"File {local_filename} not found, downloading from {url}")
            download_file( url, local_filename)
    if accelerator is not None: accelerator.wait_for_everyone()
    return local_filename

def get_ckpt(ckpt_file='music_speech_audioset_epoch_15_esc_89.98.pt',
             ckpt_base_url='https://huggingface.co/lukewys/laion_clap/blob/main',
             ckpt_dl_path=os.path.expanduser("~/checkpoints"),
             accelerator=None,
            ):
    ckpt_path = f"{ckpt_dl_path}/{ckpt_file}"
    download_if_needed( f"{ckpt_base_url}/{ckpt_file}" , ckpt_path)
    return ckpt_path

# %% ../nbs/02_fad_embed.ipynb 8
def setup_embedder(
        model_choice='clap', # 'clap' | 'vggish' | 'pann'
        device='cuda',
        ckpt_file='music_speech_audioset_epoch_15_esc_89.98.pt',  # NOTE: 'CLAP_CKPT' env var overrides ckpt_file kwarg
        ckpt_base_url='https://huggingface.co/lukewys/laion_clap/resolve/main',
        # https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt
        accelerator=None,
        ckpt_dl_path=os.path.expanduser("~/checkpoints"),
    ):
    "load the embedder model"
    embedder = None
    
    sample_rate = 16000
    if model_choice == 'clap':
        print(f"Starting basic CLAP setup")
        clap_fusion, clap_amodel = True, "HTSAT-base"
        #doesn't work:  warnings.filterwarnings('ignore')  # temporarily disable CLAP warnings as they are super annoying. 
        clap_module = laion_clap.CLAP_Module(enable_fusion=clap_fusion, device=device, amodel=clap_amodel).requires_grad_(False).eval()
        clap_ckpt_path = os.getenv('CLAP_CKPT')  # NOTE: CLAP_CKPT env var overrides ckpt_file kwarg
        if clap_ckpt_path is not None:
            #print(f"Loading CLAP from {clap_ckpt_path}")
            clap_module.load_ckpt(ckpt=clap_ckpt_path)
        else:
            print(f"No CLAP checkpoint specified, using {ckpt_file}") 
            clap_module = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
            ckpt_path = get_ckpt(ckpt_file=ckpt_file, ckpt_base_url=ckpt_base_url, ckpt_dl_path=ckpt_dl_path, accelerator=accelerator)
            clap_module.load_ckpt(ckpt_path)
            #clap_module.load_ckpt(model_id=1, verbose=False)
        #warnings.filterwarnings("default")   # turn warnings back on. 
        embedder = clap_module # synonyms 
        sample_rate = 48000
        
    # next two model loading codes from gudgud96's repo: https://github.com/gudgud96/frechet-audio-distance, LICENSE below
    elif model_choice == "vggish":   # https://arxiv.org/abs/1609.09430
        embedder = torch.hub.load('harritaylor/torchvggish', 'vggish')
        use_pca=False
        use_activation=False
        if not use_pca:  embedder.postprocess = False
        if not use_activation: embedder.embeddings = torch.nn.Sequential(*list(embedder.embeddings.children())[:-1])
        sample_rate = 16000

    elif model_choice == "pann": # https://arxiv.org/abs/1912.10211
        sample_rate = 16000
        model_path = os.path.join(torch.hub.get_dir(), "Cnn14_16k_mAP%3D0.438.pth")
        if accelerator is None or accelerator.is_local_main_process:
            if not(os.path.exists(model_path)):
                torch.hub.download_url_to_file('https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth', model_path)
        if accelerator is not None: accelerator.wait_for_everyone()
        embedder = Cnn14_16k(sample_rate=sample_rate, window_size=512, hop_size=160, mel_bins=64, fmin=50, fmax=8000, classes_num=527)
        checkpoint = torch.load(model_path, map_location=device)
        embedder.load_state_dict(checkpoint['model'])
            
    elif model_choice == "openl3" and OPENL3_VERSION == "hugo":  # hugo flores garcia's torchopenl3, https://github.com/hugofloresgarcia/torchopenl3
        # openl3 repo doesn't install its weights if you do "pip install git+...", so here we download them separately
        weights_dir = f"{site.getsitepackages()[0]}/torchopenl3/assets/weights"
        makedir(weights_dir)
        download_if_needed("https://github.com/hugofloresgarcia/torchopenl3/raw/main/torchopenl3/assets/weights/env-mel128", 
                           f"{weights_dir}/music-mel128", accelerator=accelerator)
        embedder = torchopenl3.OpenL3Embedding(input_repr='mel128', embedding_size=512, content_type='music')
        sample_rate = 48000
        
    elif model_choice == "openl3" and OPENL3_VERSION == "turian":  # turian et al's torchopenl3, https://github.com/torchopenl3/torchopenl3
        sample_rate = 48000
        embedder = torchopenl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music", embedding_size=512)
        pass # turian et al's does all its setup when it's invoked 
    else:
        raise ValueError("Sorry, other models not supported yet")
        
    if hasattr(embedder,'eval'): embedder.eval()   
    return embedder, sample_rate


GUDGUD_LICENSE = """For VGGish implementation:
MIT License

Copyright (c) 2022 Hao Hao Tan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# %% ../nbs/02_fad_embed.ipynb 10
def embed(args): 
    model_choice, real_path, fake_path, chunk_size, sr, max_batch_size, debug = args.embed_model, args.real_path, args.fake_path, args.chunk_size, args.sr, args.batch_size, args.debug
    
    sample_rate = sr
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddps = f"[{local_rank}/{world_size}]"  # string for distributed computing info, e.g. "[1/8]" 

    accelerator = Accelerator()
    hprint = HostPrinter(accelerator)  # hprint only prints on head node
    device = accelerator.device    # get_device()
    hprint(f"{ddps} args = {args}")
    hprint(f'{ddps} Using device: {device}')
    
 
    """ # let accelerate split up the files among processsors
    # get the list(s) of audio files
    real_filenames = get_audio_filenames(real_path)
    #hprint(f"{ddps} real_path, real_filenames = {real_path}, {real_filenames}")
    fake_filenames = get_audio_filenames(fake_path)
    minlen = len(real_filenames)
    if len(real_filenames) != len(fake_filenames):
        hprint(f"{ddps} WARNING: len(real_filenames)=={len(real_filenames)} != len(fake_filenames)=={len(fake_filenames)}. Truncating to shorter list") 
        minlen = min( len(real_filenames) , len(fake_filenames) )
    
    # subdivide file lists by process number
    num_per_proc = minlen // world_size
    start = local_rank * num_per_proc
    end =  minlen if local_rank == world_size-1 else (local_rank+1) * num_per_proc
    #print(f"{ddps} start, end = ",start,end) 
    real_filenames, fake_filenames = real_filenames[start:end], fake_filenames[start:end]
    """

    model_choices = [model_choice] if model_choice != 'all' else ['clap','vggish','pann','openl3']
    
    for model_choice in model_choices: # loop over multiple embedders
        hprint(f"\n ** Model_choice = {model_choice}")
        # setup embedder and dataloader
        embedder, emb_sample_rate = setup_embedder(model_choice, device=device, accelerator=accelerator)
        if sr != emb_sample_rate:
            hprint(f"\n*******\nWARNING: sr={sr} != {model_choice}'s emb_sample_rate={emb_sample_rate}. Will resample audio to the latter\n*******\n")
            sr = emb_sample_rate
        hprint(f"{ddps} Embedder '{model_choice}' ready to go!")

        # we read audio in length args.sample_size, cut it into chunks of args,chunk_size to embed, and skip args.hop_size between chunks
        # pads with zeros btw
        real_dataset = AudioDataset(real_path,  sample_rate=emb_sample_rate, sample_size=args.sample_size, return_dict=True, verbose=args.verbose)
        fake_dataset = AudioDataset(fake_path,  sample_rate=emb_sample_rate, sample_size=args.sample_size, return_dict=True, verbose=args.verbose)
        batch_size = min( len(real_dataset) // world_size , max_batch_size ) 
        hprint(f"\nGiven max_batch_size = {max_batch_size}, len(real_dataset) = {len(real_dataset)}, and world_size = {world_size}, we'll use batch_size = {batch_size}")
        real_dl = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
        fake_dl = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)

        real_dl, fake_dl, embedder = accelerator.prepare( real_dl, fake_dl, embedder )  # prepare handles distributing things among GPUs

        # note that we don't actually care if real & fake files are pulled in the same order; we'll only be comparing the *distributions* of the data.
        with torch.no_grad():
            for dl, name in zip([real_dl, fake_dl],['real','fake']):  
                for i, data_dict in enumerate(dl):  # load audio files
                    audio_sample_batch, filename_batch = data_dict['inputs'], data_dict['filename']
                    newdir_already = False
                    if not newdir_already: 
                        p = Path( filename_batch[0] )
                        dir_already = True
                        newdir = f"{p.parents[0]}_emb_{model_choice}"
                        hprint(f"creating new directory = {newdir}")
                        makedir(newdir) 
                        newdir_already = True
                    # cut audio samples into chunks spaced out by hops, and loop over them
                    hop_samples = int(args.hop_size * args.sample_size)
                    hop_starts = np.arange(0, args.sample_size, hop_samples)
                    if args.max_hops <= 0:  
                        hop_starts = hop_starts[:min(len(hop_starts), args.max_hops)]
                    if args.sample_size - hop_starts[-1] < args.hop_size: # judgement call: let's not zero-pad on the very end, rather just don't do the last hop
                        hop_starts = hop_starts[:-1]
                    for h_ind, hop_loc in enumerate(hop_starts):               # proceed through audio file batch via chunks, skipping by hop_samples each time
                        chunk = audio_sample_batch[:,:,hop_loc:hop_loc+hop_samples]
                        audio = chunk 
                        
                        #print(f"{ddps} i = {i}/{len(real_dataset)}, filename = {filename_batch[0]}")
                        audio = audio.to(device)


                        if model_choice == 'clap': 
                            while len(audio.shape) < 3: 
                                audio = audio.unsqueeze(0) # add batch and/or channel dims 
                            embeddings = accelerator.unwrap_model(embedder).get_audio_embedding_from_data(audio.mean(dim=1).to(device), use_tensor=True).to(audio.dtype)

                        elif model_choice == "vggish":
                            audio = torch.mean(audio, dim=1)   # vggish requries we convert to mono
                            embeddings = []                    # ...whoa, vggish can't even handle batches?  we have to pass 'em through singly?
                            for bi, waveform in enumerate(audio): 
                                e =  accelerator.unwrap_model(embedder.to(torch.device("cpu"))).forward(waveform.cpu().numpy(), emb_sample_rate)
                                embeddings.append(e) 
                            embeddings = torch.cat(embeddings, dim=0)

                        elif model_choice == "pann": 
                            audio = torch.mean(audio, dim=1)  # mono only.  todo:  keepdim=True ?
                            out = embedder.forward(audio, None)
                            embeddings = out['embedding'].data

                        elif model_choice == "openl3" and OPENL3_VERSION == "hugo":
                            ##audio = torch.mean(audio, dim=1)  # mono only.
                            embeddings = []
                            for bi, waveform in enumerate( audio.cpu().numpy() ): # no batch processing, expects numpy 
                                e = torchopenl3.embed(model=embedder, 
                                    audio=waveform, # shape sould be (channels, samples)
                                    sample_rate=emb_sample_rate, # sample rate of input file
                                    hop_size=1,  device=device)
                                if debug: hprint(f"bi = {bi}, waveform.shape = {waveform.shape},  e.shape = {e.shape}") 
                                embeddings.append(torch.tensor(e))
                            embeddings = torch.cat(embeddings, dim=0)

                        elif model_choice == "openl3" and OPENL3_VERSION == "turian":
                            # Note: turian's can/will do multiple time-stamped embeddings if the sample_size is long enough. but our chunks/hops precludes this

                            #not needed, turns out: audio = renot needed, turns out: arrange(audio, 'b c s -> b s c')       # this torchopen3 expects channels-first ordering
                            embeddings, timestamps = torchopenl3.get_audio_embedding(audio, emb_sample_rate, model=embedder)
                            embeddings = torch.squeeze(embeddings, 1)        # get rid of any spurious dimensions of 1 in middle position 

                        else:
                            raise ValueError(f"Unknown model_choice = {model_choice}")

                        hprint(f"embeddings.shape = {embeddings.shape}")
                        # TODO: for now we'll just dump each batch on each proc to its own file; this could be improved
                        outfilename = f"{newdir}/emb_p{local_rank}_b{i}_h{h_ind}.pt"
                        hprint(f"{ddps} Saving embeddings to {outfilename}")
                        torch.save(embeddings.cpu().detach(), outfilename)
                    
        del embedder
        torch.cuda.empty_cache()
        # end loop over various embedders
    return     

def embed_one_directory(args): 
    model_choice, path, chunk_size, sr, max_batch_size, debug = args.embed_model, args.real_path, args.chunk_size, args.sr, args.batch_size, args.debug
    
    sample_rate = sr
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddps = f"[{local_rank}/{world_size}]"  # string for distributed computing info, e.g. "[1/8]" 

    accelerator = Accelerator()
    hprint = HostPrinter(accelerator)  # hprint only prints on head node
    device = accelerator.device    # get_device()
    hprint(f"{ddps} args = {args}")
    hprint(f'{ddps} Using device: {device}')

    model_choices = [model_choice] if model_choice != 'all' else ['clap','vggish','pann','openl3']
    
    for model_choice in model_choices: # loop over multiple embedders
        hprint(f"\n ** Model_choice = {model_choice}")
        # setup embedder and dataloader
        embedder, emb_sample_rate = setup_embedder(model_choice, device=device, accelerator=accelerator)
        if sr != emb_sample_rate:
            hprint(f"\n*******\nWARNING: sr={sr} != {model_choice}'s emb_sample_rate={emb_sample_rate}. Will resample audio to the latter\n*******\n")
            sr = emb_sample_rate
        hprint(f"{ddps} Embedder '{model_choice}' ready to go!")

        # we read audio in length args.sample_size, cut it into chunks of args,chunk_size to embed, and skip args.hop_size between chunks
        # pads with zeros btw
        dataset = AudioDataset(path,  sample_rate=emb_sample_rate, sample_size=args.sample_size, return_dict=True, verbose=args.verbose)
        batch_size = min( len(dataset) // world_size , max_batch_size ) 
        hprint(f"\nGiven max_batch_size = {max_batch_size}, len(dataset) = {len(dataset)}, and world_size = {world_size}, we'll use batch_size = {batch_size}")
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        dl, embedder = accelerator.prepare( dl, embedder )  # prepare handles distributing things among GPUs

        with torch.no_grad():
            for i, data_dict in enumerate(dl):  # load audio files
                audio_sample_batch, filename_batch = data_dict['inputs'], data_dict['filename']
                newdir_already = False
                if not newdir_already: 
                    p = Path( filename_batch[0] )
                    dir_already = True
                    newdir = f"{p.parents[0]}_emb_{model_choice}"
                    hprint(f"creating new directory = {newdir}")
                    makedir(newdir) 
                    newdir_already = True
                # cut audio samples into chunks spaced out by hops, and loop over them
                hop_samples = int(args.hop_size * args.sample_size)
                hop_starts = np.arange(0, args.sample_size, hop_samples)
                if args.max_hops <= 0:  
                    hop_starts = hop_starts[:min(len(hop_starts), args.max_hops)]
                if args.sample_size - hop_starts[-1] < args.hop_size: # judgement call: let's not zero-pad on the very end, rather just don't do the last hop
                    hop_starts = hop_starts[:-1]
                for h_ind, hop_loc in enumerate(hop_starts):               # proceed through audio file batch via chunks, skipping by hop_samples each time
                    chunk = audio_sample_batch[:,:,hop_loc:hop_loc+hop_samples]
                    audio = chunk 
                    
                    #print(f"{ddps} i = {i}/{len(real_dataset)}, filename = {filename_batch[0]}")
                    audio = audio.to(device)


                    if model_choice == 'clap': 
                        while len(audio.shape) < 3: 
                            audio = audio.unsqueeze(0) # add batch and/or channel dims 
                        embeddings = accelerator.unwrap_model(embedder).get_audio_embedding_from_data(audio.mean(dim=1).to(device), use_tensor=True).to(audio.dtype)

                    elif model_choice == "vggish":
                        audio = torch.mean(audio, dim=1)   # vggish requries we convert to mono
                        embeddings = []                    # ...whoa, vggish can't even handle batches?  we have to pass 'em through singly?
                        for bi, waveform in enumerate(audio): 
                            e =  accelerator.unwrap_model(embedder.to(torch.device("cpu"))).forward(waveform.cpu().numpy(), emb_sample_rate)
                            embeddings.append(e) 
                        embeddings = torch.cat(embeddings, dim=0)

                    elif model_choice == "pann": 
                        audio = torch.mean(audio, dim=1)  # mono only.  todo:  keepdim=True ?
                        out = embedder.forward(audio, None)
                        embeddings = out['embedding'].data

                    elif model_choice == "openl3" and OPENL3_VERSION == "hugo":
                        ##audio = torch.mean(audio, dim=1)  # mono only.
                        embeddings = []
                        for bi, waveform in enumerate( audio.cpu().numpy() ): # no batch processing, expects numpy 
                            e = torchopenl3.embed(model=embedder, 
                                audio=waveform, # shape sould be (channels, samples)
                                sample_rate=emb_sample_rate, # sample rate of input file
                                hop_size=1,  device=device)
                            if debug: hprint(f"bi = {bi}, waveform.shape = {waveform.shape},  e.shape = {e.shape}") 
                            embeddings.append(torch.tensor(e))
                        embeddings = torch.cat(embeddings, dim=0)

                    elif model_choice == "openl3" and OPENL3_VERSION == "turian":
                        # Note: turian's can/will do multiple time-stamped embeddings if the sample_size is long enough. but our chunks/hops precludes this

                        #not needed, turns out: audio = renot needed, turns out: arrange(audio, 'b c s -> b s c')       # this torchopen3 expects channels-first ordering
                        embeddings, timestamps = torchopenl3.get_audio_embedding(audio, emb_sample_rate, model=embedder)
                        embeddings = torch.squeeze(embeddings, 1)        # get rid of any spurious dimensions of 1 in middle position 

                    else:
                        raise ValueError(f"Unknown model_choice = {model_choice}")

                    hprint(f"embeddings.shape = {embeddings.shape}")
                    # TODO: for now we'll just dump each batch on each proc to its own file; this could be improved
                    outfilename = f"{newdir}/emb_p{local_rank}_b{i}_h{h_ind}.pt"
                    hprint(f"{ddps} Saving embeddings to {outfilename}")
                    torch.save(embeddings.cpu().detach(), outfilename)
                    
        del embedder
        torch.cuda.empty_cache()
        # end loop over various embedders
    return        

# %% ../nbs/02_fad_embed.ipynb 11
def main(): 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('embed_model', help='choice of embedding model(s): clap | vggish | pann | openl3 | all ', default='clap')
    parser.add_argument('real_path', help='Path of files of real audio', default='real/')
    parser.add_argument('--fake_path', help='Path of files of fake audio', default='fake/')
    parser.add_argument('--one_directory', type=bool, default=False, help='MAXIMUM Batch size for computing embeddings (may go smaller)')
    parser.add_argument('--batch_size', type=int, default=64, help='MAXIMUM Batch size for computing embeddings (may go smaller)')
    parser.add_argument('--sample_size', type=int, default=2**18, help='Number of audio samples to read from each audio file')
    parser.add_argument('--chunk_size', type=int, default=24000, help='Length of chunks (in audio samples) to embed')
    parser.add_argument('--hop_size', type=float, default=0.100, help='(approximate) time difference (in seconds) between each chunk')
    parser.add_argument('--max_hops', type=int, default=-1, help="Don't exceed this many hops/chunks/embeddings per audio file. <= 0 disables this.")
    parser.add_argument('--sr', type=int, default=48000, help='sample rate (will resample inputs at this rate)')
    parser.add_argument('--verbose', action='store_true',  help='Show notices of resampling when reading files')
    parser.add_argument('--debug', action='store_true',  help='Extra messages for debugging this program')

    args = parser.parse_args()
    if args.one_directory:
        embed_one_directory(args)
    else:
        embed(args)

# %% ../nbs/02_fad_embed.ipynb 12
if __name__ == '__main__' and "get_ipython" not in dir():
    main()
