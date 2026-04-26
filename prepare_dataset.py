import gin
import copy
import librosa
import lmdb
import torch
import numpy as np
import os
import json
from tqdm import tqdm
import pickle
import pretty_midi
import click

from platune.datasets.parsers import get_parser

from platune.datasets.audio_example import AudioExample
from platune.datasets.base import SimpleDataset

from platune.datasets.transforms import BasicPitchPytorch
from platune.datasets.audio_descriptors import compute_all
from platune.datasets.process_attributes import process_midi_attributes, get_midi_notes, process_playing_technique


torch.set_grad_enabled(False)


DISCRETE_ATTRIBUTES = ['melody', 'melody_processed', 'pitch', 'pitch_processed', 'octave', 'octave_processed', 'onsets', 'dynamics', 'velocity', 'instrument']
CONTINUOUS_ATTRIBUTES = ['rms', 'loudness1s', 'integrated_loudness']


def load_embedding_model(emb_model_path, device):
    if emb_model_path == "music2latent":
        from music2latent import EncoderDecoder
        return EncoderDecoder(device=device)

    if emb_model_path == "codicodec":
        from codicodec import EncoderDecoder
        return EncoderDecoder(device=device)

    return torch.jit.load(emb_model_path).to(device)


def compute_bins(all_attr_continuous, nb_bins, continuous_descriptors):
    all_values = {}
    for i, c in enumerate(continuous_descriptors):
        data = all_attr_continuous[:, i, :].flatten()
        data.sort()
        index = np.linspace(0, len(data) - 1, nb_bins).astype(int)
        values = [data[i] for i in index]
        all_values[c] = values
    return all_values


def compute_attribute_statistics(data_path, n_bins=20,
                                 discrete_var=DISCRETE_ATTRIBUTES,
                                 continuous_var=CONTINUOUS_ATTRIBUTES):
    """Compute attribute statistics on an LMDB and write output files.

    Writes metadata_attributes.json, bins_values.pkl, lmdb_keys.pkl,
    skip_keys.pkl into *data_path*.
    """
    keys = ["z", "metadata"] + list(discrete_var) + list(continuous_var)

    dataset = SimpleDataset(path=data_path, keys=keys)

    lmdb_keys = []
    skip_keys = []
    min_max = {}
    discrete_var_count = {}
    all_attr_continuous = []

    for i in tqdm(range(len(dataset)), desc="Computing attribute stats"):
        data = dataset[i]
        attr_continuous = []

        for k, v in data.items():
            if k in ['z', 'metadata', 'midi', 'probs_list']:
                continue

            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()

            if np.isnan(v).any():
                current_key = dataset.keys[i]
                if current_key not in skip_keys:
                    print(f"Skipping example {str(current_key)} because found NaN values.")
                    skip_keys.append(current_key)
                    continue

            if k in discrete_var:
                values, count = np.unique(v, return_counts=True)
                if k not in discrete_var_count:
                    discrete_var_count[k] = {
                        int(val): int(c) for val, c in zip(values, count)
                    }
                else:
                    for j in range(len(values)):
                        if int(values[j]) not in discrete_var_count[k]:
                            discrete_var_count[k][int(values[j])] = int(count[j])
                        else:
                            discrete_var_count[k][int(values[j])] += int(count[j])

            if k in continuous_var:
                attr_continuous.append(v)
                min_value = np.min(v)
                max_value = np.max(v)
                if k not in min_max:
                    min_max[k] = {'min': float(min_value), 'max': float(max_value)}
                else:
                    if min_value < min_max[k]['min']:
                        min_max[k]['min'] = float(min_value)
                    if max_value > min_max[k]['max']:
                        min_max[k]['max'] = float(max_value)

        if len(attr_continuous) > 0:
            attr_continuous = np.stack(attr_continuous)
            all_attr_continuous.append(attr_continuous)

    print(min_max)
    print(discrete_var_count)

    if len(all_attr_continuous) > 0:
        all_attr_continuous = np.stack(all_attr_continuous)
        bins_values = compute_bins(all_attr_continuous, n_bins, continuous_var)
    else:
        bins_values = {}

    metadata = {
        'data_path': data_path,
        'continuous_attr_min_max': min_max,
        'discrete_attr_var_count': discrete_var_count,
    }

    with open(os.path.join(data_path, 'metadata_attributes.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    with open(os.path.join(data_path, "bins_values.pkl"), "wb") as f:
        pickle.dump(bins_values, f)

    print(f"nb skipped examples with nans : {len(skip_keys)}")

    with open(os.path.join(data_path, "skip_keys.pkl"), "wb") as f:
        pickle.dump(skip_keys, f)

    lmdb_keys = [key for key in dataset.keys if key not in skip_keys]
    print(f"nb examples : {len(lmdb_keys)}")

    with open(os.path.join(data_path, "lmdb_keys.pkl"), "wb") as f:
        pickle.dump(lmdb_keys, f)


def normalize_signal(x: np.ndarray, max_gain_db: int = 30, gain_margin: float = 0.9):
    peak = np.max(abs(x))
    if peak == 0:
        return x
    log_peak = 20 * np.log10(peak)
    log_gain = min(max_gain_db, -log_peak)
    gain = 10**(log_gain / 20)
    return gain_margin * x * gain


def get_midi(midi_data: pretty_midi.PrettyMIDI, chunk_number, num_signal, sample_rate):
    do_silence_check = False

    length = num_signal / sample_rate
    tstart = chunk_number * num_signal / sample_rate
    tend = (chunk_number + 1) * num_signal / sample_rate

    if len(midi_data.instruments) == 0:
        do_silence_check = True
        midi_data = None
        return do_silence_check, midi_data
    
    out_notes = []
    for note in midi_data.instruments[0].notes:
        if note.end > tstart and note.start < tend:
            note.start = max(0, note.start - tstart)
            note.end = min(note.end - tstart, length)
            out_notes.append(note)

    if len(out_notes) == 0:
        do_silence_check = True
        midi_data = None
        return do_silence_check, midi_data

    midi_data.instruments[0].notes = out_notes
    midi_data.adjust_times([0, length], [0, length])

    return do_silence_check, midi_data


@click.command()
@click.option('-i', '--input_path', default="", help='folder with the audio files')
@click.option('-o', '--output_path', default="", help='lmdb save path')
@click.option('-s', '--db_size', default=10, help='Max Size of lmdb database')
@click.option('-p', '--parser_name', default="simple_parser", help='parser function to obtain the list audio files and metadatas')
@click.option('-c', '--config', default=None, help='Name of the gin configuration file to use')
@click.option('-m', '--emb_model_path', default="", help='code to use. Either "codicodec" or a torchscript path')
@click.option('--gpu', default=0, help='device for basic-pitch and codec (-1 for cpu)')
@click.option('-n', '--num_signal', default=131_072, help='chunk sizes')
@click.option('--sr', default=44_100, help='sample rate')
@click.option('-b', '--batch_size', default=32, help='Batch size (for embedding model inference)')
@click.option('--normalize', is_flag=True, help='Normalize audio waveform (done once per file and not chunks ! )')
@click.option('--cut_silences', is_flag=True, help='Remove silence chunks')
@click.option('--save_waveform', is_flag=True, help='Wether to save the waveform in the lmdb')
@click.option('-l', '--descriptors_list', multiple=True, default=[], help='list of audio descriptors to compute on audio chunks')
@click.option('--use_basic_pitch', is_flag=True, help='use basic pitch for midi extraction from audio')
@click.option('--midi_attributes', is_flag=True, help='Whether to compute and save midi attributes on the midi chunks')
@click.option('-v', '--version', default="v1", help="data processing version")
@click.option('--val_num_chunks', default=1000, type=int, help='Number of entries to copy into a validation LMDB (0 to skip, default 1000)')
@click.option('--n_bins', default=20, type=int, help='Number of bins to quantize continuous attributes (for attribute statistics)')
def main(
        input_path, 
        output_path, 
        db_size, 
        parser_name,
        config,
        emb_model_path, 
        gpu, 
        num_signal, 
        sr, 
        batch_size, 
        normalize,     
        cut_silences, 
        save_waveform, 
        descriptors_list, 
        use_basic_pitch, 
        midi_attributes,
        version,
        val_num_chunks,
        n_bins,
    ):
    
    # cast args to python type
    descriptors_list = list(descriptors_list)

    # load pretrained codec
    device = "cuda:" + str(gpu) if torch.cuda.is_available() and gpu >= 0 else "mps" if torch.mps.is_available() and gpu >= 0 else "cpu"

    # logging
    print("-"*60)
    print(" "*20 + "Config:")
    print("-"*60)
    print("Audios path : ", input_path)
    print("parser : ", parser_name)
    print("Output path : ", output_path)
    print("gin config : ", config)
    print("signal length : ", num_signal)
    print("sample rate : ", sr)
    print("pretrained codec : ", emb_model_path)
    print("device : ", device)
    print("-"*60)
    print(" "*20 + "Attributes:")
    print("-"*60)
    print("Audio descriptors to be computed : ", descriptors_list)
    print("Using BasicPitch to extract MIDI data : ", use_basic_pitch)
    print("Processing MIDI attributes (melody, instrument) : ", midi_attributes)
    print("Validation num chunks : ", val_num_chunks)
    print("-"*60)


    emb_model = load_embedding_model(emb_model_path, device)
    z_length = None

    # load config for processing attributes
    if config is not None:
        shared_config_file = os.path.join(
            os.path.dirname(__file__), "platune", "configs", "datasets", f"{config}.gin")

        if os.path.exists(shared_config_file):
            config_file = shared_config_file
        else:
            raise FileNotFoundError(
                f"Config {config}.gin not found in {shared_config_file} "
            )
        print('loading config file : ', config_file)
        gin.parse_config_files_and_bindings([config_file],[])

        if midi_attributes:
            with gin.unlock_config():
                if version == 'v1':
                    gin.bind_parameter("process_attributes.process_midi_attributes.num_signal", num_signal)
                elif version == 'v2':
                    gin.bind_parameter("process_attributes.process_midi_attributesv2.num_signal", num_signal)
                else:
                    raise ValueError(f'version {version} does not exist!')
                gin.bind_parameter("process_attributes.process_playing_technique.num_signal", num_signal)
                
        if len(descriptors_list) > 0:
            with gin.unlock_config():
                gin.bind_parameter("compute_all.descriptors", descriptors_list)

    # initialize lmdb database
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(
        output_path,
        map_size=db_size * 1024**3,
        map_async=True,
        writemap=True,
        readahead=False,
    )

    # parse audio files
    audio_files, metadatas = get_parser(parser_name)(input_path)

    # loader BasicPitchPytorch
    if use_basic_pitch:
        BP = BasicPitchPytorch(sr=sr, device=device)

    # init
    chunks_buffer, metadatas_buffer, midis = [], [], []
    chunk_indices_buffer = []
    cur_index = 0
    skip_examples_inf_loud = 0

    # process loop
    for i, (file, metadata) in enumerate(zip(tqdm(audio_files), metadatas)):
        

        # load audio
        try:
            audio = librosa.load(file, sr=sr)[0]  # only mono not stereo
        except:
            print("error loading file : ", file)
            continue
        audio = audio.squeeze()

        if audio.shape[-1] == 0:
            print("Empty file")
            continue

        if normalize:
            audio = normalize_signal(audio)

        # check audio length to ensure power of 2 (ie. multiple of num signal)
        length = audio.shape[-1]
        if length < num_signal:
            print(f'Warning - skip audio {file} because audio length is too short : {length} < num_signal={num_signal}')
            continue
        else:
            if length % num_signal < num_signal // 2:
                length_crop = (length // num_signal) * num_signal
                audio = audio[:length_crop]
            else:
                # pad audio signal to a power of 2 (num_signal)
                audio = np.pad(audio, (0, num_signal - audio.shape[-1] % num_signal))
        
        # process MIDI data
        if use_basic_pitch:
            midi_data = BP(audio)
        elif metadata.get("midi_file_object") is not None:
            midi_data = copy.deepcopy(metadata["midi_file_object"])
        else:
            midi_data = None
        
        # reshape audio signal into chunks
        chunks = audio.reshape(-1, num_signal)
        chunk_index = 0

        # get the number of latent frames computed by the codec
        if i == 0 and emb_model is not None and z_length is None:
            ex_chunk_torch = torch.from_numpy(chunks[0]).to(device)
            if emb_model_path == "music2latent":
                ex_chunk_torch = ex_chunk_torch.reshape(-1, num_signal)
            else:
                ex_chunk_torch = ex_chunk_torch.reshape(-1, 1, num_signal)

            z_ex = emb_model.encode(ex_chunk_torch) if emb_model_path != "codicodec" else emb_model.encode(ex_chunk_torch, fix_batch_size=True)
            if emb_model_path == "codicodec":  # [B, T, lpt, 64] -> [B, 64, T*lpt]
                z_ex = z_ex.reshape(z_ex.shape[0], z_ex.shape[1]*z_ex.shape[2], z_ex.shape[3]).transpose(1, 2)
            z_length = z_ex.shape[-1]

            if config is not None:
                if midi_attributes:
                    if version == 'v1':
                        with gin.unlock_config():
                            gin.bind_parameter("process_attributes.process_midi_attributes.z_length", z_length)
                    with gin.unlock_config():
                        gin.bind_parameter("process_attributes.process_playing_technique.z_length", z_length)

                if len(descriptors_list) > 0:
                    with gin.unlock_config():
                        gin.bind_parameter("audio_descriptors.compute_all.z_length", z_length)

        empty_midis_indices = []
        for j, chunk in enumerate(chunks):

            # Chunk the midi
            if midi_data is not None:
                silence_test, midi = get_midi(
                    copy.deepcopy(midi_data), 
                    chunk_number=chunk_index,
                    num_signal=num_signal,
                    sample_rate=sr,
                )
                if midi is None:
                    empty_midis_indices.append(j)
            else:
                midi = None
                silence_test = np.max(abs(chunk)) < 0.05 if cut_silences else False

            # don't process buffer if empty slice
            if silence_test:
                chunk_index += 1
                continue



            midis.append(midi)
            chunks_buffer.append(chunk)
            metadatas_buffer.append(metadata)
            chunk_indices_buffer.append(chunk_index)

            if len(chunks_buffer) == batch_size or (j == len(chunks) - 1 and i == len(audio_files) - 1):

                # get latent representation from pretrained codec
                if emb_model is not None:
                    chunks_buffer_torch = torch.from_numpy(np.stack(chunks_buffer)).to(device)
                    if emb_model_path == "music2latent":
                        chunks_buffer_torch = chunks_buffer_torch.reshape(-1, num_signal)
                    else:
                        chunks_buffer_torch = chunks_buffer_torch.reshape(-1, 1, num_signal)
                    z = emb_model.encode(chunks_buffer_torch) if emb_model_path != "codicodec" else emb_model.encode(chunks_buffer_torch, fix_batch_size=True)
                    if emb_model_path == "codicodec":  # [B, T, lpt, 64] -> [B, 64, T*lpt]
                        z = z.reshape(z.shape[0], z.shape[1]*z.shape[2], z.shape[3]).transpose(1, 2)
                else:
                    z = [None] * len(chunks_buffer)
                
                for i, (audio_array, z_array, midi, cur_metadata, buf_chunk_idx) in enumerate(zip(chunks_buffer, z, midis, metadatas_buffer, chunk_indices_buffer)):
                    
                    assert audio_array.shape[-1] == num_signal
                    
                    if i in empty_midis_indices:
                        # do not store chunk if you require midi data but midi is None
                        continue

                    # compute audio descriptors
                    feat = compute_all(audio_array) if len(descriptors_list) > 0 else None

                    if 'integrated_loudness' in feat and np.any(feat['integrated_loudness'] == float("-inf")):
                        skip_examples_inf_loud += 1
                        continue

                    if 'loudness1s' in feat and np.any(feat['loudness1s'] == float("-inf")):
                        skip_examples_inf_loud += 1
                        continue

                    # create instance of our lmdb database
                    key = f"{cur_index:08d}"
                    ae = AudioExample()

                    # save chunk audio waveform to lmdb database
                    if save_waveform:
                        if type(audio_array) == torch.Tensor:
                            audio_array = audio_array.cpu().numpy()
                        audio_array = (audio_array * (2**15 - 1)).astype(np.int16)
                        ae.put_array("waveform", audio_array, dtype=np.int16)

                    # save latent representation
                    if z_array is not None:
                        ae.put_array("z", z_array.cpu().numpy(), dtype=np.float32)

                    # save metadata
                    cur_metadata["chunk_index"] = buf_chunk_idx
                    cur_metadata["key"] = key
                    # Strip non-serializable fields before JSON encoding
                    serializable_metadata = {k: v for k, v in cur_metadata.items()
                                             if k not in ("midi_file_object", "playing_techniques")}
                    ae.put_metadata(serializable_metadata)

                    # save MIDI data
                    if midi is not None:
                        ae.put_buffer(key="midi", b=pickle.dumps(midi), shape=None)
                        
                        if midi_attributes:
                            if version == 'v1':
                                attr_midi = process_midi_attributes(x=midi, instrument_val=cur_metadata["instrument"])
                            elif version == 'v2':
                                pitch, onset, offset, _ = get_midi_notes(midi)
                                attr_midi = process_midi_attributesv2(pitch, onset, offset, instrument_val=cur_metadata["instrument"])

                            for k, v in attr_midi.items():
                                ae.put_array(k, v, dtype=np.int32)

                    # save playing technique attribute (CCOM)
                    if cur_metadata.get("playing_techniques") is not None and midi_attributes:
                        pt_data = cur_metadata["playing_techniques"]
                        pt_attr = process_playing_technique(
                            playing_techniques=pt_data,
                            chunk_number=buf_chunk_idx,
                        )
                        ae.put_array("playing_technique", pt_attr, dtype=np.int32)

                    # save audio descriptors
                    if feat is not None:
                        for k, v in feat.items():
                            ae.put_array(k, v, dtype=np.float32)

                    # save AudioExample instance to lmdb database
                    with env.begin(write=True) as txn:
                        txn.put(key.encode(), bytes(ae))
                    cur_index += 1

                chunks_buffer, midis, metadatas_buffer, chunk_indices_buffer = [], [], [], []
            chunk_index += 1

    print("nb of audio chunks skipped because found -inf loudness: ", skip_examples_inf_loud)
    print(f"Total chunks stored: {cur_index}")

    # Shuffle all entries in the LMDB so sequential reads are in random order
    if cur_index > 0:
        print(f"\nShuffling {cur_index} entries in LMDB...")
        rng_shuffle = np.random.default_rng(42)
        perm = rng_shuffle.permutation(cur_index)

        # Build shuffled LMDB in a temp directory, then swap
        shuffled_path = output_path.rstrip('/') + "_shuffled"
        os.makedirs(shuffled_path, exist_ok=True)
        shuffled_env = lmdb.open(
            shuffled_path,
            map_size=db_size * 1024**3,
            map_async=True,
            writemap=True,
            readahead=False,
        )

        with env.begin() as src_txn:
            for new_idx, orig_idx in enumerate(tqdm(perm, desc="Shuffling")):
                orig_key = f"{int(orig_idx):08d}".encode()
                data = src_txn.get(orig_key)
                if data is not None:
                    new_key = f"{new_idx:08d}".encode()
                    with shuffled_env.begin(write=True) as dst_txn:
                        dst_txn.put(new_key, data)

        shuffled_env.close()
        env.close()

        # Replace original with shuffled
        import shutil
        shutil.rmtree(output_path)
        os.rename(shuffled_path, output_path)

        # Re-open for downstream val split
        env = lmdb.open(
            output_path,
            map_size=db_size * 1024**3,
            map_async=True,
            writemap=True,
            readahead=False,
        )
        print("  Shuffle complete.")

    if config is not None:
        # save config if processing attributes
        # NB: done at the end because otherwise the classes/functions that have not been
        # instantiated yet won't appear
        with open(os.path.join(output_path, "config.gin"),"w") as config_out:
            config_out.write(gin.operative_config_str())

    # ----------------------------------------------------------------
    # Create validation LMDB (non-overlapping subset removed from training)
    # ----------------------------------------------------------------
    if val_num_chunks > 0 and cur_index > 0:
        val_output_path = output_path.rstrip('/') + "_val"
        n_val = min(val_num_chunks, cur_index)
        rng = np.random.default_rng(42)
        val_indices = rng.choice(cur_index, size=n_val, replace=False)
        val_indices.sort()

        print(f"\nCreating validation LMDB at {val_output_path}")
        print(f"  Moving {n_val} / {cur_index} entries (removed from training LMDB)")

        os.makedirs(val_output_path, exist_ok=True)
        val_env = lmdb.open(
            val_output_path,
            map_size=max(1, db_size // 4) * 1024**3,
            map_async=True,
            writemap=True,
            readahead=False,
        )

        # Copy selected entries to val LMDB, then delete from training LMDB
        with env.begin() as read_txn:
            for new_idx, orig_idx in enumerate(val_indices):
                orig_key = f"{orig_idx:08d}".encode()
                data = read_txn.get(orig_key)
                if data is not None:
                    val_key = f"{new_idx:08d}".encode()
                    with val_env.begin(write=True) as val_txn:
                        val_txn.put(val_key, data)

        with env.begin(write=True) as del_txn:
            for orig_idx in val_indices:
                orig_key = f"{orig_idx:08d}".encode()
                del_txn.delete(orig_key)

        val_env.close()

        # Copy config.gin into validation LMDB directory
        if config is not None:
            import shutil
            src_config = os.path.join(output_path, "config.gin")
            if os.path.exists(src_config):
                shutil.copy2(src_config, os.path.join(val_output_path, "config.gin"))

        print(f"  Validation LMDB created: {n_val} entries (training LMDB now has {cur_index - n_val} entries)")
    elif val_num_chunks > 0:
        print("\nSkipping validation LMDB (no entries stored).")

    env.close()

    # ----------------------------------------------------------------
    # Compute attribute statistics (formerly compute_min_max_dataset.py)
    # ----------------------------------------------------------------
    if cur_index > 0:
        print("\n" + "-"*60)
        print("Computing attribute statistics on training LMDB...")
        print("-"*60)
        compute_attribute_statistics(output_path, n_bins=n_bins)


if __name__ == '__main__':
    main()
