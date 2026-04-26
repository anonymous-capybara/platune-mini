import os
import pathlib
import csv
import yaml
import random
from tqdm import tqdm

from typing import Iterable, Sequence


def flatten(iterator: Iterable):
    for elm in iterator:
        for sub_elm in elm:
            yield sub_elm


def search_for_audios(
    path_list: Sequence[str],
    extensions: Sequence[str] = [
        "wav", "opus", "mp3", "aac", "flac", "aif", "ogg"
    ],
):
    paths = map(pathlib.Path, path_list)
    audios = []
    for p in paths:
        for ext in extensions:
            audios.append(p.rglob(f"*.{ext}"))
    audios = flatten(audios)
    audios = [str(a) for a in audios if 'MACOS' not in str(a)]
    audios = [str(a) for a in audios if '._' not in str(a)]
    
    return audios


def simple_parser(audio_folder, filters=None):
    audios = search_for_audios([audio_folder])
    audios = map(str, audios)
    audios = map(os.path.abspath, audios)
    audios = [*audios]

    if filters is not None:
        audios = [
            a for a in audios if any([s.lower() in a.lower() for s in filters])
        ]

    random.shuffle(audios)

    metadatas = [{
        "path": audio,
        "instrument": 'unknown',
    } for audio in audios]
    print(len(audios), " files found")

    return audios, metadatas

def gtsinger_parser(audio_folder, filters=None):
    audios = search_for_audios([audio_folder])
    audios = map(str, audios)
    audios = map(os.path.abspath, audios)
    audios = [*audios]

    if filters is not None:
        audios = [
            a for a in audios if any([s.lower() in a.lower() for s in filters])
        ]

    random.shuffle(audios)

    metadatas = [{
        "path": audio,
        "instrument": 'female singer',
    } for audio in audios]
    
    print(len(audios), " files found")

    return audios, metadatas


def maestro_parser(main_folder, filters=None):

    # audios = os.listdir(os.path.join(main_folder, "audio"))
    audios = search_for_audios([main_folder])
    metadatas = []
    for a in audios:
        midi_file = a[:-4] + ".midi"
        # audio_file = os.path.join(main_folder, "audio", a)
        metadatas.append({"path": str(a), "midi_file": midi_file})
    
    # audios = [os.path.join(main_folder, "audio", f) for f in audios]

    print("sanity check: ", audios[0], metadatas[0])

    return audios, metadatas


def synthetic_parser(main_folder, filters=None):

    audios = os.listdir(os.path.join(main_folder, "audio"))
    metadatas = []
    for a in audios:
        midi_file = os.path.join(main_folder, "midi", a.split("_")[0]) + ".mid"
        instrument = "_".join(a.split("_")[1:])[:-4]
        metadatas.append({"midi_file": midi_file, "instrument": instrument})

    audios = [os.path.join(main_folder, "audio", f) for f in audios]

    print("sanity check: ", audios[0], metadatas[0])

    return audios, metadatas


def slakh_parser(audio_folder, ban_list=[]):
    tracks = [
        os.path.join(audio_folder, "train", subfolder)
        for subfolder in os.listdir(os.path.join(audio_folder, "train")) if '._' not in subfolder
    ] + [
        os.path.join(audio_folder, "test", subfolder)
        for subfolder in os.listdir(os.path.join(audio_folder, "test")) if '._' not in subfolder
    ] + [
        os.path.join(audio_folder, "validation", subfolder)
        for subfolder in os.listdir(os.path.join(audio_folder, "validation")) if '._' not in subfolder
    ]
    meta = tracks[0] + "/metadata.yaml"
    print("sanity check metadata file : ", meta)
    ban_list = [
        "Chromatic Percussion",
        "Drums",
        "Percussive",
        "Sound Effects",
        "Sound effects",
    ]  # , "Ethnic", "Organ", "Synth Pad", "Synth Lead", "Reed"

    #get_list = ["Strings", "Strings (continued)"]
    instr = []
    stem_list = []
    metadata = []
    total_stems = 0

    for trackfolder in tqdm(tracks):
        meta = trackfolder + "/metadata.yaml"
        with open(meta, "r") as file:
            d = yaml.safe_load(file)
        for k, stem in d["stems"].items():
            if stem["inst_class"] not in ban_list:
                stem_list.append(trackfolder + "/stems/" + k + ".flac")
                instr.append(stem["midi_program_name"])
                metadata.append(stem)
            total_stems += 1

    print(set(instr), "instruments remaining")
    print(total_stems, "stems in total")
    print(len(stem_list), "stems retained")
    audios = stem_list
    metadatas = [{
        "path": audio,
        "instrument": inst
    } for audio, inst in zip(audios, instr)]
    return audios, metadatas


def get_urmp_midi_file_path(midi_folder, audio_path):
    _, n, inst, audio_idx, audio_name = audio_path.stem.split('_')

    midi_files_candidates = list(pathlib.Path(midi_folder).rglob(f'{audio_idx}_{audio_name}_{inst}*.mid'))

    if len(midi_files_candidates) > 1:
        
        if (inst == 'vn' and n == '1') or (inst == 'fl' and n == '1') or (inst == 'tpt' and n == '1') or (inst == 'sax' and n == '1') or (inst == 'va' and n == '3'):
            midi_filepath = [str(p) for p in midi_files_candidates if f"{inst}.mid" in str(p)][0]
        
        elif (inst == 'vn' and n == '2') or (inst == 'fl' and n == '2') or (inst == 'tpt' and n == '2') or (inst == 'sax' and n == '2') or (inst == 'va' and n == '4'):
            midi_filepath = [str(p) for p in midi_files_candidates if f"{inst}_" in str(p)][0]
        
        else:
            raise ValueError(f'Could not find midi file for file : {str(audio_path)}')
    
    else:
        midi_filepath = str(midi_files_candidates[0])
    return midi_filepath


def urmp_parser(audio_folder):
    audios = search_for_audios([audio_folder])

    # instruments = {
    #     'vn': 'Violin',
    #     'va': 'Viola',
    #     'vc': 'Cello',
    #     'db': 'Double Basse',
    #     'fl': 'Flute',
    #     'ob': 'Oboe',
    #     'cl': 'Clarinet',
    #     'sax': 'Saxophone',
    #     'bn': 'Bassoon',
    #     'tpt': 'Trumpet',
    #     'hn': 'Horn',
    #     'tbn': 'Trombone',
    #     'tba': 'Tuba',
    # }
    instruments = {
        'vn': 'violin',
        'fl': 'flute',
        'cl': 'clarinet',
        'tpt': 'trumpet',
    }

    metadata = []
    for audio in tqdm(audios):
        inst = pathlib.Path(audio).stem.split("_")[2]
        if inst not in instruments:
            print(f"Warning: instrument {inst} not recognized, skipping file {audio}")
            continue
        inst_name = instruments[inst]
        data = {
            "path": audio,
            "instrument": inst_name,
        }
        metadata.append(data)

    print(len(audios), " files found")
    print(f"selected files : {len(metadata)} / {len(audios)}")
    print("sanity check: ", audios[0], metadata[0])

    return audios, metadata


def medley_solos_mono_parser(audio_folder):
    audios = search_for_audios([audio_folder])

    metadata_filepath = os.path.join(
        str(pathlib.Path(audios[0]).parent.parent),
        "Medley-solos-DB_metadata.csv")

    # all_inst_list = [
    #     'clarinet',
    #     'distorted electric guitar',
    #     'female singer',
    #     'flute',
    #     'piano',
    #     'tenor saxophone',
    #     'trumpet',
    #     'violin'
    # ]

    filter_ployphonic_instruments = ['distorted electric guitar', 'piano']

    raw_metadata = {}
    with open(metadata_filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_metadata[row['uuid4']] = {'instrument': row['instrument']}

    metadata = []
    audios_filtered = []

    for audio in tqdm(audios):

        uuid = pathlib.Path(audio).stem.split("_")[-1]

        inst_name = raw_metadata[uuid]['instrument']

        if inst_name not in filter_ployphonic_instruments:
            audios_filtered.append(audio)

            data = {
                "path": audio,
                "instrument": inst_name,
            }
            metadata.append(data)

    print(len(audios), " files found")
    print(f"selected files : {len(audios_filtered)} / {len(audios)}")
    print("sanity check: ", audios_filtered[0], metadata[0])

    return audios_filtered, metadata


def ccom_parser(audio_folder):
    """Parser for the CCOM Chinese instrument dataset.

    Expected layout::

        audio_folder/<Instrument>/<Piece>/
            <Piece>.wav
            <Piece>-PT.csv   (onset, f0, duration, PT, PT1-1, PT1-2, PT1-3)

    Builds an in-memory PrettyMIDI from the PT.csv (f0 -> MIDI note) and
    collects per-note PT1-1 playing-technique labels.
    """
    import pretty_midi
    import math

    audios = []
    metadatas = []

    instrument_dirs = sorted([
        d for d in os.listdir(audio_folder)
        if os.path.isdir(os.path.join(audio_folder, d)) and not d.startswith('.')
    ])

    for instrument_name in instrument_dirs:
        instrument_path = os.path.join(audio_folder, instrument_name)
        piece_dirs = sorted([
            d for d in os.listdir(instrument_path)
            if os.path.isdir(os.path.join(instrument_path, d)) and not d.startswith('.')
        ])

        for piece_name in piece_dirs:
            piece_path = os.path.join(instrument_path, piece_name)

            wav_file = os.path.join(piece_path, f"{piece_name}.wav")
            pt_csv_file = os.path.join(piece_path, f"{piece_name}-PT.csv")

            if not os.path.isfile(wav_file):
                print(f"Warning: no .wav found at {wav_file}, skipping")
                continue
            if not os.path.isfile(pt_csv_file):
                print(f"Warning: no -PT.csv found at {pt_csv_file}, skipping")
                continue

            # Read PT.csv and build PrettyMIDI + playing technique list
            midi_obj = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0)
            playing_techniques = []  # list of (onset, duration, PT1-1) tuples

            with open(pt_csv_file, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    onset = float(row['onset'])
                    f0 = float(row['f0'])
                    duration = float(row['duration'])

                    # Convert f0 (Hz) to MIDI note number
                    if f0 <= 0:
                        continue
                    midi_note = int(round(69 + 12 * math.log2(f0 / 440.0)))
                    midi_note = max(0, min(127, midi_note))

                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=midi_note,
                        start=onset,
                        end=onset + duration,
                    )
                    inst.notes.append(note)

                    pt_label = row.get('PT1-1', '').strip()
                    playing_techniques.append((onset, duration, pt_label))

            midi_obj.instruments.append(inst)

            audios.append(os.path.abspath(wav_file))
            metadatas.append({
                "path": os.path.abspath(wav_file),
                "instrument": instrument_name,
                "midi_file_object": midi_obj,
                "playing_techniques": playing_techniques,
            })

    combined = list(zip(audios, metadatas))
    random.shuffle(combined)
    if combined:
        audios, metadatas = zip(*combined)
        audios, metadatas = list(audios), list(metadatas)

    print(f"{len(audios)} files found")
    if audios:
        print(f"sanity check: {audios[0]}, {metadatas[0]['instrument']}")

    return audios, metadatas


def get_parser(parser_name):
    if parser_name == "simple_parser":
        parser = simple_parser
    elif parser_name == "synthetic_parser":
        parser = synthetic_parser
    elif parser_name == "urmp_parser":
        parser = urmp_parser
    elif parser_name == "medley_solos_mono_parser":
        parser = medley_solos_mono_parser
    elif parser_name == "maestro_parser":
        parser = maestro_parser
    elif parser_name == "gtsinger_parser":
        parser = gtsinger_parser
    elif parser_name == "slakh_parser":
        parser = slakh_parser
    elif parser_name == "ccom_parser":
        parser = ccom_parser
    else:
        raise NotImplementedError(f'No parser method named : {parser_name}.')
    return parser
