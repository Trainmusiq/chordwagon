"""
Prototipo mínimo de chordwagon — corre el modelo ChordNet (ISMIR2019 Large-Vocabulary
Chord Recognition, music-x-lab, MIT) sobre un archivo real y arma la progresión con timestamps.

Reimplementación standalone (sin el framework "mir" de terceros) de la arquitectura descrita
en chordnet_ismir_naive.py del repo original, cargando los pesos pre-entrenados (.sdict)
tal cual, para validar en este equipo: (a) que el modelo es portable (solo Conv2d/InstanceNorm2d/
MaxPool2d/LSTM/Linear, sin ops custom), (b) su tamaño real, (c) su velocidad de inferencia CPU
como proxy razonable de factibilidad ONNX Runtime Web, y (d) la calidad de las etiquetas contra
un tema real.

Simplificaciones deliberadas respecto del pipeline original (aceptables para un prototipo
sin UI, no para el producto final — quedan documentadas en especificacion.md):
  - 1 solo fold (s0) en vez del ensemble de 5.
  - Sin decodificación HMM (extractors/xhmm_ismir.py): se usa argmax por frame + agrupación
    por run-length con un filtro de mediana corto, que alcanza para validar la calidad de
    las etiquetas de acordes completos (séptimas, sus, dim).
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

SPEC_DIM = 252
SHIFT_HIGH = 6
SHIFT_STEP = 3
SR = 22050
HOP_LENGTH = 512

TRIAD_LIMIT = 6
SEVENTH_LIMIT = 3
NINTH_LIMIT = 3
ELEVENTH_LIMIT = 2
THIRTEENTH_LIMIT = 2

BASS_SLICE_BEGIN = TRIAD_LIMIT * 12 + 1  # 73
BASS_DEL = BASS_SLICE_BEGIN + 12 + 1  # 86
SEVENTH_DEL = BASS_DEL + SEVENTH_LIMIT + 1  # 90
NINTH_DEL = SEVENTH_DEL + NINTH_LIMIT + 1  # 94
ELEVENTH_DEL = NINTH_DEL + ELEVENTH_LIMIT + 1  # 97
THIRTEENTH_DEL = ELEVENTH_DEL + THIRTEENTH_LIMIT + 1  # 100

NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
TRIAD_TYPE_NAMES = {1: 'maj', 2: 'min', 3: 'sus4', 4: 'sus2', 5: 'dim', 6: 'aug'}


class CNNFeatureExtractor(nn.Module):
    def norm_layer(self, channels):
        return nn.InstanceNorm2d(channels)

    def __init__(self):
        super().__init__()
        self.cdim1, self.cdim2, self.cdim3, self.cdim4 = 16, 32, 64, 80
        self.conv1a = nn.Conv2d(1, self.cdim1, (3, 3), padding=(1, 1))
        self.norm1a = self.norm_layer(self.cdim1)
        self.conv1b = nn.Conv2d(self.cdim1, self.cdim1, (3, 3), padding=(1, 1))
        self.norm1b = self.norm_layer(self.cdim1)
        self.conv1c = nn.Conv2d(self.cdim1, self.cdim1, (3, 3), padding=(1, 1))
        self.norm1c = self.norm_layer(self.cdim1)
        self.pool1 = nn.MaxPool2d((1, 3))
        self.conv2a = nn.Conv2d(self.cdim1, self.cdim2, (3, 3), padding=(1, 1))
        self.norm2a = self.norm_layer(self.cdim2)
        self.conv2b = nn.Conv2d(self.cdim2, self.cdim2, (3, 3), padding=(1, 1))
        self.norm2b = self.norm_layer(self.cdim2)
        self.conv2c = nn.Conv2d(self.cdim2, self.cdim2, (3, 3), padding=(1, 1))
        self.norm2c = self.norm_layer(self.cdim2)
        self.pool2 = nn.MaxPool2d((1, 3))
        self.conv3a = nn.Conv2d(self.cdim2, self.cdim3, (3, 3), padding=(1, 1))
        self.norm3a = self.norm_layer(self.cdim3)
        self.conv3b = nn.Conv2d(self.cdim3, self.cdim3, (3, 3), padding=(1, 1))
        self.norm3b = self.norm_layer(self.cdim3)
        self.pool3 = nn.MaxPool2d((1, 4))
        self.conv4a = nn.Conv2d(self.cdim3, self.cdim4, (3, 3), padding=(1, 0))
        self.norm4a = self.norm_layer(self.cdim4)
        self.conv4b = nn.Conv2d(self.cdim4, self.cdim4, (3, 3), padding=(1, 0))
        self.norm4b = self.norm_layer(self.cdim4)
        self.output_size = 3 * self.cdim4

    def forward(self, x):
        batch_size, seq_length = x.shape[0], x.shape[1]
        x = x.view((batch_size, 1, seq_length, SPEC_DIM))
        x = F.selu(self.norm1a(self.conv1a(x)))
        x = F.selu(self.norm1b(self.conv1b(x)))
        x = F.selu(self.norm1c(self.conv1c(x)))
        x = self.pool1(x)
        x = F.selu(self.norm2a(self.conv2a(x)))
        x = F.selu(self.norm2b(self.conv2b(x)))
        x = F.selu(self.norm2c(self.conv2c(x)))
        x = self.pool2(x)
        x = F.selu(self.norm3a(self.conv3a(x)))
        x = F.selu(self.norm3b(self.conv3b(x)))
        x = self.pool3(x)
        x = F.selu(self.norm4a(self.conv4a(x)))
        x = F.selu(self.norm4b(self.conv4b(x)))
        x = x.transpose(1, 2).contiguous().view((batch_size, seq_length, self.output_size))
        return x


class ChordNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_feature_block = CNNFeatureExtractor()
        # declarado en el original pero no usado en forward() — se mantiene para que
        # el state_dict cargue con strict=True.
        chord_limit_triad = TRIAD_LIMIT
        self.condition_linear = nn.Linear(self.audio_feature_block.output_size + 12 + chord_limit_triad + 12, 128)
        self.hidden_dim1 = 192
        self.lstm1 = nn.LSTM(
            input_size=self.audio_feature_block.output_size,
            hidden_size=self.hidden_dim1 // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.final_fc1 = nn.Linear(self.hidden_dim1, THIRTEENTH_DEL)

    def init_hidden(self, batch_size, hidden_dim):
        c_0 = torch.zeros(2, batch_size, hidden_dim // 2)
        h_0 = torch.zeros(2, batch_size, hidden_dim // 2)
        return (c_0, h_0)

    def forward(self, x):
        batch_size, seq_length = x.shape[0], x.shape[1]
        x = self.audio_feature_block(x)
        x1 = self.lstm1(x, self.init_hidden(batch_size, self.hidden_dim1))[0]
        x1 = self.final_fc1(x1).reshape((batch_size * seq_length, THIRTEENTH_DEL))
        return (
            x1[:, :BASS_SLICE_BEGIN],
            x1[:, BASS_SLICE_BEGIN:BASS_DEL],
            x1[:, BASS_DEL:SEVENTH_DEL],
            x1[:, SEVENTH_DEL:NINTH_DEL],
            x1[:, NINTH_DEL:ELEVENTH_DEL],
            x1[:, ELEVENTH_DEL:THIRTEENTH_DEL],
        )


def decode_frame(triad_id, seventh_id, ninth_id, eleventh_id, thirteenth_id):
    if triad_id == 0:
        return 'N'
    triad_type = (triad_id - 1) // 12 + 1
    root = (triad_id - 1) % 12
    root_name = NOTE_NAMES[root]
    tname = TRIAD_TYPE_NAMES.get(triad_type, '?')

    exts = []
    if seventh_id == 1:
        exts.append('maj7' if tname == 'maj' else '7')  # add_7 (major 7th interval)
    elif seventh_id == 2:
        exts.append('7')  # add_b7 (minor 7th interval, dominant/min7)
    elif seventh_id == 3:
        exts.append('dim7')  # add_bb7 (diminished 7th)
    if ninth_id == 1:
        exts.append('9')
    elif ninth_id == 2:
        exts.append('#9')
    elif ninth_id == 3:
        exts.append('b9')
    if eleventh_id == 1:
        exts.append('11')
    elif eleventh_id == 2:
        exts.append('#11')
    if thirteenth_id == 1:
        exts.append('13')
    elif thirteenth_id == 2:
        exts.append('b13')

    # Casos con nombre estándar de acorde
    if tname == 'dim' and seventh_id == 3:
        return f'{root_name}dim7'
    if tname == 'dim' and seventh_id == 2:
        return f'{root_name}m7b5'  # half-diminished
    if tname == 'maj' and seventh_id == 1 and len(exts) == 1:
        return f'{root_name}maj7'
    if tname == 'maj' and seventh_id == 2 and len(exts) == 1:
        return f'{root_name}7'
    if tname == 'min' and seventh_id in (1, 2) and len(exts) == 1:
        return f'{root_name}m7' if seventh_id == 2 else f'{root_name}m(maj7)'
    if tname in ('sus4', 'sus2'):
        base = f'{root_name}{tname}'
        return base + ('(' + ','.join(exts) + ')' if exts else '')
    if tname == 'aug':
        base = f'{root_name}aug'
        return base + ('(' + ','.join(exts) + ')' if exts else '')

    base = f'{root_name}{"" if tname == "maj" else "m" if tname == "min" else tname}'
    if exts:
        return base + '(' + ','.join(exts) + ')'
    return base


def main(audio_path, ckpt_path):
    print(f'Cargando checkpoint: {ckpt_path}')
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    net_sd = sd['net']

    model = ChordNet()
    model.load_state_dict(net_sd, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parámetros totales: {n_params:,} (~{n_params * 4 / 1e6:.2f} MB fp32)')

    print(f'Decodificando audio: {audio_path}')
    t0 = time.time()
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    duration = len(y) / sr
    print(f'  {duration:.1f}s @ {sr}Hz, decodificación en {time.time()-t0:.2f}s')

    print('Calculando CQT híbrido (bins_per_octave=36, n_bins=288, fmin=F#0, hop=512)...')
    t0 = time.time()
    cqt = librosa.hybrid_cqt(
        y, sr=sr, bins_per_octave=36, fmin=librosa.note_to_hz('F#0'),
        n_bins=288, tuning=None, hop_length=HOP_LENGTH,
    ).T
    cqt = np.abs(cqt).astype(np.float32)
    print(f'  CQT shape {cqt.shape} en {time.time()-t0:.2f}s')

    offset = SHIFT_HIGH * SHIFT_STEP
    x = cqt[:, offset:offset + SPEC_DIM]
    assert x.shape[1] == SPEC_DIM, x.shape

    print('Corriendo inferencia (CNN + BiLSTM, CPU, single-thread)...')
    t0 = time.time()
    with torch.no_grad():
        xt = torch.from_numpy(x).unsqueeze(0)  # (1, seq, SPEC_DIM)
        out = model(xt)
        probs = [F.softmax(o, dim=1).numpy() for o in out]
    infer_time = time.time() - t0
    print(f'  Inferencia: {infer_time:.2f}s para {x.shape[0]} frames ({infer_time/x.shape[0]*1000:.2f} ms/frame)')

    triad_ids = probs[0].argmax(axis=1)
    bass_ids = probs[1].argmax(axis=1)
    seventh_ids = probs[2].argmax(axis=1)
    ninth_ids = probs[3].argmax(axis=1)
    eleventh_ids = probs[4].argmax(axis=1)
    thirteenth_ids = probs[5].argmax(axis=1)

    # Filtro de mediana corto (gana estabilidad temporal sin HMM completo)
    from scipy.ndimage import median_filter
    win = 9
    triad_ids = median_filter(triad_ids, size=win, mode='nearest')
    seventh_ids = median_filter(seventh_ids, size=win, mode='nearest')
    ninth_ids = median_filter(ninth_ids, size=win, mode='nearest')
    eleventh_ids = median_filter(eleventh_ids, size=win, mode='nearest')
    thirteenth_ids = median_filter(thirteenth_ids, size=win, mode='nearest')

    labels = [
        decode_frame(triad_ids[i], seventh_ids[i], ninth_ids[i], eleventh_ids[i], thirteenth_ids[i])
        for i in range(len(triad_ids))
    ]

    frame_time = HOP_LENGTH / SR
    print('\n=== Progresión de acordes (run-length, min 0.5s) ===')
    prev_label = None
    seg_start = 0
    min_frames = max(1, int(0.5 / frame_time))
    segments = []
    for i, lab in enumerate(labels + [None]):
        if lab != prev_label:
            if prev_label is not None:
                segments.append((seg_start, i, prev_label))
            seg_start = i
            prev_label = lab
    # fusionar segmentos muy cortos con el vecino más largo (limpieza mínima)
    merged = []
    for seg in segments:
        if merged and (seg[1] - seg[0]) < min_frames:
            merged[-1] = (merged[-1][0], seg[1], merged[-1][2])
        else:
            merged.append(list(seg))
    for s, e, lab in merged:
        t_start = s * frame_time
        t_end = e * frame_time
        print(f'  {t_start:6.2f}s - {t_end:6.2f}s  {lab}')

    print(f'\nTotal frames: {len(labels)}, segmentos: {len(merged)}, duración: {duration:.1f}s')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
