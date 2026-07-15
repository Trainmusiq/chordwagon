# Benchmark del motor de acordes — 15 jul 2026

Sesión de apertura de chordwagon, pasos 1-2 del método (ver `docs/especificacion.md` y
`manual-continuidad.md` §4 del repo `trainmusiq/trainmusiq`). Resultado completo y decisión
razonada: `docs/especificacion.md` §3. Esto es el registro reproducible del experimento.

## Qué se evaluó

**ChordMini** (`ptnghia-j/ChordMiniApp`, MIT) resultó ser una aplicación (Next.js + backend
Flask/PyTorch), no un modelo en sí — sirve como *wrapper* de dos modelos de investigación
independientes:

- **BTC** (`jayg996/BTC-ISMIR19`, MIT): Transformer bidireccional + CRF, checkpoint ~12 MB.
- **ChordNet** (`music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition`, MIT): CNN + BiLSTM
  de una capa, "Chord Structure Decomposition" — descompone el acorde en tríada + bajo + 7ª +
  9ª + 11ª + 13ª como clasificadores independientes. Este es el que da los "301 labels"
  referenciados en el roadmap.

Se eligió **ChordNet** para el prototipo por ser sustancialmente más liviano (arquitectura sin
atención ni CRF) y por declarar explícitamente el vocabulario completo (tríadas + alteraciones)
como diseño central del paper, no como extensión.

## Cómo correr esto

Reproducible con Python 3.9+ en un entorno nuevo:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch numpy librosa soundfile onnx onnxruntime scipy

mkdir -p models
curl -sL -o models/chordnet_s0.sdict \
  "https://raw.githubusercontent.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition/master/cache_data/joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s0.best.sdict"

python run_chordnet.py /ruta/a/tu/audio.mp3 models/chordnet_s0.sdict
```

`run_chordnet.py` reimplementa la arquitectura de `chordnet_ismir_naive.py` del repo original
en un único archivo, sin depender del framework "mir" de terceros (que solo existe para
entrenamiento/gestión de datasets, no para inferencia). Carga el `.sdict` con
`torch.load(..., weights_only=True)` — nunca `weights_only=False` sobre un checkpoint de
terceros, que ejecutaría pickle arbitrario.

## Resultados medidos en este equipo (MacBook Pro M1 Max, 32GB, CPU-only)

| Métrica | Valor |
|---|---|
| Parámetros totales | 500.772 (~2,0 MB fp32) |
| Export a ONNX (opset 17) | ✓ sin errores, 1,9 MB |
| Ops usadas en el grafo ONNX | `Conv`, `InstanceNormalization`, `MaxPool`, `Selu`, `LSTM`, `MatMul`, `Add`, `Mul`, `Concat`, `Slice`, `Reshape`, `Transpose`, `Gather`, `Constant*`, `Shape`, `Unsqueeze` — todas soportadas por onnxruntime-web (backend wasm), ninguna requiere hilos/SharedArrayBuffer |
| Inferencia nativa (PyTorch CPU, 1 hilo lógico) — tema de 2:17 (Amelie) | 0,62 s (5920 frames CQT) |
| Inferencia nativa (PyTorch CPU) — tema de 3:26 (Cable A Tierra) | 1,53 s (8855 frames CQT) |
| Decodificación de audio (librosa, incluye demux) | 1-46 s según formato/codec — cuello de botella real es esto, no la red |

Extrapolando con margen conservador (ONNX Runtime Web en wasm de un solo hilo suele ser
5-20× más lento que PyTorch nativo con BLAS optimizado para este tamaño de modelo): incluso
en el peor caso razonable, la inferencia completa de una canción de 3-5 minutos cae en el
rango de **unos pocos segundos**, muy por debajo del precedente ya aceptado en el ecosistema
(demucs.cpp WASM para stems corre en minutos en navegador, ver roadmap §2 v2.0).

## Calidad de las etiquetas (validación de oído pendiente del fundador)

Progresión completa de dos temas de `test/private/` (no commiteado, ver `.gitignore`):
detalle y tablas en `docs/especificacion.md` §3. Resumen: se observaron tríadas (maj/min/dim/
sus4), séptimas completas (maj7, 7, m7, dim7, semidisminuido m7♭5) y extensiones (9, ♭9, ♯11,
13) en un tema real de banda completa — confirma que el modelo entrega el vocabulario
"completo" requerido, no solo tríadas.

## Limitaciones conocidas de este prototipo (no del modelo)

- Un solo fold de los 5 del ensemble original (`_s0`) — el paper promedia los 5 para su mejor
  precisión reportada.
- Sin decodificación HMM (el repo original trae `extractors/xhmm_ismir.py` para suavizado
  temporal); se sustituyó por un filtro de mediana corto + agrupación por largo de racha. Es
  notablemente más ruidoso en los cambios de acorde que una HMM entrenada — visible en el
  ejemplo del tema de banda completa (unos pocos frames sueltos con alteraciones
  espurias/inestables). Portar (o reimplementar) la HMM es trabajo real de la próxima sesión
  de construcción, no de esta.
- Sin beat tracking (fuera de alcance de esta sesión — ver especificación §5, decisión
  diferida a la sesión de construcción de v3.0).
