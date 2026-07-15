# chordwagon — Especificación del proyecto

**Ecosistema:** trainmusiq (herramientas independientes para aprender/entrenar música) · **Producto:** chordwagon — detección de acordes y el tren armónico funcional
**Versión:** 0.1 (spec-ready) · Julio 2026
**Autor:** Juanma (Punta Arenas) con Claude
**Estado:** sesión de apertura de etapa (15 jul 2026) — benchmark del motor y prototipo mínimo validados en este documento; la app todavía no se construye. Documentos hermanos: `roadmap.md` §2 v3.0/§6 (el CUÁNDO/ORDEN/POR QUÉ comercial) y `manual-continuidad.md` (método) — viven en el repo privado `trainmusiq/trainmusiq`. Spec hermana: `centrail/docs/especificacion.md` (misma estructura, etapa 1, publicada).

**Nombre:** "chordwagon" — el vagón-acorde. La metáfora central del ecosistema (la progresión de acordes ES el tren; cada acorde, un vagón; la canción, el viaje) pertenece propiamente a esta herramienta, no al ecosistema completo (corrección de arquitectura de marca, roadmap.md §5, 6 jul 2026).

---

## 1. Visión

Una aplicación web que detecta la progresión de acordes de una grabación musical real — con **vocabulario completo** desde la primera versión funcional (tríadas, séptimas, suspendidos, disminuidos, aumentados y extensiones: 9/11/13, no solo tríadas — requisito duro del fundador, no negociable a "premium" en v1). Sobre esa progresión detectada nace la identidad de producto: el tren armónico, cada acorde un vagón, la canción como el viaje que dura lo que dura la música.

Mismo espíritu que centrail: procesamiento 100% client-side por defecto (el audio nunca sale del equipo del usuario), hosting estático en GitHub Pages, costo operativo $0, open source GPL v3. **Este documento encuentra evidencia de que ese objetivo es alcanzable para chordwagon** — a diferencia de la suposición inicial del roadmap ("si es muy pesado ⇒ nace server-first, v2.5"), el motor elegido es lo bastante liviano como para no necesitar esa salida (ver §3).

## 2. Ecosistema — dónde vive chordwagon

- **centrail** (publicada) — afinación de referencia, la puerta de entrada.
- **trackjunction** (etapa 2, en construcción) — stems + estudio.
- **chordwagon** (esta spec, etapa 3) — acordes + el tren armónico funcional.
- **triptheory / pianostation** (horizonte) — pedagogía y piano tab.

Detalle completo de la escalera de versiones y monetización: `roadmap.md` en `trainmusiq/trainmusiq`. Regla madre (inmutable): no se abre una etapa sin publicar la anterior — esta sesión ejecuta los pasos 1-3 del método de apertura (benchmark, prototipo, spec), **no construye producto**.

## 3. Motor de reconocimiento de acordes — decidido con evidencia

### 3.1 Lo evaluado

**ChordMini** (`ptnghia-j/ChordMiniApp`, MIT, referenciado en el roadmap) resultó ser, tras inspección, una **aplicación** (Next.js + backend Flask/PyTorch obligatorio para toda inferencia — "you must run the Python backend on localhost:5001"), no un modelo client-side. Es un *wrapper* de dos modelos de investigación independientes, ambos MIT y ambos con los pesos versionados directamente en su repo (no Git LFS, no Drive):

| Modelo | Repo | Licencia | Arquitectura | Checkpoint |
|---|---|---|---|---|
| **ChordNet** | `music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition` | MIT | CNN (Conv2d+InstanceNorm2d+MaxPool2d+SELU) + BiLSTM de 1 capa (hidden 192) | ~5,7 MB (incluye estado del optimizador; ~2,0 MB solo pesos) |
| **BTC** | `jayg996/BTC-ISMIR19` | MIT | Transformer bidireccional + CRF | ~12 MB |

Otras alternativas descartadas tras revisión rápida: **autochord** (Apache-2.0, mismo enfoque CRNN pero menor vocabulario documentado, sin ventaja sobre ChordNet); Chordino/NNLS-Chroma (GPL, plugin Vamp en C++, sin puerto WASM conocido, motor de template-matching sobre chroma — vocabulario más pobre que un modelo entrenado); madmom (GPL, Python puro, sin runtime de navegador). Ninguna superaba a ChordNet en la combinación licencia+vocabulario completo+tamaño.

**Decisión: ChordNet**, por ser sustancialmente más liviano que BTC (sin atención ni CRF) y por declarar el vocabulario completo (tríada + séptima + novena + oncena + trecena como clasificadores independientes — "chord structure decomposition") como diseño central del paper, exactamente el requisito duro del fundador. BTC queda anotado como candidato de mayor precisión potencial para una revisión futura (ej. v3.x, si la validación de oído en producción revela que ChordNet se queda corto), no bloquea esta decisión.

### 3.2 Evidencia medida en este equipo (MacBook Pro M1 Max, 32 GB, CPU-only)

Reproducible en `research/2026-07-15-chordnet-benchmark/` (script + README con pasos exactos).

| Métrica | Valor |
|---|---|
| Parámetros totales | 500.772 (~2,0 MB fp32) |
| Export a ONNX (opset 17) | ✓ sin errores — 1,9 MB |
| Operadores en el grafo ONNX | `Conv`, `InstanceNormalization`, `MaxPool`, `Selu`, `LSTM`, `MatMul`, `Add`, `Mul`, `Concat`, `Slice`, `Reshape`, `Transpose`, `Gather`, `Shape`, `Unsqueeze`, `Constant*` — todos soportados por `onnxruntime-web` backend `wasm`; **ninguno requiere hilos/SharedArrayBuffer** (regla dura §11) |
| Inferencia nativa (PyTorch CPU, single-thread) — tema 2:17 | 0,62 s para 5.920 frames CQT (0,10 ms/frame) |
| Inferencia nativa (PyTorch CPU, single-thread) — tema 3:26 | 1,53 s para 8.855 frames CQT (0,17 ms/frame) |

Extrapolando con margen conservador (`onnxruntime-web` en wasm de un solo hilo suele rendir 5-20× más lento que PyTorch nativo con BLAS para este tamaño de modelo): inferencia completa de una canción de 3-5 min en **pocos segundos**, muy por debajo del precedente ya aceptado en el ecosistema (demucs.cpp WASM para stems corre en minutos, roadmap §2 v2.0). **Conclusión: chordwagon puede nacer 100% client-side en v3.0** — no se necesita la salida server-first (v2.5) contemplada como plan B en el roadmap.

El cuello de botella real no es la red sino la decodificación/CQT del audio (1-46 s según formato en las pruebas), exactamente el mismo tipo de costo que centrail ya resuelve con Workers + progreso honesto (§9).

### 3.3 Prototipo mínimo — progresión real detectada (para validación del fundador)

Corrido end-to-end sin UI: decodificar → CQT híbrido (`bins_per_octave=36`, `n_bins=288`, `fmin=F#0`, `hop_length=512`, `sr=22050`, parámetros exactos del paper) → red neuronal → argmax + filtro de mediana corto (sustituto simplificado de la HMM del paper original, ver limitaciones abajo) → progresión con timestamps.

**Tema 1 — "Comptine d'un autre été" (Yann Tiersen, piano cover), `test/private/`:**

Progresión detectada: `N → Em → G → D → Em → G → D...` (patrón i-III-VII en Mi menor, o vi-I-V si se lee en Sol mayor), constante durante los ~137 s del tema, con un `Bm` ocasional de paso. Todas tríadas — coherente con ser una pieza de piano solo en arpegios, sin séptimas explícitas. Tabla completa de segmentos en `research/2026-07-15-chordnet-benchmark/`.

**Tema 2 — "Cable A Tierra" (banda completa), `test/private/`:** este es el que demuestra el vocabulario completo. Se detectaron, entre otros: tríadas (`A`, `Bm`, `F#m`), séptimas completas (`Dmaj7`, `B7`, `F#m7`, `Fmaj7`), disminuidos (`Ebdim`, `F#dim`, `Adim`), semidisminuido (`Fm7b5`), suspendidos (`Esus4`, `Gsus4(7)`, `Bsus4`) y una sección final con extensiones (`Bb(maj7,9,#11,13)`, `C(maj7,9)`) en una secuencia de modulación cromática ascendente muy propia de un final/coda de balada — la progresión de raíces se mantiene coherente durante todo ese tramo, buena señal de que el mapeo croma→acorde no tiene un bug de transposición sistemático.

**Nota honesta:** ambos temas son de `test/private/` (no commiteados, contienen copyright del fundador, regla §11.9 de `CLAUDE.md`) — la validación de oído queda pendiente del fundador, tal como pide el prompt de apertura de esta sesión.

### 3.4 Limitaciones conocidas del prototipo (no del modelo — trabajo real de la sesión de construcción)

- **Un solo fold** de los 5 del ensemble original (`_s0`) — el paper promedia los 5 para su mejor precisión reportada. Para v3.0: evaluar si conviene promediar 2-3 folds offline en un único set de pesos horneado (mejor que servir 5 modelos al navegador) o si un fold alcanza.
- **Sin decodificación HMM.** El repo original trae `extractors/xhmm_ismir.py` para suavizado temporal (Viterbi sobre una cadena de Markov oculta con probabilidades de transición entre acordes); se sustituyó acá por un filtro de mediana corto + agrupación por largo de racha. Es notablemente más ruidoso en los cambios de acorde — visible como alteraciones espurias de 1-2 frames en el ejemplo de banda completa. **Portar o reimplementar la HMM en JS es el ítem de ingeniería más importante pendiente para v3.0**, no cosmético: de esto depende que la timeline de acordes en UI se vea estable y no "parpadee".
- **CQT en JS es trabajo nuevo, no reutilización directa.** `librosa.hybrid_cqt` es Python/C; no hay un puerto directo evaluado a JS/WASM. Centrail ya resuelve análisis espectral custom en JS (`engine/detect.mjs`, FFT de 32.768 puntos) pero es FFT lineal, no Constant-Q — chordwagon necesita implementar (o vendorizar, si aparece una librería MIT/BSD/Apache adecuada — a evaluar: essentia.js es AGPL, licencia a revisar con cuidado antes de vendorizar) el cálculo CQT desde cero. Riesgo real, no trivial, primer punto de la sesión de construcción.
- **Sin beat tracking.** Requisito del roadmap (§2 v3.0) para sincronizar la timeline; no se evaluó en esta sesión (fuera del alcance de los pasos 1-2, línea de corte explícita del prompt de apertura). Candidatos para la sesión de construcción: extraer un `beat_track` liviano del propio `librosa`/reimplementación simple de detección de onsets+tempo (mismo espíritu de "medir con honestidad" que centrail), o evaluar Beat-Transformer (usado por ChordMini, también PyTorch/ONNX-exportable, a confirmar tamaño). Decisión diferida, no bloquea esta spec.

## 4. Etapa 3 — chordwagon v3.0 (alcance de la construcción, no de esta sesión)

- **Timeline de acordes sincronizada con reproducción**, con reproducción del propio archivo cargado (reutiliza el patrón de decodificación+Worker de centrail).
- **Modo simple (triadas) / completo (alteraciones)** — ambos gratis desde v1 (decisión del fundador: el vocabulario completo no se segmenta a premium en esta primera versión funcional; una futura segmentación comercial, si ocurre, se decide con datos de uso reales, no de entrada).
- **Beat tracking** — sincroniza la grilla de acordes al pulso (decisión de motor pendiente, §3.4).
- **Detección de tonalidad (key) global** — perfiles de croma sobre los acordes ya detectados; "casi gratis" una vez que el pipeline de acordes existe (no se adelanta a centrail porque el método de afinación es intencionalmente independiente de la tonalidad).
- **Transposición de charts** — reutiliza directamente la lógica de transposición por semitonos ya validada en centrail (`pitchScale = 2^(semitonos/12)` es irrelevante acá porque no hay audio que transportar, pero el shift de índice de acorde mod 12 es análogo y trivial).
- **Identidad visual vagones-acorde** nace acá — coordinar con fase Design (brief-diseno.md, trainmusiq/trainmusiq) cuando el fundador lo agende; no bloquea el arranque técnico.

### 4.1 v3.5 — Análisis armónico (candidato premium fuerte, horizonte de la siguiente etapa)

Sobre los acordes detectados: numerales romanos, cadencias, identificación de progresiones/bloques armónicos que se repiten o varían (la metáfora vagones-acorde hecha feature medible: "este tramo del tren se repite en el minuto 2:14"), comparación con progresiones célebres — con **music21** (MIT, corre en Python; evaluar viabilidad de puerto a Pyodide/WASM client-side o si esta feature específica justifica un tier servidor liviano, decisión de la sesión que abra v3.5). Corpus de referencia candidato: Isophonics (~180 canciones de Beatles anotadas). Cómputo/valor difícil de replicar ⇒ monetizable con legitimidad (roadmap §2).

## 5. Criterios de "terminado" para v3.0 (medibles — a validar en la sesión de construcción)

- Detecta la progresión de acordes de un archivo real de 3-5 minutos en el navegador, con vocabulario completo (no solo tríadas), en un tiempo total (decodificación + CQT + inferencia) del orden de segundos a low-single-digit-minutos — no debe sentirse más lento que el precedente de trackjunction.
- La timeline de acordes es temporalmente estable (sin parpadeo de 1-2 frames entre acordes vecinos) — gate directo de que la HMM/suavizado (§3.4) quedó bien portada, no del filtro de mediana del prototipo.
- Validación de oído del fundador sobre al menos 2 temas reales de `test/private/` con vocabulario variado (uno simple, uno con séptimas/alteraciones) — mínimo antes de declarar la detección "validada", mismo estándar que centrail exigió para su algoritmo de afinación.
- Publicado en GitHub Pages, GPL v3, sin CDN externos, sin servidor.

## 6. Riesgos y decisiones abiertas

- **CQT en JS** (§3.4): el riesgo técnico más grande de la etapa, sin precedente directo reusable del resto del ecosistema. Primer punto de la sesión de construcción.
- **HMM/suavizado temporal** (§3.4): sin esto, la timeline de acordes se ve poco profesional aunque el modelo esté acertando la mayoría de los frames.
- **Beat tracking sin decisión de motor** (§3.4): no bloquea el arranque de detección de acordes (son features independientes), pero si se subestima puede correr el riesgo de dispersión (regla madre) si se intenta resolver a mitad de la construcción de la timeline.
- **Un solo fold vs. ensemble de 5**: posible pérdida de precisión frente al paper original; a validar con oído real antes de decidir si vale la pena el costo de servir/promediar más de un modelo.
- **Licencia de una eventual librería CQT vendorizada**: si se evalúa essentia.js u otra, verificar compatibilidad GPL v3 con el mismo rigor que rubberband-wasm en centrail (AGPL, por ejemplo, es compatible pero exige atención en un proyecto GPL v3 puro — revisar antes de vendorizar, no asumir).
- **Dispersión** (riesgo #1 según el fundador): regla única — no se abre v3.5 sin publicar v3.0; nada se agrega a "terminado" (§5) una vez iniciada la construcción.

## 7. Reutilización del patrón del ecosistema

- **Decodificación de audio**: mismo principio que centrail (§CLAUDE.md regla 1) — un solo audio decodificado alimenta CQT y reproducción sincronizada, nunca dos caminos distintos.
- **Web Workers para todo el trabajo pesado**: CQT + inferencia ONNX corren en un Worker dedicado, nunca en el hilo principal — mismo hallazgo validado en centrail (9 jul 2026): los Workers dedicados están exentos del throttling de timers de Chrome en pestañas en segundo plano, crítico para chordwagon si un usuario cambia de pestaña durante el análisis.
- **Progreso honesto**: decodificar / calculando CQT / detectando acordes, con % real y ETA — mismas etapas nombradas que exige el principio de la casa.
- **Vendorización con versión fijada**: el modelo ONNX + `onnxruntime-web` (backend wasm sin threads) se vendorizan igual que rubberband-wasm — bundle sin imports externos, licencia verificada (MIT, ver §3.1), commiteado en `vendor/`.
- **GPL v3** desde el primer commit — MIT (ChordNet) es compatible como dependencia de un proyecto GPL v3.

## 8. Seguridad

Mismo principio que centrail: **la arquitectura client-side es la primera medida de seguridad.** Sin servidor, sin base de datos, sin cuentas, se eliminan por diseño las categorías principales de riesgo web. Lo específico de chordwagon:

- **COOP/COEP — a verificar en la sesión de construcción, mismo protocolo que centrail §11**: confirmar que el build de `onnxruntime-web` vendorizado es el backend `wasm` de un solo hilo (no el `wasm-threaded`, que sí requiere `SharedArrayBuffer` y por tanto headers COOP/COEP que GitHub Pages no permite configurar). Verificar con el mismo método ya validado: grep de `pthread|USE_PTHREADS|Atomics` en el bundle + `exports.memory.buffer instanceof SharedArrayBuffer` → debe dar `false`. Dado que el modelo tiene solo 500K parámetros, no hay necesidad de rendimiento que empuje hacia el build con threads — la opción single-thread alcanza sin concesiones de velocidad reales.
- **Cadena de suministro**: el checkpoint `.sdict` de terceros es un pickle de PyTorch — **nunca cargarlo con `torch.load(..., weights_only=False)`** en ningún script de este repo (riesgo de ejecución de código arbitrario); usar `weights_only=True` siempre, tal como se hizo en el benchmark de esta sesión (bloqueado automáticamente por el propio entorno de Claude Code al primer intento incorrecto — ver `research/2026-07-15-chordnet-benchmark/`). El modelo se exporta una sola vez a ONNX (formato de solo-datos, sin ejecución de código) y es ese `.onnx` el que se vendoriza y se sirve en producción — el `.sdict` de PyTorch nunca llega al navegador ni al repo público.
- **Dependabot + npm mínimo**: mismo patrón que centrail (repo ya configurado, `.github/dependabot.yml`).
- **test/private/ en `.gitignore`**: mismo audio de prueba con copyright del fundador reusado desde centrail para las validaciones de esta sesión — nunca commiteado ni referenciado por ruta en código público (solo mencionado por nombre en documentación, como arriba).
