# chordwagon

**trainmusiq** — detección de acordes y el tren armónico funcional. Cada acorde, un vagón.

**Estado:** spec-ready (15 jul 2026) — benchmark de motor y prototipo mínimo validados, la
construcción de la app todavía no empieza. Especificación completa:
[`docs/especificacion.md`](docs/especificacion.md).

## Qué hará

Detecta la progresión de acordes de una grabación real, con vocabulario **completo** desde el
día uno: tríadas, séptimas (maj7/7/m7/dim7/semidisminuido), suspendidos, disminuidos,
aumentados y extensiones (9/11/13) — no solo tríadas. Sobre esa progresión, la identidad
central de la herramienta: la armonía funcional como un tren, cada acorde un vagón, la
canción como el viaje.

## Motor (decidido con evidencia — ver especificación §3)

**ChordNet** (CNN + BiLSTM, [ISMIR 2019 Large-Vocabulary Chord Recognition](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition),
MIT), corriendo client-side vía ONNX Runtime Web. ~500K parámetros (~2 MB), export a ONNX
verificado sin operadores exóticos, sin necesidad de hilos/SharedArrayBuffer — compatible con
GitHub Pages tal cual, mismo patrón que el resto del ecosistema.

## Ecosistema

Hermana de [centrail](https://github.com/trainmusiq/centrail) (afinación, publicada) y
trackjunction (stems, en construcción). Mismo espíritu: 100% client-side, sin cuentas, sin
subida de audio a ningún servidor, GPL v3.

## Licencia

[GNU GPL v3](LICENSE).

---

# chordwagon (English)

**trainmusiq** — chord detection and the functional harmony train. Every chord, a wagon.

**Status:** spec-ready (Jul 15, 2026) — engine benchmark and minimal prototype validated; app
construction hasn't started yet. Full spec (Spanish): [`docs/especificacion.md`](docs/especificacion.md).

## What it will do

Detects the chord progression of a real recording, with a **complete** vocabulary from day
one: triads, sevenths (maj7/7/m7/dim7/half-diminished), suspended, diminished, augmented and
extensions (9/11/13) — not just triads. On top of that progression, the tool's core identity:
functional harmony as a train, each chord a wagon, the song as the journey.

## Engine (evidence-based decision — see spec §3)

**ChordNet** (CNN + BiLSTM, [ISMIR 2019 Large-Vocabulary Chord Recognition](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition),
MIT), running client-side via ONNX Runtime Web. ~500K parameters (~2 MB), ONNX export
verified with no exotic operators, no threads/SharedArrayBuffer needed — works on GitHub
Pages as-is, same pattern as the rest of the ecosystem.

## Ecosystem

Sibling of [centrail](https://github.com/trainmusiq/centrail) (tuning, published) and
trackjunction (stems, in progress). Same spirit: 100% client-side, no accounts, no audio
uploaded to any server, GPL v3.

## License

[GNU GPL v3](LICENSE).
