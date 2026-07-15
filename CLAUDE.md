# CLAUDE.md — chordwagon / trainmusiq

## Qué es este proyecto
chordwagon: detección de acordes (vocabulario completo: tríadas, séptimas, suspendidos,
disminuidos, alteraciones) y el tren armónico funcional — cada acorde un vagón. Tercera
herramienta del ecosistema trainmusiq (etapa 3), hermana de
[centrail](https://github.com/trainmusiq/centrail) (afinación, publicada) y trackjunction
(stems, etapa 2). ANTES DE CUALQUIER TAREA: lee `docs/especificacion.md` (el qué técnico de
chordwagon, vive aquí). Los documentos transversales del ecosistema (roadmap, método,
identidad) viven en el repo privado `trainmusiq/trainmusiq`, clonado en este equipo en
`/Users/juanma/Aat/Trainmusiq/docs/` — léelos ahí, no se duplican en este repo.

## Estado
Spec-ready (15 jul 2026): benchmark de motor + prototipo mínimo validados (ver
`docs/especificacion.md` §3 y `research/2026-07-15-chordnet-benchmark/`). Construcción de la
app aún no empezó — no hay `engine/` ni `index.html` todavía.

## Reglas duras (heredadas de centrail + trainmusiq; no re-discutir, el porqué está en docs/)
1. **Tubería de decodificación unificada**: el mismo audio decodificado una sola vez alimenta
   CQT/features y cualquier reproducción sincronizada — nunca medir por un camino y mostrar/
   reproducir por otro (mismo espíritu que la regla 1 de centrail, aplicada aquí a features en
   vez de a pitch).
2. **Sin recursos de CDN externos**: todo auto-hosteado (fuentes, modelo ONNX, runtime
   incluidos).
3. **Sin dependencias con hilos/SharedArrayBuffer** (GitHub Pages no permite COOP/COEP).
   Verificar TODA librería WASM nueva, incluyendo el build de `onnxruntime-web` elegido: debe
   ser el backend `wasm` sin `-threaded`, grep por `pthread|USE_PTHREADS|Atomics` + test de
   instanciación. Ver hallazgo en §11 de la especificación.
4. **Vendorizar con versión fijada** todo motor/runtime de inferencia (modelo ONNX +
   onnxruntime-web): bundle sin imports externos, licencia verificada compatible GPL v3,
   commiteada en `vendor/`. Mismo patrón que rubberband-wasm en centrail.
5. **Progreso honesto**: toda operación >1s reporta % real, etapa nombrada y ETA. Nunca
   spinner indeterminado.
6. **Copy de UI**: sincero, directo y afectivo; disuasivo, no imperativo. Detalle técnico
   secundario/expandible.
7. **Actuar-mínimo-e-informar**: decisiones técnicas se aplican con el mínimo necesario y se
   informan; jamás se le preguntan al usuario.
8. **Nada entra a una versión en construcción**: ideas nuevas → `roadmap.md` §6 del repo
   `trainmusiq/trainmusiq`.
9. **test/private/ está en `.gitignore`**: puede contener audio con copyright del fundador.
   JAMÁS commitearlo ni referenciarlo en código público.
10. **Nunca `torch.load(..., weights_only=False)` sobre un checkpoint de terceros** al
    reproducir benchmarks de investigación (riesgo de pickle arbitrario) — usar
    `weights_only=True`; si el checkpoint trae objetos no-tensor (como el `.sdict` de
    ChordNet, que envuelve el state dict en un diccionario con métricas de entrenamiento),
    inspeccionar antes de asumir su forma.
11. **Commits por hito**, mensajes descriptivos, push al cierre de cada hito.
12. **Idioma — español neutro (registrada 15 jul 2026)**: todo texto producido — copy de producto (UI, correos, páginas) Y reportes/comunicaciones de sesión — se escribe en español neutro: tuteo estándar (quieres, suelta, haz clic, puedes), nunca voseo (querés, soltá, hacé, podés) ni regionalismos de ningún país (ver `brief-diseno.md` de `trainmusiq/trainmusiq`).

## Al cerrar cada sesión
Reportar: checklist de lo pedido con ✓/✗/⚠ (los ⚠ honestos valen más que ✓ de cortesía),
commits hechos, y qué quedó pendiente con su porqué. Actualizar `docs/especificacion.md` si la
realidad enseñó algo nuevo (hallazgos con fecha).
