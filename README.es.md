# fast-alpr-node

**Idioma:** [English](README.md) · [Português](README.pt.md) · [Español](README.es.md)

Port de [fast-alpr](https://github.com/ankandrew/fast-alpr) (reconocimiento automático de matrículas) a Node.js / TypeScript.

Usa los mismos modelos ONNX de la biblioteca Python a través de `onnxruntime-node`; solo se ha
reimplementado el pre y el postprocesamiento. En la primera ejecución los modelos se descargan en
`~/.cache/open-image-models` y `~/.cache/fast-plate-ocr` (la misma caché que usan las bibliotecas Python).

## Uso

```ts
import { ALPR } from "fast-alpr-node";

const alpr = await ALPR.create({
  detectorModel: "yolo-v9-t-384-license-plate-end2end", // valor por defecto
  ocrModel: "cct-xs-v2-global-model", // valor por defecto
});

const results = await alpr.predict("assets/test_image.png");
for (const { detection, ocr } of results) {
  console.log(detection.boundingBox, detection.confidence, ocr?.text, ocr?.region);
}

// También aceptados: Buffer / Uint8Array, base64 / data URL, stream de bytes o un `Image` BGR
// await alpr.predict(buffer);
// await alpr.predict(base64String);
// await alpr.predict(fs.createReadStream("foto.jpg"));
```

`predict` devuelve un `ALPRResult` por matrícula: `detection` (caja, etiqueta, confianza) y `ocr`
(`text`, `confidence` por carácter, `region` / `regionConfidence` opcionales, o `null`).

Entradas aceptadas: ruta de archivo, bytes de la imagen (`Buffer` / `Uint8Array` / `ArrayBuffer`),
cadena base64 o URL `data:image/...;base64,...`, stream asíncrono de bytes (p. ej. `Readable` de Node)
o un `Image` BGR ya decodificado.

Ejemplo CLI: `npm run predict -- path/to/image.png`

## Cómo funciona

El pipeline tiene dos etapas: un detector de matrículas YOLOv9 y un OCR CCT en cada recorte.
Consulta la guía completa (diagramas de flujo, geometría del letterbox, decodificación de slots del OCR,
mapa de módulos y el resultado de la imagen de prueba):

- [English](docs/how-it-works.en.md)
- [Português](docs/how-it-works.pt.md)
- [Español](docs/how-it-works.es.md)

## Diferencias respecto a la versión Python

- **Backend OpenCV**: `@techstark/opencv-js` (WASM, el mismo código C++) en lugar de `opencv-python`. Solo
  `src/cv.ts` toca OpenCV, así que se puede sustituir por un binding nativo. La decodificación de archivos
  usa `sharp`.
- **API asíncrona**: los modelos se crean con `await ALPR.create()` y `predict` devuelve una Promise.
- **Aún no portado**: `draw_predictions`, inferencia por lotes.
- **Providers**: el valor por defecto es `["cpu"]`. Pasa `detectorProviders` / `ocrProviders`
  (p. ej. `["coreml", "cpu"]`) para cambiarlo.
- Los valores de píxel pueden diferir en 1 nivel de gris respecto a Python tras el resize (builds
  distintas de OpenCV), así que las confianzas coinciden en ~1e-3 en caracteres seguros, no bit a bit.
  Texto, región y cajas coinciden.

## Desarrollo

```sh
npm install
npm run typecheck
npm test            # pruebas unitarias + comparación con la biblioteca Python
npm run build
```

### Regenerar las salidas de referencia de Python

Los `test/golden/*.json` provienen de la biblioteca original:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "fast-alpr[onnx]"
.venv/bin/python tools/make_variants.py                 # deriva imágenes extra a partir de assets/test_image.png
.venv/bin/python tools/gen_golden.py assets/*.png
```

## Licencia

MIT, como el original. Los pesos de los modelos se descargan de los proyectos upstream y tienen sus propios términos.
