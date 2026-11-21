# fast-alpr-node

**Idioma:** [English](README.md) · [Português](README.pt.md) · [Español](README.es.md)

Porta do [fast-alpr](https://github.com/ankandrew/fast-alpr) (reconhecimento automático de placas) para Node.js / TypeScript.

Usa os mesmos modelos ONNX da biblioteca Python via `onnxruntime-node`; só o pré e o pós-processamento foram
reimplementados. Na primeira execução os modelos são baixados para `~/.cache/open-image-models` e
`~/.cache/fast-plate-ocr` (o mesmo cache das bibliotecas Python).

## Instalação

```sh
npm install fast-alpr-node
```

Requer **Node.js ≥ 20**. As dependências nativas (`sharp`, `onnxruntime-node`) trazem binários por plataforma.
Só Node — não é pacote para browser. Os pesos ONNX são baixados no primeiro uso para o cache acima.

## Uso

```ts
import { ALPR } from "fast-alpr-node";

const alpr = await ALPR.create({
  detectorModel: "yolo-v9-t-384-license-plate-end2end", // padrão
  ocrModel: "cct-xs-v2-global-model", // padrão
});

const results = await alpr.predict("assets/test_image.png");
for (const { detection, ocr } of results) {
  console.log(detection.boundingBox, detection.confidence, ocr?.text, ocr?.region);
}

// Também aceitos: Buffer / Uint8Array, base64 / data URL, stream de bytes ou um `Image` BGR
// await alpr.predict(buffer);
// await alpr.predict(base64String);
// await alpr.predict(fs.createReadStream("foto.jpg"));
```

`predict` devolve um `ALPRResult` por placa: `detection` (caixa, rótulo, confiança) e `ocr`
(`text`, `confidence` por caractere, `region` / `regionConfidence` opcionais, ou `null`).

Entradas aceitas: caminho de arquivo, bytes da imagem (`Buffer` / `Uint8Array` / `ArrayBuffer`),
string base64 ou URL `data:image/...;base64,...`, stream assíncrono de bytes (ex.: `Readable` do Node)
ou um `Image` BGR já decodificado.

Exemplo CLI: `npm run predict -- path/to/image.png`

## Como funciona

O pipeline tem duas etapas: um detector de placas YOLOv9 e um OCR CCT em cada recorte.
Veja o passo a passo completo (fluxogramas, geometria do letterbox, decodificação dos slots do OCR,
mapa de módulos e o resultado da imagem de teste):

- [English](docs/how-it-works.en.md)
- [Português](docs/how-it-works.pt.md)
- [Español](docs/how-it-works.es.md)

## Diferenças em relação à versão Python

- **Backend OpenCV**: `@techstark/opencv-js` (WASM, mesmo código C++) em vez de `opencv-python`. Só
  `src/cv.ts` toca no OpenCV, então dá para trocar por um binding nativo. A decodificação de arquivos
  usa `sharp`.
- **API assíncrona**: os modelos são criados com `await ALPR.create()` e `predict` devolve uma Promise.
- **Ainda não portado**: `draw_predictions`, inferência em lote.
- **Providers**: o padrão é `["cpu"]`. Passe `detectorProviders` / `ocrProviders` (ex.: `["coreml", "cpu"]`)
  para mudar.
- Os valores de pixel podem diferir em 1 nível de cinza em relação ao Python após o resize (builds
  diferentes do OpenCV), então as confianças batem em ~1e-3 nos caracteres seguros, não bit a bit.
  Texto, região e caixas coincidem.

## Desenvolvimento

```sh
npm install
npm run typecheck
npm test            # testes unitários + comparação com a biblioteca Python
npm run build
```

### Regenerar as saídas de referência do Python

Os `test/golden/*.json` vêm da biblioteca original:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "fast-alpr[onnx]"
.venv/bin/python tools/make_variants.py                 # deriva imagens extras a partir de assets/test_image.png
.venv/bin/python tools/gen_golden.py assets/*.png
```

## Licença

MIT, como o original. Os pesos dos modelos são baixados dos projetos upstream e têm termos próprios.
