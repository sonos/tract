// Supertonic releases, as Supertone published them and as their archive keeps them on
// Hugging Face, pinned to a revision. Both share one pipeline and one set of graph inputs;
// the third release reads more languages with a larger vector estimator.

const LANGUAGES = {
  en: 'English', ko: 'Korean', ja: 'Japanese', ar: 'Arabic', bg: 'Bulgarian', cs: 'Czech',
  da: 'Danish', de: 'German', el: 'Greek', es: 'Spanish', et: 'Estonian', fi: 'Finnish',
  fr: 'French', hi: 'Hindi', hr: 'Croatian', hu: 'Hungarian', id: 'Indonesian', it: 'Italian',
  lt: 'Lithuanian', lv: 'Latvian', nl: 'Dutch', pl: 'Polish', pt: 'Portuguese', ro: 'Romanian',
  ru: 'Russian', sk: 'Slovak', sl: 'Slovenian', sv: 'Swedish', tr: 'Turkish', uk: 'Ukrainian',
  vi: 'Vietnamese',
};

export const languageName = (code) => LANGUAGES[code] || code;

export const VOICES = ['F1', 'F2', 'F3', 'F4', 'F5', 'M1', 'M2', 'M3', 'M4', 'M5'];

export const RELEASES = [
  {
    id: 'supertonic-3',
    name: 'Supertonic 3',
    note: '31 languages',
    repo: 'supertone-oss-archive/supertonic-3',
    revision: 'aafc6e32416a594460b32413efc49d7fe4ce6d46',
    languages: Object.keys(LANGUAGES),
    graphs: {
      duration_predictor: 3700147,
      text_encoder: 36416150,
      vector_estimator: 256534781,
      vocoder: 101424195,
    },
  },
  {
    id: 'supertonic-2',
    name: 'Supertonic 2',
    note: '5 languages, smaller',
    repo: 'supertone-oss-archive/supertonic-2',
    revision: '7edbcd0bba8cd579100a353a9e0b2b49cad7c875',
    languages: ['en', 'ko', 'es', 'pt', 'fr'],
    graphs: {
      duration_predictor: 1521526,
      text_encoder: 27431318,
      vector_estimator: 132471364,
      vocoder: 101405066,
    },
  },
];

export const release = (id) => {
  const found = RELEASES.find((candidate) => candidate.id === id);
  if (!found) throw new Error(`Unknown model ${id}`);
  return found;
};

export const fileUrl = (model, path) =>
  `https://huggingface.co/${model.repo}/resolve/${model.revision}/${path}`;

export const licenseUrl = (model) => `https://huggingface.co/${model.repo}/blob/${model.revision}/LICENSE`;

export const graphBytes = (model) => Object.values(model.graphs).reduce((a, b) => a + b, 0);

export const megabytes = (bytes) => (bytes / 1e6).toFixed(bytes >= 10e6 ? 0 : 1);

// One line per language, so a visitor hears the model without typing.
export const SAMPLES = {
  en: 'This voice is synthesized in your browser. The text never leaves your machine: tract runs the model right here, on your own graphics card.',
  fr: 'Cette voix est synthétisée dans votre navigateur. Le texte ne quitte jamais votre machine : tract fait tourner le modèle ici même, sur votre carte graphique.',
  es: 'Esta voz se sintetiza en tu navegador. El texto nunca sale de tu equipo: tract ejecuta el modelo aquí mismo, en tu propia tarjeta gráfica.',
  pt: 'Esta voz é sintetizada no seu navegador. O texto nunca sai do seu computador: o tract executa o modelo aqui mesmo, na sua placa gráfica.',
  de: 'Diese Stimme wird in Ihrem Browser erzeugt. Der Text verlässt nie Ihren Rechner: tract führt das Modell direkt hier aus, auf Ihrer eigenen Grafikkarte.',
  it: 'Questa voce è sintetizzata nel tuo browser. Il testo non lascia mai il tuo computer: tract esegue il modello proprio qui, sulla tua scheda grafica.',
  nl: 'Deze stem wordt in je browser gemaakt. De tekst verlaat je computer nooit: tract draait het model hier, op je eigen grafische kaart.',
  ko: '이 목소리는 브라우저에서 합성됩니다. 텍스트는 컴퓨터를 떠나지 않습니다.',
  ja: 'この声はブラウザの中で合成されています。テキストがあなたのコンピューターから出ることはありません。',
};
