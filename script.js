// ======================================================
// Load ONNX Model
// ======================================================
let session = null;

async function loadModel() {
  try {
    session = await ort.InferenceSession.create("./model.onnx?v=1");
    console.log("Model loaded successfully.");
  } catch (err) {
    console.error("Failed to load ONNX model:", err);
  }
}

loadModel();


// ======================================================
// Feature Order (MUST match ONNX training order)
// ======================================================
const FEATURE_HEADERS = [
  'Marital status',
  'Application mode',
  'Application order',
  'Course',
  'Daytime/evening attendance',
  'Previous qualification',
  'Previous qualification (grade)',
  'Nacionality',
  "Mother's qualification",
  "Father's qualification",
  "Mother's occupation",
  "Father's occupation",
  'Admission grade',
  'Displaced',
  'Educational special needs',
  'Debtor',
  'Tuition fees up to date',
  'Gender',
  'Scholarship holder',
  'Age at enrollment',
  'International',
  'Curricular units 1st sem (credited)',
  'Curricular units 1st sem (enrolled)',
  'Curricular units 1st sem (evaluations)',
  'Curricular units 1st sem (approved)',
  'Curricular units 1st sem (grade)',
  'Curricular units 1st sem (without evaluations)',
  'Curricular units 2nd sem (credited)',
  'Curricular units 2nd sem (enrolled)',
  'Curricular units 2nd sem (evaluations)',
  'Curricular units 2nd sem (approved)',
  'Curricular units 2nd sem (grade)',
  'Curricular units 2nd sem (without evaluations)',
  'Unemployment rate',
  'Inflation rate',
  'GDP'
];


// ======================================================
// Collect Inputs From HTML
// ======================================================
function collectModelInput() {
  const values = {};

  FEATURE_HEADERS.forEach(header => {
    const id = header
      .toLowerCase()
      .replace(/[\s()']/g, "_")
      .replace(/_+/g, "_");

    values[header] = Number(document.getElementById(id).value);
  });

  return FEATURE_HEADERS.map(h => values[h]);
}


// ======================================================
// Run ONNX Model
// ======================================================
async function runModel() {
  if (!session) {
    alert("Model not loaded yet. Wait a moment and try again.");
    return;
  }

  const inputVector = collectModelInput();

  try {
    // Build tensor for ONNX model
    const tensor = new ort.Tensor(
      'float32',
      Float32Array.from(inputVector),
      [1, 37]
    );

    // Run model
    const results = await session.run({ input: tensor });

    // ONNX output tensor
    const outputTensor = results.output;
    const predictedClass = outputTensor.data[0];

    // Map class → label
    const LABELS = ["Dropout", "Enrolled", "Graduate"];
    const predictedLabel = LABELS[predictedClass] || "Unknown";

    // Display result
    document.getElementById('result').innerText =
      `Prediction: ${predictedClass} (${predictedLabel})`;

  } catch (err) {
    console.error("Error running model:", err);
    document.getElementById('result').innerText =
      "Error running model. Check console.";
  }
}

