// -------------------------------
// Load ONNX Student Model
// -------------------------------
let session = null;

async function loadModel() {
  try {
    // Make sure DLnet_Data.onnx is in the same folder as index.html and index.js
    session = await ort.InferenceSession.create("./DLnet2026_WineData.onnx");
    console.log("Model loaded successfully.");
  } catch (err) {
    console.error("Failed to load ONNX model:", err);
  }
}

loadModel();


// -------------------------------
// Exact feature order for ONNX
// -------------------------------
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


// -------------------------------
// Collect inputs from HTML
// -------------------------------
function collectModelInput() {
  const values = {};

  values['Marital status'] = Number(document.getElementById('marital_status').value);
  values['Application mode'] = Number(document.getElementById('application_mode').value);
  values['Application order'] = Number(document.getElementById('application_order').value);
  values['Course'] = Number(document.getElementById('course').value);
  values['Daytime/evening attendance'] = Number(document.getElementById('attendance').value);
  values['Previous qualification'] = Number(document.getElementById('prev_qual').value);
  values['Previous qualification (grade)'] = Number(document.getElementById('prev_qual_grade').value);
  values['Nacionality'] = Number(document.getElementById('nacionality').value);
  values["Mother's qualification"] = Number(document.getElementById('mother_qual').value);
  values["Father's qualification"] = Number(document.getElementById('father_qual').value);
  values["Mother's occupation"] = Number(document.getElementById('mother_occ').value);
  values["Father's occupation"] = Number(document.getElementById('father_occ').value);
  values['Admission grade'] = Number(document.getElementById('admission_grade').value);
  values['Displaced'] = Number(document.getElementById('displaced').value);
  values['Educational special needs'] = Number(document.getElementById('special_needs').value);
  values['Debtor'] = Number(document.getElementById('debtor').value);
  values['Tuition fees up to date'] = Number(document.getElementById('tuition_up_to_date').value);
  values['Gender'] = Number(document.getElementById('gender').value);
  values['Scholarship holder'] = Number(document.getElementById('scholarship').value);
  values['Age at enrollment'] = Number(document.getElementById('age').value);
  values['International'] = Number(document.getElementById('international').value);
  values['Curricular units 1st sem (credited)'] = Number(document.getElementById('c1_credited').value);
  values['Curricular units 1st sem (enrolled)'] = Number(document.getElementById('c1_enrolled').value);
  values['Curricular units 1st sem (evaluations)'] = Number(document.getElementById('c1_evaluations').value);
  values['Curricular units 1st sem (approved)'] = Number(document.getElementById('c1_approved').value);
  values['Curricular units 1st sem (grade)'] = Number(document.getElementById('c1_grade').value);
  values['Curricular units 1st sem (without evaluations)'] = Number(document.getElementById('c1_without_eval').value);
  values['Curricular units 2nd sem (credited)'] = Number(document.getElementById('c2_credited').value);
  values['Curricular units 2nd sem (enrolled)'] = Number(document.getElementById('c2_enrolled').value);
  values['Curricular units 2nd sem (evaluations)'] = Number(document.getElementById('c2_evaluations').value);
  values['Curricular units 2nd sem (approved)'] = Number(document.getElementById('c2_approved').value);
  values['Curricular units 2nd sem (grade)'] = Number(document.getElementById('c2_grade').value);
  values['Curricular units 2nd sem (without evaluations)'] = Number(document.getElementById('c2_without_eval').value);
  values['Unemployment rate'] = Number(document.getElementById('unemployment').value);
  values['Inflation rate'] = Number(document.getElementById('inflation').value);
  values['GDP'] = Number(document.getElementById('gdp').value);

  // Build ordered vector
  return FEATURE_HEADERS.map(h => values[h]);
}


// -------------------------------
// Run ONNX model
// -------------------------------
async function runModel() {
  if (!session) {
    alert("Model not loaded yet. Wait a moment and try again.");
    return;
  }

  const inputVector = collectModelInput();

  try {
    const tensor = new ort.Tensor('float32', Float32Array.from(inputVector), [1, 37]);

    // input name and output name must match your ONNX graph
    const results = await session.run({ input1: tensor });

    const prediction = results.output1.data[0];

    document.getElementById('result').innerText =
      "Prediction: " + prediction.toFixed(4);

  } catch (err) {
    console.error("Error running model:", err);
    document.getElementById('result').innerText =
      "Error running model. Check console.";
  }
}


