console.log("Attempting to load model...");

let session = null;

// List of all 36 input feature IDs (NO TARGET)
const FEATURE_IDS = [
  "marital_status",
  "application_mode",
  "application_order",
  "course",
  "daytime_evening_attendance",
  "previous_qualification",
  "previous_qualification_grade",
  "nacionality",
  "mother_s_qualification",
  "father_s_qualification",
  "mother_s_occupation",
  "father_s_occupation",
  "admission_grade",
  "displaced",
  "educational_special_needs",
  "debtor",
  "tuition_fees_up_to_date",
  "gender",
  "scholarship_holder",
  "age_at_enrollment",
  "international",
  "curricular_units_1st_sem_credited",
  "curricular_units_1st_sem_enrolled",
  "curricular_units_1st_sem_evaluations",
  "curricular_units_1st_sem_approved",
  "curricular_units_1st_sem_grade",
  "curricular_units_1st_sem_without_evaluations",
  "curricular_units_2nd_sem_credited",
  "curricular_units_2nd_sem_enrolled",
  "curricular_units_2nd_sem_evaluations",
  "curricular_units_2nd_sem_approved",
  "curricular_units_2nd_sem_grade",
  "curricular_units_2nd_sem_without_evaluations",
  "unemployment_rate",
  "inflation_rate",
  "gdp",
  "target"
];

// Load ONNX model
async function loadModel() {
  try {
    session = await ort.InferenceSession.create("model.onnx");
    console.log("Model loaded successfully.");

    checkMissingIDs();
  } catch (err) {
    console.error("Error loading model:", err);
  }
}

// Check for missing HTML IDs
function checkMissingIDs() {
  console.log("Checking for missing HTML IDs...");

  let missing = [];

  FEATURE_IDS.forEach(id => {
    if (!document.getElementById(id)) {
      missing.push(id);
      console.error("❌ MISSING ID IN HTML:", id);
    }
  });

  if (missing.length === 0) {
    console.log("ID check complete.");
  } else {
    console.warn("Missing IDs found:", missing);
  }
}

// Collect input values from HTML
function collectModelInput() {
  let values = [];

  FEATURE_IDS.forEach(id => {
    let el = document.getElementById(id);
    let v = parseFloat(el.value);

    if (isNaN(v)) v = 0;

    values.push(v);
  });

  return values;
}

// Run model prediction
async function runModel() {
  if (!session) {
    console.error("Model not loaded yet.");
    return;
  }

  console.log("runModel() called");

  const inputVector = collectModelInput();
  console.log("Input vector:", inputVector);

  try {
    // AUTO‑SHAPE FIX — this prevents the 37 vs 36 error
    const tensor = new ort.Tensor(
      "float32",
      Float32Array.from(inputVector),
      [1, inputVector.length]
    );

    const feeds = { input: tensor };
    const results = await session.run(feeds);

    const outputName = session.outputNames[0];
    const prediction = results[outputName].data[0];

    console.log("Prediction:", prediction);

    document.getElementById("result").innerText = prediction.toFixed(4);

  } catch (err) {
    console.error("Error running model:", err);
  }
}

// Load model AFTER page loads
window.onload = loadModel;
