function print(msg) {
  const log = document.getElementById("log");
  log.innerText += msg + "\n";
}

print("Attempting to load model...");

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
  "gdp"
];

// Label mapping
function mapPredictionToLabel(value) {
  if (value < 0.5) return "Dropout";
  if (value < 1.5) return "Enrolled";
  return "Graduate";
}

// Load ONNX model
async function loadModel() {
  try {
    session = await ort.InferenceSession.create("model.onnx");
    print("Model loaded successfully.");

    checkMissingIDs();
  } catch (err) {
    print("Error loading model: " + err);
  }
}

// Check for missing HTML IDs
function checkMissingIDs() {
  print("Checking for missing HTML IDs...");

  let missing = [];

  FEATURE_IDS.forEach(id => {
    if (!document.getElementById(id)) {
      missing.push(id);
      print("❌ MISSING ID IN HTML: " + id);
    }
  });

  if (missing.length === 0) {
    print("ID check complete.");
  } else {
    print("Missing IDs found: " + missing.join(", "));
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
    print("Model not loaded yet.");
    return;
  }

  print("runModel() called");

  const inputVector = collectModelInput();
  print("Input vector: " + JSON.stringify(inputVector));

  try {
    const tensor = new ort.Tensor(
      "float32",
      Float32Array.from(inputVector),
      [1, inputVector.length]
    );

    const feeds = { input: tensor };
    const results = await session.run(feeds);

    const outputName = session.outputNames[0];
    let raw = results[outputName].data;

    // UNIVERSAL FIX: extract number from ANY shape
    let prediction = Number(
      Array.isArray(raw) ? raw[0] :
      raw?.[0] ??
      raw
    );

    print("Raw prediction value: " + prediction);

    // Show numeric prediction (safe)
    document.getElementById("result").innerText = isNaN(prediction)
      ? "N/A"
      : prediction.toFixed(4);

    // Convert to label
    const label = mapPredictionToLabel(prediction);
    document.getElementById("result_label").innerText = label;

    print("Predicted Label: " + label);

  } catch (err) {
    print("Error running model: " + err);
  }
}

    // FIX: ensure prediction is a number
    let prediction = Number(results[outputName].data[0]);

    print("Prediction (raw): " + prediction);

    // Show numeric prediction
    document.getElementById("result").innerText = prediction.toFixed(4);

    // Convert to label
    const label = mapPredictionToLabel(prediction);
    document.getElementById("result_label").innerText = label;

    print("Predicted Label: " + label);

  } catch (err) {
    print("Error running model: " + err);
  }
}


    // Show numeric prediction
    document.getElementById("result").innerText = prediction.toFixed(4);

    // Convert to label
    const label = mapPredictionToLabel(prediction);
    document.getElementById("result_label").innerText = label;

    print("Predicted Label: " + label);

  } catch (err) {
    print("Error running model: " + err);
  }
}

// Load model AFTER page loads
window.onload = loadModel;

