// ===============================
// Load ONNX Model
// ===============================
let session = null;

async function loadModel() {
  console.log("Loading model...");
  try {
    session = await ort.InferenceSession.create("./model.onnx");
    console.log("Model loaded successfully.");
  } catch (err) {
    console.error("Model load error:", err);
  }
}

loadModel();


// ===============================
// Explicit Feature → HTML ID Mapping
// ===============================
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


// ===============================
// Collect Inputs
// ===============================
function collectModelInput() {
  const values = [];

  FEATURE_IDS.forEach(id => {
    const el = document.getElementById(id);

    if (!el) {
      console.error("Missing HTML element for ID:", id);
      values.push(0);
      return;
    }

    const num = Number(el.value);
    values.push(num);
  });

  console.log("Input vector:", values);
  return values;
}


// ===============================
// Run Model
// ===============================
async function runModel() {
  console.log("runModel() called");

  if (!session) {
    console.log("Model not loaded yet.");
    document.getElementById("result").innerText = "Model not loaded yet.";
    return;
  }

  const inputVector = collectModelInput();

  try {
    const tensor = new ort.Tensor("float32", Float32Array.from(inputVector), [1, 37]);

    const results = await session.run({ input: tensor });

    console.log("Output keys:", Object.keys(results));

    const outputName = Object.keys(results)[0];
    const outputTensor = results[outputName];

    const predictedClass = outputTensor.data[0];

    const LABELS = ["Dropout", "Enrolled", "Graduate"];
    const predictedLabel = LABELS[predictedClass] || "Unknown";

    document.getElementById("result").innerText =
      `Prediction: ${predictedClass} (${predictedLabel})`;

  } catch (err) {
    console.error("Error running model:", err);
    document.getElementById("result").innerText = "Error running model.";
  }
}

