// -------------------------------
// Load ONNX Student Model
// -------------------------------
let session = null;

async function loadModel() {
    try {
        session = await ort.InferenceSession.create("./DLnet_Data.onnx");
        console.log("Model loaded successfully.");
    } catch (err) {
        console.error("Failed to load ONNX model:", err);
    }
}

loadModel();


// -------------------------------
// EXACT FEATURE ORDER FOR ONNX
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
// Collect Inputs From HTML
// -------------------------------
function collectModelInput() {
    const values = {};

    FEATURE_HEADERS.forEach(header => {
        const id = header
            .toLowerCase()
            .replace(/[\s()']/g, "_")   // convert to safe ID format
            .replace(/_+/g, "_");       // collapse multiple underscores

        const element = document.getElementById(id);

        if (!element) {
            console.warn("Missing HTML input for:", header, " → expected ID:", id);
            values[header] = 0; // fallback
        } else {
            values[header] = Number(element.value);
        }
    });

    return FEATURE_HEADERS.map(h => values[h]);
}


// -------------------------------
// Run ONNX Model
// -------------------------------
async function runModel() {
    if (!session) {
        alert("Model not loaded yet. Please wait a moment.");
        return;
    }

    const inputVector = collectModelInput();

    try {
        const tensor = new ort.Tensor("float32", Float32Array.from(inputVector), [1, 37]);

        const output = await session.run({ input1: tensor });

        const prediction = output.output1.data[0];

        document.getElementById("result").innerText =
            "Prediction: " + prediction.toFixed(4);

    } catch (err) {
        console.error("Error running model:", err);
        document.getElementById("result").innerText =
            "Error running model. Check console.";
    }
}

