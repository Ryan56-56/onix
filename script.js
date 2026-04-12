// script.js

const outputEl = document.getElementById('output');
const runBtn = document.getElementById('run-btn');

async function loadAndRunModel() {
  try {
    outputEl.textContent = 'Loading model...';

    // Load your actual ONNX model
    const session = await ort.InferenceSession.create('./DLnet2026_WineData.onnx', {
      executionProviders: ['wasm']
    });

    // Example dummy input — update shape to match your model
    const inputData = new Float32Array([0.1, 0.2, 0.3]);
    const tensor = new ort.Tensor('float32', inputData, [1, 3]);

    // Replace 'input' with your model's real input name
    const feeds = { input: tensor };

    outputEl.textContent = 'Running inference...';

    const results = await session.run(feeds);

    const outputNames = Object.keys(results);
    const firstOutputName = outputNames[0];
    const outputTensor = results[firstOutputName];

    outputEl.textContent =
      'Output name: ' + firstOutputName + '\n' +
      'Data: ' + JSON.stringify(Array.from(outputTensor.data), null, 2);

  } catch (err) {
    console.error(err);
    outputEl.textContent = 'Error: ' + err.message;
  }
}

runBtn.addEventListener('click', loadAndRunModel);


