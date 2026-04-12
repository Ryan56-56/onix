// ======================================================
//  ONNX INFERENCE FUNCTIONS
//  (Your exact functions, organized into sections)
// ======================================================


// ------------------------------------------------------
// 1. Real Data Inverse (4 → 7)
// ------------------------------------------------------
async function runExample1() {
  const x = new Float32Array(4);
  x[0] = parseFloat(box0c1.value) || 0;
  x[1] = parseFloat(box1c1.value) || 0;
  x[2] = parseFloat(box2c1.value) || 0;
  x[3] = parseFloat(box3c1.value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 4]);

  try {
    const session = await ort.InferenceSession.create("./resNet_Inverse_realData_new.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data;

    predictions1.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td><td id="c1td0">${output[0].toFixed(2)}</td></tr>
        <tr><td>i_h2_temp</td><td id="c1td1">${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td><td id="c1td2">${output[2].toFixed(2)}</td></tr>
        <tr><td>i_pci_rate</td><td id="c1td3">${output[3].toFixed(2)}</td></tr>
        <tr><td>i_o2_volfract</td><td id="c1td4">${output[4].toFixed(2)}</td></tr>
        <tr><td>i_hbtemp</td><td id="c1td5">${output[5].toFixed(2)}</td></tr>
        <tr><td>i_wind_rt</td><td id="c1td6">${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

  runAvg();
}


// ------------------------------------------------------
// 2. Real Data Inverse (To → 7)
// ------------------------------------------------------
async function runExample2() {
  const x = new Float32Array(4);
  x[0] = parseFloat(box0c2.value) || 0;
  x[1] = parseFloat(box1c2.value) || 0;
  x[2] = parseFloat(box2c2.value) || 0;
  x[3] = parseFloat(box3c2.value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 4]);

  try {
    const session = await ort.InferenceSession.create("./resNet_Inverse_realData_new.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data;

    predictions2.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td><td id="c2td0">${output[0].toFixed(2)}</td></tr>
        <tr><td>i_h2_temp</td><td id="c2td1">${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td><td id="c2td2">${output[2].toFixed(2)}</td></tr>
        <tr><td>i_pci_rate</td><td id="c2td3">${output[3].toFixed(2)}</td></tr>
        <tr><td>i_o2_volfract</td><td id="c2td4">${output[4].toFixed(2)}</td></tr>
        <tr><td>i_hbtemp</td><td id="c2td5">${output[5].toFixed(2)}</td></tr>
        <tr><td>i_wind_rt</td><td id="c2td6">${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

  runAvg();
}


// ------------------------------------------------------
// 3. Synthetic Inverse (8 → 8)
// ------------------------------------------------------
async function runExampleSynthetic() {
  const x = new Float32Array(8);
  x[0] = parseFloat(box0c1.value) || 0;
  x[1] = parseFloat(box1c1.value) || 0;
  x[2] = parseFloat(box2c1.value) || 0;
  x[3] = parseFloat(box3c1.value) || 0;
  x[4] = parseFloat(box0c2.value) || 0;
  x[5] = parseFloat(box1c2.value) || 0;
  x[6] = parseFloat(box2c2.value) || 0;
  x[7] = parseFloat(box3c2.value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 8]);

  try {
    const session = await ort.InferenceSession.create("./resNet_Inverse_syntheticData.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data;

    predSynthetic.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td><td id="syntd0">${output[0].toFixed(2)}</td></tr>
        <tr><td>i_h2_temp</td><td id="syntd1">${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td><td id="syntd2">${output[2].toFixed(2)}</td></tr>
        <tr><td>i_pci_rate</td><td id="syntd3">${output[3].toFixed(2)}</td></tr>
        <tr><td>i_o2_volfract</td><td id="syntd4">${output[4].toFixed(2)}</td></tr>
        <tr><td>i_hbtemp</td><td id="syntd5">${output[5].toFixed(2)}</td></tr>
        <tr><td>i_wind_rt</td><td id="syntd6">${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

  runAvg();
}


// ------------------------------------------------------
// 4. Hong RealGen (F1F2) — Combined
// ------------------------------------------------------
async function runExampleRealGenHong() {
  await runExampleHW1();
  await runExampleHW2();
  runAvgHW();
}


// ------------------------------------------------------
// 5. Hong RealGen (From → 7)
// ------------------------------------------------------
async function runExampleHW1() {
  const x = new Float32Array(4);
  x[0] = parseFloat(box0c1.value) || 0;
  x[1] = parseFloat(box1c1.value) || 0;
  x[2] = parseFloat(box2c1.value) || 0;
  x[3] = parseFloat(box3c1.value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 4]);

  try {
    const session = await ort.InferenceSession.create("./F1F2_Inverse_HongRealGen.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data;

    predictionsHW1.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td><td id="c1td0HW">${output[0].toFixed(2)}</td></tr>
        <tr><td>i_h2_temp</td><td id="c1td1HW">${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td><td id="c1td2HW">${output[2].toFixed(2)}</td></tr>
        <tr><td>i_pci_rate</td><td id="c1td3HW">${output[3].toFixed(2)}</td></tr>
        <tr><td>i_o2_volfract</td><td id="c1td4HW">${output[4].toFixed(2)}</td></tr>
        <tr><td>i_hbtemp</td><td id="c1td5HW">${output[5].toFixed(2)}</td></tr>
        <tr><td>i_wind_rt</td><td id="c1td6HW">${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

  runAvgHW();
}


// ------------------------------------------------------
// 6. Hong RealGen (To → 7)
// ------------------------------------------------------
async function runExampleHW2() {
  const x = new Float32Array(4);
  x[0] = parseFloat(box0c2.value) || 0;
  x[1] = parseFloat(box1c2.value) || 0;
  x[2] = parseFloat(box2c2.value) || 0;
  x[3] = parseFloat(box3c2.value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 4]);

  try {
    const session = await ort.InferenceSession.create("./F1F2_Inverse_HongRealGen.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data;

    predictionsHW2.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td><td id="c2td0HW">${output[0].toFixed(2)}</td></tr>
        <tr><td>i_h2_temp</td><td id="c2td1HW">${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td><td id="c2td2HW">${output[2].toFixed(2)}</td></tr>
        <tr><td>i_pci_rate</td><td id="c2td3HW">${output[3].toFixed(2)}</td></tr>
        <tr><td>i_o2_volfract</td><td id="c2td4HW">${output[4].toFixed(2)}</td></tr>
        <tr><td>i_hbtemp</td><td id="c2td5HW">${output[5].toFixed(2)}</td></tr>
        <tr><td>i_wind_rt</td><td id="c2td6HW">${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

  runAvgHW();
}


// ------------------------------------------------------
// 7. Forward Model (7 → 4)
// ------------------------------------------------------
async function runExampleLEforward() {
  const x = new Float32Array(7);
  x[0] = parseFloat(LEbox0c1.value) || 0;
  x[1] = parseFloat(LEbox1c1.value) || 0;
  x[2] = parseFloat(LEbox2c1.value) || 0;
  x[3] = parseFloat(LEbox3c1.value) || 0;
  x[4] = parseFloat(LEbox4c1.value) || 0;
  x[5] = parseFloat(LEbox5c1.value) || 0;
  x[6] = parseFloat(LEbox6c1.value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 7]);

  try {
    const session = await ort.InferenceSession.create("./LEPINE_model_Forward.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data;

    predictionsLEforward.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>tgt</td><td>${output[0].toFixed(2)}</td></tr>
        <tr><td>hmt</td><td>${output[1].toFixed(2)}</td></tr>
        <tr><td>prod rate</td><td>${output[2].toFixed(2)}</td></tr>
        <tr><td>fta</td><td>${output[3].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }
}


// ------------------------------------------------------
// 8. Inverse Model (4 → 7)
// ------------------------------------------------------
async function runExampleLEinverse() {
  const x = new Float32Array(4);
  x[0] = parseFloat(LEbox0c2.value) || 0;
  x[1] = parseFloat(LEbox1c2.value) || 0;
  x[2] = parseFloat(LEbox2c2.value) || 0;
  x[3] = parseFloat(LEbox3c2.value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 4]);

  try {
    const session = await ort.InferenceSession.create("./LEPINE_model_Inverse_2.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data;

    predictionsLEinverse.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td><td>${output[0].toFixed(2)}</td></tr>
        <tr><td>pci rate</td><td>${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td><td>${output[2].toFixed(2)}</td></tr>
        <tr><td>o2 vol fract</td><td>${output[3].toFixed(2)}</td></tr>
        <tr><td>h2 temp</td><td>${output[4].toFixed(2)}</td></tr>
        <tr><td>hb temp</td><td>${output[5].toFixed(2)}</td></tr>
        <tr><td>wind rate</td><td>${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }
}


// ------------------------------------------------------
// 9. Averages (Real + To)
// ------------------------------------------------------
async function runAvg() {
  const c1 = [
    parseFloat(c1td0.innerHTML),
    parseFloat(c1td1.innerHTML),
    parseFloat(c1td2.innerHTML),
    parseFloat(c1td3.innerHTML),
    parseFloat(c1td4.innerHTML),
    parseFloat(c1td5.innerHTML),
    parseFloat(c1td6.innerHTML)
  ];

  const c2 = [
    parseFloat(c2td0.innerHTML),
    parseFloat(c2td1.innerHTML),
    parseFloat(c2td2.innerHTML),
    parseFloat(c2td3.innerHTML),
    parseFloat(c2td4.innerHTML),
    parseFloat(c2td5.innerHTML),
    parseFloat(c2td6.innerHTML)
  ];

  const avg = c1.map((v, i) => ((v + c2[i]) / 2).toFixed(2));

  difference.innerHTML = `
    <hr> Average is: <br/>
    <table>
      <tr><td>i_h2i_rate</td><td>${avg[0]}</td></tr>
      <tr><td>i_h2_temp</td><td>${avg[1]}</td></tr>
      <tr><td>i_ngi_rate</td><td>${avg[2]}</td></tr>
      <tr><td>i_pci_rate</td><td>${avg[3]}</td></tr>
      <tr><td>i_o2_volfract</td><td>${avg[4]}</td></tr>
      <tr><td>i_hbtemp</td><td>${avg[5]}</td></tr>
      <tr><td>i_wind_rt</td><td>${avg[6]}</td></tr>
    </table>`;
}


// ------------------------------------------------------
// 10. Averages (Hong RealGen)
// ------------------------------------------------------
async function runAvgHW() {
  const c1 = [
    parseFloat(c1td0HW.innerHTML),
    parseFloat(c1td1HW.innerHTML),
    parseFloat(c1td2HW.innerHTML),
    parseFloat(c1td3HW.innerHTML),
    parseFloat(c1td4HW.innerHTML),
    parseFloat(c1td5HW.innerHTML),
    parseFloat(c1td6HW.innerHTML)
  ];

  const c2 = [
    parseFloat(c2td0HW.innerHTML),
    parseFloat(c2td1HW.innerHTML),
    parseFloat(c2td2HW.innerHTML),

