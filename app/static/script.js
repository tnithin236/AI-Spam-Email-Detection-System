const API = ""; // same origin

async function loadModels() {
  const res = await fetch(`${API}/models`);
  const data = await res.json();

  const select = document.getElementById("model-select");
  select.innerHTML = "";
  data.available_models.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    if (name === "Linear SVM") opt.selected = true; // best F1 from training
    select.appendChild(opt);
  });

  renderMetrics(data.metrics);
}

function renderMetrics(metrics) {
  const container = document.getElementById("model-metrics");
  const rows = Object.entries(metrics)
    .map(
      ([name, m]) => `
      <tr>
        <td>${name}</td>
        <td>${(m.accuracy * 100).toFixed(1)}%</td>
        <td>${(m.precision * 100).toFixed(1)}%</td>
        <td>${(m.recall * 100).toFixed(1)}%</td>
        <td>${(m.f1 * 100).toFixed(1)}%</td>
      </tr>`
    )
    .join("");

  container.innerHTML = `
    <table class="metrics">
      <thead>
        <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

async function loadStats() {
  const res = await fetch(`${API}/stats`);
  const data = await res.json();

  document.getElementById("stats").innerHTML = `
    <div class="stat-box">
      <div class="stat-value">${data.total}</div>
      <div class="stat-label">Emails analyzed</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">${data.spam}</div>
      <div class="stat-label">Spam detected</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">${data.not_spam}</div>
      <div class="stat-label">Legitimate</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">${data.spam_percentage}%</div>
      <div class="stat-label">Spam rate</div>
    </div>
  `;
}

async function analyzeEmail() {
  const subject = document.getElementById("subject").value;
  const body = document.getElementById("body").value;
  const model = document.getElementById("model-select").value;
  const btn = document.getElementById("analyze-btn");
  const resultBox = document.getElementById("result");

  if (!subject.trim() && !body.trim()) {
    alert("Enter a subject or email body first.");
    return;
  }

  btn.disabled = true;
  btn.textContent = "Analyzing...";

  try {
    const res = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject, body, model }),
    });

    if (!res.ok) throw new Error((await res.json()).detail || "Request failed");
    const data = await res.json();

    resultBox.className = `result ${data.label}`;
    const labelText = data.label === "spam" ? "🚨 SPAM DETECTED" : "✅ NOT SPAM";
    const confidencePct = (data.confidence * 100).toFixed(1);

    const chips = data.top_indicators
      .map(
        (ind) =>
          `<span class="indicator-chip ${ind.direction === "spam" ? "spam-word" : "safe-word"}">${ind.word}</span>`
      )
      .join("");

    resultBox.innerHTML = `
      <div class="result-label">${labelText}</div>
      <div class="result-confidence">${confidencePct}% confidence · ${data.model_used}</div>
      ${chips ? `<div class="indicators">${chips}</div>` : ""}
    `;
    resultBox.classList.remove("hidden");

    loadStats();
  } catch (err) {
    resultBox.className = "result";
    resultBox.innerHTML = `<div class="result-label" style="color:#E5484D">Error</div><div class="result-confidence">${err.message}</div>`;
    resultBox.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze Email";
  }
}

async function analyzeCsv() {
  const fileInput = document.getElementById("csv-file");
  const resultBox = document.getElementById("batch-result");
  const model = document.getElementById("model-select").value;

  if (!fileInput.files.length) {
    alert("Choose a CSV file first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  resultBox.classList.remove("hidden");
  resultBox.textContent = "Processing...";

  try {
    const res = await fetch(`${API}/predict/batch?model=${encodeURIComponent(model)}`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) throw new Error((await res.json()).detail || "Request failed");

    const totalRows = res.headers.get("X-Total-Rows");
    const spamCount = res.headers.get("X-Spam-Count");

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "predictions.csv";
    a.click();

    resultBox.innerHTML = `Done — ${totalRows} emails analyzed, ${spamCount} flagged as spam. Results downloaded as <code>predictions.csv</code>.`;
    loadStats();
  } catch (err) {
    resultBox.textContent = `Error: ${err.message}`;
  }
}

document.getElementById("analyze-btn").addEventListener("click", analyzeEmail);
document.getElementById("batch-btn").addEventListener("click", analyzeCsv);

loadModels();
loadStats();
