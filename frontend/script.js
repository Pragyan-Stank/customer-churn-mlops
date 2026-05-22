/**
 * Customer Churn Prediction — Frontend Logic
 * ──────────────────────────────────────────
 * Pure vanilla JS · async/await · modular
 */

/* ═══════════════════════════════════════════
   Configuration
   ═══════════════════════════════════════════ */
const CONFIG = {
  /**
   * API base URL — configured per environment:
   *
   *   Local dev  (python -m http.server):  "http://localhost:8000"
   *   Docker Compose / Production:         "/api"
   *
   * In Docker Compose the browser hits Nginx on port 3000.
   * Nginx reverse-proxies  /api/*  →  backend:8000/*  internally.
   * No hardcoded container IPs. No localhost between containers.
   */
  API_BASE_URL: "/api",
};

/* ═══════════════════════════════════════════
   DOM References
   ═══════════════════════════════════════════ */
const dom = {
  form:             document.getElementById("predictForm"),
  formCard:         document.getElementById("formCard"),
  loadingCard:      document.getElementById("loadingCard"),
  resultCard:       document.getElementById("resultCard"),
  errorCard:        document.getElementById("errorCard"),
  predictBtn:       document.getElementById("predictBtn"),
  resetBtn:         document.getElementById("resetBtn"),
  closeResult:      document.getElementById("closeResult"),
  newPredictionBtn: document.getElementById("newPredictionBtn"),
  retryBtn:         document.getElementById("retryBtn"),

  // Result elements
  probRingFill:     document.getElementById("probRingFill"),
  probPercent:      document.getElementById("probPercent"),
  riskBadge:        document.getElementById("riskBadge"),
  resultExplanation:document.getElementById("resultExplanation"),

  // Error
  errorMessage:     document.getElementById("errorMessage"),
};

/* ═══════════════════════════════════════════
   Feature Encoding Helpers
   ═══════════════════════════════════════════ */

/**
 * Feature encoding to match the model's training preprocessing.
 *
 * Preprocessing pipeline (src/data/preprocessing.py):
 *   1. Drops: RowNumber, CustomerId, Surname
 *   2. pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
 *   3. Drops target column 'Exited'
 *   4. StandardScaler applied
 *
 * Resulting column order (11 features):
 *   CreditScore, Age, Tenure, Balance, NumOfProducts,
 *   HasCrCard, IsActiveMember, EstimatedSalary,
 *   Geography_Germany, Geography_Spain, Gender_Male
 *
 * NOTE: France is the dropped reference for Geography.
 *       Female is the dropped reference for Gender.
 *       The backend applies the fitted StandardScaler at inference.
 */

function encodeGeography(value) {
  // Returns [Geography_Germany, Geography_Spain]
  // France is the dropped reference category
  switch (value) {
    case "germany": return [1, 0];
    case "spain":   return [0, 1];
    case "france":
    default:        return [0, 0];
  }
}

function encodeGender(value) {
  // Gender_Male: Male = 1, Female = 0 (Female is dropped)
  return value === "male" ? 1 : 0;
}

/**
 * Build the 11-element feature array from form values.
 * Must match the exact order used during training.
 */
function encodeFeatures(values) {
  const [geoGermany, geoSpain] = encodeGeography(values.geography);
  return [
    parseFloat(values.creditScore),
    parseFloat(values.age),
    parseFloat(values.tenure),
    parseFloat(values.balance),
    parseFloat(values.numProducts),
    parseFloat(values.hasCreditCard),
    parseFloat(values.isActiveMember),
    parseFloat(values.estimatedSalary),
    geoGermany,
    geoSpain,
    encodeGender(values.gender),
  ];
}

/* ═══════════════════════════════════════════
   Validation
   ═══════════════════════════════════════════ */
function getFormValues() {
  return {
    creditScore:     document.getElementById("creditScore").value,
    geography:       document.getElementById("geography").value,
    gender:          document.getElementById("gender").value,
    age:             document.getElementById("age").value,
    tenure:          document.getElementById("tenure").value,
    balance:         document.getElementById("balance").value,
    numProducts:     document.getElementById("numProducts").value,
    hasCreditCard:   document.getElementById("hasCreditCard").value,
    isActiveMember:  document.getElementById("isActiveMember").value,
    estimatedSalary: document.getElementById("estimatedSalary").value,
  };
}

function validateForm(values) {
  let valid = true;

  // Clear previous invalid states
  dom.form.querySelectorAll(".invalid").forEach(el => el.classList.remove("invalid"));

  const checks = [
    { id: "creditScore",     test: v => { const n = Number(v); return !isNaN(n) && n >= 300 && n <= 850; } },
    { id: "geography",       test: v => ["france","spain","germany"].includes(v) },
    { id: "gender",          test: v => ["male","female"].includes(v) },
    { id: "age",             test: v => { const n = Number(v); return !isNaN(n) && n >= 18 && n <= 100; } },
    { id: "tenure",          test: v => { const n = Number(v); return !isNaN(n) && n >= 0  && n <= 10; } },
    { id: "balance",         test: v => { const n = Number(v); return !isNaN(n) && n >= 0; } },
    { id: "numProducts",     test: v => { const n = Number(v); return !isNaN(n) && n >= 1  && n <= 4; } },
    { id: "hasCreditCard",   test: v => v === "0" || v === "1" },
    { id: "isActiveMember",  test: v => v === "0" || v === "1" },
    { id: "estimatedSalary", test: v => { const n = Number(v); return !isNaN(n) && n >= 0; } },
  ];

  for (const check of checks) {
    const val = values[check.id];
    if (val === "" || val === null || val === undefined || !check.test(val)) {
      document.getElementById(check.id).classList.add("invalid");
      valid = false;
    }
  }

  return valid;
}

/** Convert kebab-case ID to camelCase key (simple version). */
function toCamel(id) {
  // Our IDs are already camelCase, so just return as-is.
  return id;
}

/* ═══════════════════════════════════════════
   Risk Interpretation
   ═══════════════════════════════════════════ */
function interpretRisk(probability) {
  if (probability < 0.3) {
    return {
      level: "low",
      label: "Low Churn Risk",
      explanation: "This customer shows strong retention signals. Continue current engagement strategy.",
    };
  } else if (probability < 0.6) {
    return {
      level: "medium",
      label: "Medium Churn Risk",
      explanation: "Moderate risk detected. Consider proactive outreach or personalised offers to improve retention.",
    };
  } else {
    return {
      level: "high",
      label: "High Churn Risk",
      explanation: "High probability of churn. Immediate intervention recommended — review account activity and engagement.",
    };
  }
}

/* ═══════════════════════════════════════════
   UI State Management
   ═══════════════════════════════════════════ */
function showCard(cardId) {
  ["formCard", "loadingCard", "resultCard", "errorCard"].forEach(id => {
    dom[id].classList.toggle("hidden", id !== cardId);
  });

  // Scroll to visible card
  if (cardId !== "formCard") {
    dom[cardId].scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function showResult(probability) {
  const pct = Math.round(probability * 100);
  const risk = interpretRisk(probability);

  // Update ring
  const circumference = 2 * Math.PI * 52; // r = 52
  const offset = circumference - (probability * circumference);
  dom.probRingFill.style.strokeDashoffset = circumference; // reset
  // Force reflow to restart animation
  void dom.probRingFill.offsetWidth;
  requestAnimationFrame(() => {
    dom.probRingFill.style.strokeDashoffset = offset;
  });

  // Update text
  animateCounter(dom.probPercent, 0, pct, 800);

  // Risk badge
  dom.riskBadge.textContent = risk.label;
  dom.riskBadge.className = "risk-badge " + risk.level;

  // Explanation
  dom.resultExplanation.textContent = risk.explanation;

  // Show result card alongside form
  dom.formCard.classList.remove("hidden");
  dom.resultCard.classList.remove("hidden");
  dom.loadingCard.classList.add("hidden");
  dom.errorCard.classList.add("hidden");

  dom.resultCard.scrollIntoView({ behavior: "smooth", block: "center" });
}

function showError(message) {
  dom.errorMessage.textContent = message || "An unexpected error occurred. Please try again.";
  dom.formCard.classList.remove("hidden");
  dom.errorCard.classList.remove("hidden");
  dom.loadingCard.classList.add("hidden");
  dom.resultCard.classList.add("hidden");

  dom.errorCard.scrollIntoView({ behavior: "smooth", block: "center" });
}

function showLoading() {
  dom.loadingCard.classList.remove("hidden");
  dom.resultCard.classList.add("hidden");
  dom.errorCard.classList.add("hidden");
}

/* ═══════════════════════════════════════════
   Counter Animation
   ═══════════════════════════════════════════ */
function animateCounter(el, from, to, duration) {
  const start = performance.now();
  function step(timestamp) {
    const progress = Math.min((timestamp - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
    el.textContent = Math.round(from + (to - from) * eased);
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ═══════════════════════════════════════════
   API Integration
   ═══════════════════════════════════════════ */
async function predict(features) {
  const url = `${CONFIG.API_BASE_URL}/predict`;

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`Server responded with ${response.status}: ${errorBody}`);
  }

  return response.json();
}



/* ═══════════════════════════════════════════
   Event Handlers
   ═══════════════════════════════════════════ */
dom.form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const values = getFormValues();
  if (!validateForm(values)) return;

  const features = encodeFeatures(values);

  // Disable button
  dom.predictBtn.disabled = true;
  showLoading();

  try {
    const data = await predict(features);
    showResult(data.prediction);
  } catch (err) {
    console.error("Prediction error:", err);
    showError(err.message);
  } finally {
    dom.predictBtn.disabled = false;
  }
});

dom.closeResult.addEventListener("click", () => {
  dom.resultCard.classList.add("hidden");
});

dom.newPredictionBtn.addEventListener("click", () => {
  dom.resultCard.classList.add("hidden");
  dom.form.reset();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

dom.retryBtn.addEventListener("click", () => {
  dom.errorCard.classList.add("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
});

dom.form.addEventListener("reset", () => {
  // Clear invalid states after a tick (reset fires before clearing values)
  setTimeout(() => {
    dom.form.querySelectorAll(".invalid").forEach(el => el.classList.remove("invalid"));
    dom.resultCard.classList.add("hidden");
    dom.errorCard.classList.add("hidden");
  }, 0);
});


