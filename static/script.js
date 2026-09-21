/**
 * Frontend logic for the Loan Approval Prediction app (Phases A + C).
 *
 * Drives the 3-step form (progress bar + per-step validation), posts the
 * payload to `/predict` and renders the animated result.
 */

// Keep these lists in sync with `app/schemas.py` -> `LoanApplication`.
const TEXT_FIELDS = ["applicant_name"];
const CATEGORICAL_FIELDS = ["Education", "Self_employed"];
const NUMERIC_FIELDS = [
  "No_of_dependents",
  "Income_annum",
  "Loan_amount",
  "Loan_term",
  "Cibil_score",
  "Residential_assets_value",
  "Commercial_assets_value",
  "Luxury_assets_value",
  "Bank_asset_value",
];

const STEP_FIELDS = {
  1: ["applicant_name", "No_of_dependents", "Education", "Self_employed"],
  2: ["Income_annum", "Loan_amount", "Loan_term", "Cibil_score"],
  3: [
    "Residential_assets_value",
    "Commercial_assets_value",
    "Luxury_assets_value",
    "Bank_asset_value",
  ],
};

const TOTAL_STEPS = 3;

const form = document.getElementById("loan-form");
const panels = [...document.querySelectorAll(".step-panel")];
const stepItems = [...document.querySelectorAll(".step")];
const progressBar = document.getElementById("progress-bar");
const backButton = document.getElementById("back-btn");
const nextButton = document.getElementById("next-btn");
const submitButton = document.getElementById("submit-btn");
const resultBox = document.getElementById("result");

let currentStep = 1;

/* ------------------------------- validation ------------------------------ */
const MESSAGES = {
  applicant_name: "Please enter the applicant's name.",
  No_of_dependents: "Enter a number of dependents between 0 and 10.",
  Income_annum: "Annual income cannot be negative.",
  Loan_amount: "Loan amount cannot be negative.",
  Loan_term: "Loan term must be between 1 and 40 years.",
  Cibil_score: "CIBIL score must be between 300 and 900.",
  Residential_assets_value: "Please enter the residential asset value.",
  Commercial_assets_value: "Commercial assets cannot be negative.",
  Luxury_assets_value: "Luxury assets cannot be negative.",
  Bank_asset_value: "Bank assets cannot be negative.",
};

function fieldError(input) {
  const fallback = MESSAGES[input.id];
  if (input.validity.valueMissing) {
    return fallback || "This field is required.";
  }
  if (input.validity.rangeUnderflow || input.validity.rangeOverflow) {
    return fallback || "Value is out of range.";
  }
  return fallback || "Please enter a valid value.";
}

function validateField(input) {
  const wrapper = input.closest(".field");
  if (!wrapper) return true;

  const errorSlot = wrapper.querySelector(".field__error");
  const valid = input.checkValidity();

  wrapper.classList.toggle("field--invalid", !valid);
  if (errorSlot) {
    errorSlot.textContent = valid ? "" : fieldError(input);
  }
  return valid;
}

function validateStep(step) {
  let firstInvalid = null;

  for (const id of STEP_FIELDS[step]) {
    const input = document.getElementById(id);
    if (!input) continue;
    if (!validateField(input) && !firstInvalid) {
      firstInvalid = input;
    }
  }

  if (firstInvalid) {
    firstInvalid.focus();
    return false;
  }
  return true;
}

/* ------------------------------- navigation ------------------------------ */
function showStep(step) {
  currentStep = step;

  panels.forEach((panel) => {
    panel.classList.toggle("is-active", Number(panel.dataset.panel) === step);
  });

  stepItems.forEach((item) => {
    const index = Number(item.dataset.step);
    item.classList.toggle("is-active", index === step);
    item.classList.toggle("is-done", index < step);
  });

  progressBar.style.width = `${(step / TOTAL_STEPS) * 100}%`;

  backButton.hidden = step === 1;
  nextButton.hidden = step === TOTAL_STEPS;
  submitButton.hidden = step !== TOTAL_STEPS;

  resultBox.classList.add("hidden");
}

/* -------------------------------- payload -------------------------------- */
function buildPayload() {
  const payload = {};

  for (const id of TEXT_FIELDS) {
    payload[id] = document.getElementById(id).value.trim();
  }
  for (const id of CATEGORICAL_FIELDS) {
    payload[id] = document.getElementById(id).value;
  }
  for (const id of NUMERIC_FIELDS) {
    payload[id] = Number(document.getElementById(id).value);
  }

  return payload;
}

/* ------------------------------- rendering ------------------------------- */
const ESCAPES = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ESCAPES[char]);
}

function showResult(result, probability, applicantName) {
  const approved = result === "Approved";
  const percent = (probability * 100).toFixed(1);

  resultBox.className = `result ${approved ? "result--approved" : "result--rejected"}`;
  resultBox.innerHTML = `
    <span class="result__badge">${approved ? "&#10003;" : "&#10007;"}</span>
    <h2>${approved ? "Approved" : "Rejected"}</h2>
    <p><strong>${escapeHtml(applicantName)}</strong> &middot; model confidence ${percent}%</p>
    <div class="result__meter"><span style="width:${percent}%"></span></div>
    <div class="result__actions">
      <a href="${window.loanlensApiUrl("/history")}">Saved to history &middot; view submissions &rarr;</a>
    </div>
  `;
  resultBox.classList.remove("hidden");
}

function showError(message) {
  resultBox.className = "result result--error";
  resultBox.innerHTML = `
    <span class="result__badge">!</span>
    <h2>Something went wrong</h2>
    <p>${escapeHtml(message)}</p>
  `;
  resultBox.classList.remove("hidden");
}

/* --------------------------------- events -------------------------------- */
nextButton.addEventListener("click", () => {
  if (validateStep(currentStep) && currentStep < TOTAL_STEPS) {
    showStep(currentStep + 1);
  }
});

backButton.addEventListener("click", () => {
  if (currentStep > 1) {
    showStep(currentStep - 1);
  }
});

form.addEventListener("input", (event) => {
  const wrapper = event.target.closest ? event.target.closest(".field") : null;
  if (wrapper && wrapper.classList.contains("field--invalid")) {
    validateField(event.target);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!validateStep(TOTAL_STEPS)) return;

  const payload = buildPayload();

  submitButton.disabled = true;
  submitButton.textContent = "Predicting...";

  try {
    const response = await fetch(window.loanlensApiUrl("/predict"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(
        typeof errorBody.detail === "string"
          ? errorBody.detail
          : `Request failed with status ${response.status}`
      );
    }

    const data = await response.json();
    showResult(data.result, data.probability, payload.applicant_name);
  } catch (error) {
    showError(error.message);
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Predict loan";
  }
});

/* ---------------------------- CIBIL slider sync --------------------------- */
const cibilInput = document.getElementById("Cibil_score");
const cibilRange = document.getElementById("Cibil_score_range");

cibilRange.addEventListener("input", () => {
  cibilInput.value = cibilRange.value;
});

cibilInput.addEventListener("input", () => {
  cibilRange.value = cibilInput.value;
});

/* ---------------------------------- init --------------------------------- */
window.loanlensWireNavigation(document);
showStep(1);

