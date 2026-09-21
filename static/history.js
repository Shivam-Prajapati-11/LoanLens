/**
 * History dashboard logic (Phase C).
 *
 * Fetches stored predictions from `/api/history` and renders summary stats plus
 * a filterable, sortable table. Cell values are written with `textContent`, so
 * stored data can never inject markup into the page.
 */

const COLUMNS = [
  { key: "id", label: "#", numeric: true },
  { key: "applicant_name", label: "Applicant" },
  { key: "no_of_dependents", label: "Dependents", numeric: true },
  { key: "education", label: "Education" },
  { key: "self_employed", label: "Self employed" },
  { key: "income_annum", label: "Annual income", numeric: true },
  { key: "loan_amount", label: "Loan amount", numeric: true },
  { key: "loan_term", label: "Term (yrs)", numeric: true },
  { key: "cibil_score", label: "CIBIL", numeric: true },
  { key: "residential_assets_value", label: "Residential", numeric: true },
  { key: "commercial_assets_value", label: "Commercial", numeric: true },
  { key: "luxury_assets_value", label: "Luxury", numeric: true },
  { key: "bank_asset_value", label: "Bank", numeric: true },
  { key: "probability", label: "Confidence", numeric: true },
  { key: "prediction", label: "Result" },
  { key: "created_at", label: "Submitted", date: true },
];

const table = document.getElementById("history-table");
const headRow = table.querySelector("thead tr");
const tbody = table.querySelector("tbody");
const stateBox = document.getElementById("table-state");
const searchInput = document.getElementById("search");
const refreshButton = document.getElementById("refresh-btn");

const numberFormat = new Intl.NumberFormat("en-IN");
const dateFormat = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

let submissions = [];
let sortKey = "created_at";
let sortDir = "desc";

/* --------------------------------- header -------------------------------- */
function buildHeader() {
  headRow.innerHTML = "";

  for (const column of COLUMNS) {
    const th = document.createElement("th");
    th.textContent = column.label;
    th.dataset.key = column.key;
    th.tabIndex = 0;
    th.title = `Sort by ${column.label}`;
    th.addEventListener("click", () => toggleSort(column.key));
    th.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        toggleSort(column.key);
      }
    });
    headRow.appendChild(th);
  }

  paintSortIndicators();
}

function paintSortIndicators() {
  for (const th of headRow.children) {
    if (th.dataset.key === sortKey) {
      th.dataset.dir = sortDir;
    } else {
      delete th.dataset.dir;
    }
  }
}

function toggleSort(key) {
  if (sortKey === key) {
    sortDir = sortDir === "asc" ? "desc" : "asc";
  } else {
    sortKey = key;
    sortDir = key === "created_at" ? "desc" : "asc";
  }
  paintSortIndicators();
  render();
}

/* --------------------------------- render -------------------------------- */
function formatCell(column, value) {
  if (value === null || value === undefined || value === "") return "\u2014";
  if (column.date) return dateFormat.format(new Date(value));
  if (column.key === "probability") return `${(value * 100).toFixed(1)}%`;
  if (column.numeric) return numberFormat.format(value);
  return String(value);
}

function compareRows(a, b) {
  const column = COLUMNS.find((item) => item.key === sortKey);
  if (column && (column.numeric || column.date)) {
    return Number(a[sortKey]) - Number(b[sortKey]);
  }
  return String(a[sortKey]).localeCompare(String(b[sortKey]), undefined, {
    sensitivity: "base",
  });
}

function render() {
  const query = searchInput.value.trim().toLowerCase();

  const rows = submissions.filter((row) => {
    if (!query) return true;
    return COLUMNS.some((column) =>
      String(row[column.key] ?? "").toLowerCase().includes(query)
    );
  });

  rows.sort(compareRows);
  if (sortDir === "desc") rows.reverse();

  tbody.innerHTML = "";

  if (!rows.length) {
    stateBox.hidden = false;
    stateBox.textContent = submissions.length
      ? "No submissions match your search."
      : "No submissions yet - score a loan application to see it here.";
    return;
  }

  stateBox.hidden = true;

  const fragment = document.createDocumentFragment();

  for (const row of rows) {
    const tr = document.createElement("tr");

    for (const column of COLUMNS) {
      const td = document.createElement("td");

      if (column.key === "prediction") {
        const tag = document.createElement("span");
        tag.className =
          row.prediction === "Approved" ? "tag tag--approved" : "tag tag--rejected";
        tag.textContent = row.prediction;
        td.appendChild(tag);
      } else {
        td.textContent = formatCell(column, row[column.key]);
      }

      tr.appendChild(td);
    }

    fragment.appendChild(tr);
  }

  tbody.appendChild(fragment);
}

function renderStats() {
  const total = submissions.length;
  const approved = submissions.filter((row) => row.prediction === "Approved").length;
  const rejected = total - approved;
  const rate = total ? ((approved / total) * 100).toFixed(1) : "0.0";

  document.getElementById("stat-total").textContent = numberFormat.format(total);
  document.getElementById("stat-approved").textContent = numberFormat.format(approved);
  document.getElementById("stat-rejected").textContent = numberFormat.format(rejected);
  document.getElementById("stat-rate").textContent = `${rate}%`;
}

/* ---------------------------------- load --------------------------------- */
async function load() {
  stateBox.hidden = false;
  stateBox.textContent = "Loading submissions...";
  refreshButton.disabled = true;

  try {
    const response = await fetch(`${window.loanlensApiUrl("/api/history")}?limit=1000`);
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const data = await response.json();
    submissions = data.submissions;
    renderStats();
    render();
  } catch (error) {
    tbody.innerHTML = "";
    stateBox.hidden = false;
    stateBox.textContent = `Could not load history: ${error.message}`;
  } finally {
    refreshButton.disabled = false;
  }
}

searchInput.addEventListener("input", render);
refreshButton.addEventListener("click", load);

window.loanlensWireNavigation(document);
buildHeader();
load();

