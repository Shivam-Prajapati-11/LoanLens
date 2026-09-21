/**
 * Runtime configuration for the LoanLens frontend.
 *
 * The API calls in script.js / history.js are built from `window.LOANLENS_API_BASE`:
 *
 *   - "" (default) -> same origin. Use this whenever the FastAPI app serves the
 *     frontend too (locally, in Docker, and on the Render Web Service).
 *   - Any absolute URL -> the frontend and the API live on different hosts.
 *     Example: the UI on Vercel + the API on Render:
 *
 *       window.LOANLENS_API_BASE = "https://loanlens-api.onrender.com";
 *
 *     No trailing slash. Remember to add the frontend's origin to the API's
 *     ALLOWED_ORIGINS environment variable, otherwise the browser blocks the
 *     request with a CORS error.
 */
window.LOANLENS_API_BASE = "https://loan-approval-prediction-qsnu.onrender.com";

/**
 * Build an absolute URL for a path served by the API.
 *
 * @param {string} path Application path, e.g. "/predict".
 * @returns {string} Either "/predict" (same origin) or the full URL.
 */
window.loanlensApiUrl = function loanlensApiUrl(path) {
  const base = String(window.LOANLENS_API_BASE || "").replace(/\/+$/, "");
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${base}${suffix}`;
};

/**
 * Point the top navigation at whichever host owns the data.
 *
 * With a split deployment (Vercel UI + Render API) the pages live on Vercel but
 * ``/`` and ``/history`` are served by the API on Render.
 *
 * @param {Document} document The page being rendered.
 */
window.loanlensWireNavigation = function loanlensWireNavigation(document) {
  const base = String(window.LOANLENS_API_BASE || "").replace(/\/+$/, "");
  if (!base) return;

  for (const link of document.querySelectorAll(".brand, .navlink")) {
    const target = link.getAttribute("href");
    if (target === "/" || target === "/history") {
      link.href = window.loanlensApiUrl(target);
    }
  }
};
