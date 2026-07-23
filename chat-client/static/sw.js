/**
 * Service worker for LLM Search Chat PWA.
 *
 * Provides:
 *  - Installability (browser requires a SW to trigger "Add to Home Screen")
 *  - Offline fallback: caches the app shell on first load, serves from cache
 *    when the network is unavailable.
 */

const CACHE_NAME = 'llm-search-v1';

// Static assets that make up the "app shell"
const SHELL_ASSETS = [
  '/',
  '/static/index.html',
  '/static/app.js',
  '/static/style.css',
  '/static/icon.svg',
  '/static/manifest.json',
];

// ── Install: pre-cache shell assets ───────────────────────────────

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_ASSETS))
  );
  // Activate immediately — don't wait for old tabs to close
  self.skipWaiting();
});

// ── Activate: clean old caches ────────────────────────────────────

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      )
    )
  );
  // Take control of all clients immediately
  self.clients.claim();
});

// ── Fetch: cache-first for shell, network-first for API ───────────

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Don't cache API / streaming requests — let them pass through
  if (url.pathname.startsWith('/v1/') || url.pathname === '/health' || url.pathname === '/stats') {
    return;
  }

  // Cache-first for static shell assets
  event.respondWith(
    caches.match(event.request).then((cached) => {
      // Return cached response immediately; update cache in background
      const fetchPromise = fetch(event.request)
        .then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(() => null);

      return cached || fetchPromise;
    })
  );
});
