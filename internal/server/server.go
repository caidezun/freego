// Package server wires up the freego HTTP routes and handlers.
package server

import (
	"encoding/json"
	"net/http"
	"time"
)

// New returns an http.Handler with all freego routes registered.
func New() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", handleRoot)
	mux.HandleFunc("GET /healthz", handleHealthz)
	mux.HandleFunc("GET /api/hello", handleHello)
	return mux
}

func handleRoot(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(indexHTML))
}

func handleHealthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok",
		"time":   time.Now().UTC().Format(time.RFC3339),
	})
}

func handleHello(w http.ResponseWriter, r *http.Request) {
	name := r.URL.Query().Get("name")
	if name == "" {
		name = "world"
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"message": "hello, " + name + "!",
	})
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

const indexHTML = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>freego</title>
  <style>
    :root { color-scheme: light dark; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0; min-height: 100vh; display: grid; place-items: center;
      background: radial-gradient(1200px 600px at 50% -10%, #1e3a8a22, transparent), #0b1020;
      color: #e6edf3;
    }
    .card {
      background: #131a2ecc; border: 1px solid #2a3a5f; border-radius: 16px;
      padding: 40px 48px; box-shadow: 0 20px 60px #0008; text-align: center; max-width: 520px;
    }
    h1 { margin: 0 0 8px; font-size: 40px; letter-spacing: -1px; }
    p { margin: 4px 0; color: #9fb0c7; }
    code { background: #0b1020; border: 1px solid #2a3a5f; border-radius: 6px; padding: 2px 6px; color: #7ee787; }
    a { color: #79c0ff; }
  </style>
</head>
<body>
  <div class="card">
    <h1>freego</h1>
    <p>A minimal Go HTTP service.</p>
    <p>Try <a href="/api/hello?name=freego"><code>/api/hello?name=freego</code></a></p>
    <p>Health check at <a href="/healthz"><code>/healthz</code></a></p>
  </div>
</body>
</html>
`
