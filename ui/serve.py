"""
Static file server for the Fantasy Draft Assistant UI that disables browser
caching entirely.

Why this exists: `python -m http.server` does not send any Cache-Control
headers, so browsers can serve a stale cached copy of app.js/style.css even
after a refresh (sometimes even after a hard refresh, depending on the
browser and how the page was reached). This wrapper subclasses
SimpleHTTPRequestHandler to force `Cache-Control: no-store` on every
response, so the browser always fetches the current files from disk.

Usage (run from the ui/ directory):
    python serve.py            # serves on port 8765
    python serve.py 8080       # serves on a custom port

Use this INSTEAD OF `python -m http.server 8765` going forward.

Note: this no-cache header is a second layer of defense. The primary fix for
stale JS/CSS is the "?v=..." version query string on the <script>/<link>
tags in index.html (see the comment there) — that version string must be
bumped on every change to app.js or style.css.
"""
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main():
    port = 8765
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]!r}")
            sys.exit(1)

    server = HTTPServer(("", port), NoCacheHandler)
    print(f"Serving ui/ with no-cache headers at http://localhost:{port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.shutdown()


if __name__ == "__main__":
    main()
