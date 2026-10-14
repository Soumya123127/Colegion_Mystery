# Local API

The workshop backend is `scripts/server.py`, using Python's standard-library `ThreadingHTTPServer` bound to `127.0.0.1`. OpenCV operations and clue/report services are implemented in `src/the_last_chai`; see the root README for setup/startup. Endpoints are `GET /api/status`, `POST /api/mission/{eyes-online,missing-clue,red-bag,follow-trail}`, `POST /api/clue`, `POST /api/accusation`, and `POST /api/report`. The server is intended for local workshop use, not public deployment; it has no authentication, TLS, or production hardening.
