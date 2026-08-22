# Azure App Service deployment notes — analyzeapi

This file exists because a missing Startup Command and/or missing Azure OpenAI
environment variables previously took the whole app down at boot with:

```
Container did not respond to startup probe on port 8000 within the expected
time limit of 230s. No listening ports were detected in the container.
```

That happens whenever the Python process crashes *before* gunicorn/uvicorn
binds a port — e.g. a bad import or a client library that raises at
construction time. Use this checklist after any redeploy or config change.

## 1. Startup Command

Azure Portal → App Service (`analyzeapi`) → **Settings → Configuration →
General settings → Startup Command**

Set it explicitly (don't leave it blank — Oryx's auto-detection can pick the
wrong entrypoint for this layout, since the FastAPI `app` object lives in
`app/main.py` inside a package that's also named `app`). Critically, always
include `--bind=0.0.0.0:8000` — gunicorn's default bind is `127.0.0.1:8000`
(loopback only), which is invisible to Azure's health-check probe and is
what produces "No listening ports were detected in the container":

```
python -m gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app --bind=0.0.0.0:8000 --timeout 180 --access-logfile /dev/null
```

`--timeout 180` is a safety net for a fully wedged worker process, not a
"normal" wait time -- `UvicornWorker` heartbeats independently of any single
in-flight request. The actual worst-case latency a user experiences is
bounded by the Azure OpenAI call settings in `app/openai_service_azure.py`
(client `timeout=25.0`, 3 retry attempts ~= 81s worst case), which is why
180s here is chosen to sit comfortably above that without ever being the
active bottleneck. See section 4 below.

## 2. Environment variables

Azure Portal → App Service (`analyzeapi`) → **Settings → Environment
variables → App settings**

These are required — `app/openai_service_azure.py` needs them at runtime
(not just in a local `.env`, which is never deployed):

| Name | Required | Notes |
|---|---|---|
| `AZURE_OPENAI_API_KEY` | Yes | From the Azure OpenAI resource, Keys and Endpoint blade |
| `AZURE_OPENAI_ENDPOINT` | Yes | e.g. `https://<resource-name>.openai.azure.com/` |
| `AZURE_OPENAI_API_VERSION` | No | Defaults to `2024-12-01-preview` if unset |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | No | Defaults to `gpt-4o` if unset |

As of the current code, a missing `AZURE_OPENAI_API_KEY` or
`AZURE_OPENAI_ENDPOINT` no longer crashes the whole app at startup — the
client is built lazily on the first call to `/analyze/`, and a missing
credential returns a `500` from that endpoint instead of failing the health
check. But the app should still be configured correctly for `/analyze/` to
actually work.

## 3. Confirming a startup failure

If the health check fails again, check the actual traceback before guessing:

Azure Portal → App Service → **Log stream**, or **Diagnose and solve
problems → Application Logs**, or the Kudu console:
`https://analyzeapi.scm.azurewebsites.net/api/logs/docker`

Look at the last error logged right before the probe timeout — it names the
real cause directly (import error, missing package, bad credential, etc.)
rather than needing to guess from the generic "no listening ports" message.

## 4. Request latency / timeout budget

`app/openai_service_azure.py` calls Azure OpenAI on every `/analyze/`
request. Two settings there bound how long a user can end up waiting:

- The `AzureOpenAI` client is constructed with `timeout=25.0` (seconds per
  attempt). The SDK's own default is a 10-minute per-call timeout, which
  would otherwise let a single hung attempt run far longer than Azure App
  Service's own front-end request timeout allows.
- `call_openai_api` retries up to 3 times (`stop_after_attempt(3)`) with a
  2s wait between attempts, but does **not** retry on `AuthenticationError`,
  `BadRequestError`, or `AzureOpenAINotConfigured` -- those fail immediately.

Worst case for a hanging backend: 3 x (25s + 2s) ~= 81 seconds before the
endpoint returns a clean 500, well under Azure App Service (Linux)'s fixed
~240 second front-end/load-balancer timeout -- which is not configurable
and will kill the connection with its own generic timeout error if a
request runs longer than that, regardless of any app-level setting.

If either the client timeout or the retry count is changed later, re-check
that attempts x (timeout + wait) still leaves headroom under ~240s.

## 5. Known layout gotcha

The FastAPI app object is `app/main.py:app`, inside a package directory also
named `app` (with an empty `app/__init__.py`). This is valid, but it's easy
for tooling (Oryx auto-detect, an IDE, a copy-pasted Procfile) to guess
`app:app` instead of `app.main:app` and silently import the wrong (or empty)
module. Always reference it as `app.main:app` in startup commands.
