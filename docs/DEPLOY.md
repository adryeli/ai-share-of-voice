# Deploying the demo

GitHub Pages cannot host this app: it only serves static files, and the app is a Python
server (Gradio, SQLite checkpoints, background runs). It needs a host that runs Python.
Render is the free option described here; Hugging Face Spaces also works but hosting a
Gradio app there on free CPU requires a PRO subscription.

## What protects the model budget

- `DEMO_PASSWORD` is a secret of the host. In live mode the app asks for it before any call
  that spends budget: loading the proposed questions and starting a measurement.
- Five wrong guesses in one minute lock the gate for everyone until the minute passes, so the
  password cannot be brute-forced.
- The simulated demo needs no password because it calls no model.
- `SPEND_CAP_USD` limits **each measurement**, not the total. Keep it small and rotate the
  provider key if the password leaks.
- Never write the password or any key in the repository (it is public). Set them only as
  secrets of the host.

## Render (free)

1. Create an account at <https://render.com> and sign in with GitHub, allowing it to read your
   fork of this repository.
2. *New → Blueprint*, pick the fork and the branch that contains `render.yaml`. Render reads
   the service from that file (Python 3.12, `pip install -r requirements.txt`, `python app.py`).
3. When asked for the secrets, leave them empty for the free simulated demo. For real
   measurements set, in the service's *Environment* tab:

   | Kind | Name | Value |
   | --- | --- | --- |
   | Secret | `DEMO_PASSWORD` | the shared password |
   | Secret | `LLM_API_KEY` | your provider key |
   | Variable | `LLM_MODE` | `live` |
   | Variable | `SEARCH_PROVIDER` | `duckduckgo` (or `none` for fixtures) |

4. Deploy. The address is shown at the top of the service, like
   `https://ai-share-of-voice.onrender.com`.

The app reads `GRADIO_SERVER_NAME` and `GRADIO_SERVER_PORT` and listens there; `render.yaml`
sets them.

## Limits to know

- The disk of a free service is not persistent: `runs/` and the history are lost on restart.
- A free service sleeps after about 15 minutes without visits; the first visit then takes
  around a minute to wake it.
- A free service has little memory (512 MB). If the build or the app is killed for memory, use
  a larger plan or another host.
- Search-service quotas and the provider's own limits are separate from `SPEND_CAP_USD`.

## Hugging Face Spaces (needs PRO for Gradio)

Create a Gradio Space and put this header at the top of its `README.md`, on a throwaway
branch so the project README stays clean:

```yaml
---
title: AI Share of Voice
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
---
```

Set the same secrets and variables in *Settings → Variables and secrets* and push that branch
to the Space's `main`. The Space sets `GRADIO_SERVER_NAME` and `GRADIO_SERVER_PORT` itself.
