# ScoreSpeak Web Editor

The web editor renders MusicXML with OpenSheetMusicDisplay and sends natural
language edit requests to the ScoreSpeak agent.

## Install

From the repository root:

```bash
pip install -e ".[dev]"
```

Or from this directory:

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` before using chat or voice editing.

## Run

```bash
python server.py
```

Open `http://localhost:5001`.

## Main Endpoints

- `POST /api/new`: create a new score.
- `POST /api/load`: upload a MusicXML, XML, or MXL file.
- `POST /api/chat`: send a ScoreSpeak agent edit request.
- `POST /api/chat/stream`: stream ScoreSpeak agent progress.
- `GET /api/musicxml`: fetch the current MusicXML.
- `GET /api/status`: fetch session status.
