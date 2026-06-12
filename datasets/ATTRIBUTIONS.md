# Dataset Attributions

The benchmark datasets include MusicXML excerpts derived from public-domain
classical compositions. The copyright status of the underlying composition is
separate from the license or status of a particular MusicXML transcription,
edition, annotation layer, or benchmark-derived target file.

File-level source and rights metadata is listed in `datasets/sources.csv`.
Where source MusicXML files already contain `<creator>`, `<source>`, or
`<rights>` metadata, those embedded notices are preserved.

## Long-Task Reconstruction

All long-task reconstruction cases except the Beethoven Symphony No. 5 case are
based on files from the music21 core corpus:

- `music21/corpus/beethoven/opus59no1/movement4.mxl`
- `music21/corpus/haydn/opus74no1/movement2.mxl`
- `music21/corpus/mozart/k155/movement1.mxl`
- `music21/corpus/mozart/k156/movement1.mxl`
- `music21/corpus/mozart/k545/movement1_exposition.mxl`

The music21 project notes in `music21/corpus/license.txt` that the BSD license
for music21's Python code does not apply to corpus files. The corpus files are
distributed with permission of their encoders and, where needed, composers or
arrangers; music21 also notes that some corpus encodings may have commercial or
other restrictions. The included ScoreSpeak files should therefore be cited as
music21 corpus excerpts unless a more specific embedded source or rights notice
is present.

The Beethoven Symphony No. 5 long-task case preserves embedded MusicXML
metadata indicating `Score: CC0 1.0 Universal; Annotations: CC-By-SA` and
`IMSLP #575952`.

## Precise-Edit Base Scores

The precise-edit benchmark base scores preserve embedded MusicXML metadata from
their sources. Current embedded notices identify OpenScore, MuseScore, IMSLP,
and CC0 or CC-BY-SA-related rights where available. See `datasets/sources.csv`
for file-level details.

## Dataset License Notes

- CC0 files may be reused without attribution, but attribution is provided here
  for provenance.
- Files or annotations marked with CC-BY-SA components require attribution and
  compatible sharing of adaptations.
- Dataset file rights may differ from the ScoreSpeak source-code license.
