# 0014 - Search results come with a video of the person

- **Status:** accepted
- **Date:** 2026-11-12

## Context

The final-year pipeline answered a photo query with a video: the frames where
the target was found, the person boxed (`output_annotated.mp4`) or cropped
(`output.mp4`). The API (decision 0012) answered with track IDs and frame
numbers only, and deleted each uploaded video once it was indexed, so there was
nothing to cut a clip from.

## Decision

- **Keep the original of every indexed video** as `<video_dir>/<video_id>.<ext>`
  (`/data/videos` in the container). Uploading a video that is already indexed
  stores it without indexing it again, which is also how the Week 2 videos get
  their originals onto a server (`make api-videos` does it locally).
  `PS_API__KEEP_VIDEOS=false` turns this off: no footage on the server, no clips.
- **`clip=true` on `POST /search`** cuts the best result into a video and returns
  its `clip_url`. For a photo query it holds exactly the frames the paper's rule
  selected in each matched video (the same frames as the golden test), so it is
  the final-year output, cut to the matches. For a description it is the top track.
- **Every result track** has `start_s` / `end_s` and a `clip_url`:
  `GET /videos/{video_id}/tracks/{track_id}/clip` renders every frame of that
  track, `boxed` (full frame, person boxed, time in the video) or `crop` (the
  person only, 128 x 256).
- **H.264 MP4, 12 fps** (the sampled stream, so it plays in real time), at most
  960 px wide, encoded by imageio-ffmpeg's bundled ffmpeg (libx264, `veryfast`,
  CRF 23, `+faststart`). OpenCV's pip wheels can only write MPEG-4 part 2, which
  browsers do not play. The clip URL opens in a browser tab.
- **Clips are cached by content** (`<video_id>_<track or digest of frames>_<style>.mp4`
  in `/data/clips`, oldest pruned beyond 200): the same result renders once.
- A clip that cannot be made (original not stored, unknown track) is a 404 with
  the reason; inside a search it becomes a `notes` entry, never a failed search.

## Results (docs/results/week3_api.md, "Result clips")

Test case 4, real models: `clip=true` returned an H.264 clip of exactly the 62
frames the golden run selects (tracks 7 and 42), made in 3.2 s; track 7's crop
clip in 0.6 s, 4 ms when cached. Same in the Docker image.

## Consequences

- The server now holds footage of real people: disk space, and consent and
  access control before a public demo (an API key at least).
- For the online demo the 4 test videos (~134 MB) must reach the server too:
  upload them through `/ingest` once, or add them to the assets repo (step 4).
- Rendering decodes the video up to the last result frame: a few seconds for a
  long video, once per result.
