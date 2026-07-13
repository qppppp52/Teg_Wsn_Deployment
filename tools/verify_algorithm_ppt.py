from __future__ import annotations

import json
import os
from pathlib import Path

import win32com.client


def main():
    ppt_path = Path(os.environ["ALGORITHM_PPT_OUTPUT"]).resolve()
    preview_dir = Path(__file__).resolve().parents[1] / ".test_outputs" / "algorithm_ppt_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    app = win32com.client.DispatchEx("PowerPoint.Application")
    app.Visible = True
    pres = app.Presentations.Open(str(ppt_path), WithWindow=False)
    selected = [1, 13, 24, 43, 49]
    exported = []
    for index in selected:
        out = preview_dir / f"slide_{index:02d}.png"
        pres.Slides(index).Export(str(out), "PNG", 1920, 1080)
        exported.append({"slide": index, "path": str(out), "bytes": out.stat().st_size})
    result = {
        "slides": pres.Slides.Count,
        "shapes": sum(slide.Shapes.Count for slide in pres.Slides),
        "exported": exported,
    }
    pres.Close()
    app.Quit()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
