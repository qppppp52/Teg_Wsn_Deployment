from __future__ import annotations

import json
import os
from pathlib import Path

import win32com.client


def main():
    path = Path(os.environ["ALGORITHM_PPT_OUTPUT"]).resolve()
    app = win32com.client.DispatchEx("PowerPoint.Application")
    app.Visible = True
    pres = app.Presentations.Open(str(path), WithWindow=False)
    overflows = []
    out_of_bounds = []
    for slide in pres.Slides:
        for shape in slide.Shapes:
            if shape.Left < -1 or shape.Top < -1 or shape.Left + shape.Width > 961 or shape.Top + shape.Height > 541:
                out_of_bounds.append([slide.SlideIndex, shape.Name])
            try:
                if not shape.HasTextFrame or not shape.TextFrame.HasText:
                    continue
                frame = shape.TextFrame
                available_h = shape.Height - frame.MarginTop - frame.MarginBottom
                available_w = shape.Width - frame.MarginLeft - frame.MarginRight
                bound_h = frame.TextRange.BoundHeight
                bound_w = frame.TextRange.BoundWidth
                if bound_h > available_h + 3 or bound_w > available_w + 3:
                    overflows.append({
                        "slide": slide.SlideIndex,
                        "shape": shape.Name,
                        "bound_h": round(bound_h, 1),
                        "available_h": round(available_h, 1),
                        "bound_w": round(bound_w, 1),
                        "available_w": round(available_w, 1),
                    })
            except Exception:
                pass
    pres.Close()
    app.Quit()
    print(json.dumps({
        "text_overflows": overflows,
        "out_of_bounds": out_of_bounds,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
