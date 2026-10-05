"""Figure 1: the four policies around one compaction, drawn as a schematic (no data, no model calls).

Run from the repository root:
  python3 results/paper-mlsys2027-draft/figures/overview/build_figure.py [--paste]
Columns are policies and rows are phases, read from top to bottom: the context when it crosses the threshold, what
happens while the summary is written, and what the first request after the switch contains. The labels on the left
tie each row to the research question measured there. Numbers of steps are illustrative.
--paste replaces the generated block in main.tex so the manuscript stays self-contained.
Only 7pt and 9pt roman text is used: the offline TeX cache has no 5pt font and no small bold font.
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
BEGIN, END = "% BEGIN generated overview figure", "% END generated overview figure"
W, GAP, X0 = 3.5, 0.12, 3.0          # policy column width, gap, left edge of the first column (cm)
SQ, PITCH = 0.28, 0.33               # step square size and spacing (cm)
SUMMARY_W = 1.25
COLS = [("S", "Sync", "polS"), ("A", "Async", "polA"), ("K", "Sync-Keep", "polK"), ("L", "Limit-only", "polL")]
R1, R2, R3, BAN = (-0.5, -1.2), (-1.45, -2.65), (-2.9, -3.85), (-4.05, -4.7)   # (top, bottom) of each row
CAPTION = (r"The four policies around one compaction, with time running down; numbers of steps are illustrative. "
           r"When the context crosses the threshold, S and K wait while the summary is written, A keeps working, "
           r"and L does not compact until its 1M-token limit. The first request after the switch then sees the "
           r"summary alone (S), the summary followed by the steps taken meanwhile (A) or by the two steps before "
           r"the threshold (K), or, for L, the whole history. The labels on the left mark where each research "
           r"question is measured: the wait, in paired API runs and on one GPU (Section~\ref{sec:wait}); the next "
           r"actions, in full runs and same-state branches (Section~\ref{sec:behavior}); and the repayment, in "
           r"money at API cache prices and in time on one GPU (Section~\ref{sec:cost}).")


def f(v):
    return f"{v:.3f}".rstrip("0").rstrip(".")


def draw():
    out = []
    o = out.append

    def square(x, y, fill="black!12", draw="black!45", lw=None, size=SQ):
        style = f"draw={draw},fill={fill}" + (f",line width={lw}" if lw else "")
        o(f"\\draw[{style}] ({f(x)},{f(y - size / 2)}) rectangle ++({f(size)},{f(size)});")

    def squares(x, y, n, **kw):
        for i in range(n):
            square(x + i * PITCH, y, **kw)

    def note(x, y, text):
        o(f"\\node[text=black!60] at ({f(x)},{f(y)}) {{{text}}};")

    o("\\begin{tikzpicture}[x=1cm,y=1cm,font=\\scriptsize,>=latex]")
    # legend
    lx, ly = X0, 0.42
    for kw, text in [(dict(), "agent step"), (dict(fill="polA!35", draw="polA"), "step taken while A summarizes"),
                     (dict(fill="polK!25", draw="polK", lw="1pt"), "step that K keeps")]:
        square(lx, ly, **kw)
        o(f"\\node[anchor=west,inner sep=0pt] at ({f(lx + SQ + 0.1)},{f(ly)}) {{{text}}};")
        lx += SQ + 0.6 + 0.13 * len(text)
    # time arrow and row labels
    o(f"\\draw[->,black!45] (0.08,{f(R1[0])}) -- (0.08,{f(BAN[1])});")
    o(f"\\node[rotate=90,anchor=south,text=black!55,inner sep=1pt] at (0.08,{f((R1[0] + BAN[1]) / 2)}) {{time}};")
    for (top, bot), text, rq in [(R1, "At the threshold", None), (R2, "While summarizing", "RQ1: the wait"),
                                 (R3, "After the switch", "RQ2: next actions"), (BAN, "Later steps", "RQ3: repaid?")]:
        rq = f"\\\\[2pt]\\textcolor{{black!60}}{{{rq}}}" if rq else ""
        o(f"\\node[anchor=west,align=left,inner sep=0pt] at (0.35,{f((top + bot) / 2)}) {{{text}{rq}}};")
    for c, (p, name, col) in enumerate(COLS):
        x = X0 + c * (W + GAP)
        xc = x + W / 2
        frame = "black" if p == "L" else col
        o(f"\\node[text={col},font=\\small] at ({f(xc)},-0.2) {{{p}: {name}}};")
        for top, bot in (R1, R2, R3):
            o(f"\\draw[{frame}!40,fill={frame}!5,rounded corners=2pt] ({f(x)},{f(top)}) rectangle ({f(x + W)},{f(bot)});")
        for a, b in ((R1[1], R2[0]), (R2[1], R3[0])):
            o(f"\\draw[->,black!45] ({f(xc)},{f(a - 0.02)}) -- ({f(xc)},{f(b + 0.02)});")
        # the same history for every policy; K marks the two steps it keeps
        y1, sx = sum(R1) / 2, xc - (8 * PITCH - (PITCH - SQ)) / 2
        squares(sx, y1, 6 if p == "K" else 8)
        if p == "K":
            squares(sx + 6 * PITCH, y1, 2, fill="polK!25", draw="polK", lw="1pt")
        # while the summary is written
        ytop, ybot = R2[0] - 0.33, R2[1] + 0.33
        box = "black!40,densely dashed" if p == "L" else f"{col},fill={col}!22"
        o(f"\\draw[{box},rounded corners=1.5pt] ({f(xc - 0.85)},{f(ytop - 0.19)}) rectangle ++(1.7,0.38);")
        o(f"\\node{'[text=black!55]' if p == 'L' else ''} at ({f(xc)},{f(ytop)}) "
          f"{{{'no summary' if p == 'L' else 'summarizing'}}};")
        if p in ("S", "K"):
            for dx in (0, 0.1):
                o(f"\\fill[black!60] ({f(xc - 0.8 + dx)},{f(ybot - 0.12)}) rectangle ++(0.055,0.24);")
            o(f"\\node[anchor=west,inner sep=0pt] at ({f(xc - 0.55)},{f(ybot)}) {{agent waits}};")
        else:
            o(f"\\fill[black!60] ({f(xc - 1.42)},{f(ybot - 0.12)}) -- ++(0,0.24) -- ++(0.2,-0.12) -- cycle;")
            o(f"\\node[anchor=west,inner sep=0pt] at ({f(xc - 1.12)},{f(ybot)}) {{keeps working}};")
            squares(xc + 0.68, ybot, 3, **(dict(fill="polA!35", draw="polA") if p == "A" else {}))
        # context of the first request after the switch, with a one-line reading
        y3, yn = R3[0] - 0.36, R3[1] + 0.22
        if p == "L":
            n, size, pitch = 11, 0.22, 0.26
            sx = xc - (n * pitch - (pitch - size)) / 2
            for i in range(n):
                square(sx + i * pitch, y3, size=size)
            note(xc, yn, "whole history")
            continue
        kept = {"S": 0, "A": 3, "K": 2}[p]
        width = SUMMARY_W + (0.1 + kept * PITCH - (PITCH - SQ) if kept else 0)
        gx = xc - width / 2
        o(f"\\draw[{col},fill={col}!15,rounded corners=1.5pt] ({f(gx)},{f(y3 - 0.17)}) rectangle ++({f(SUMMARY_W)},0.34);")
        o(f"\\node[inner sep=0pt] at ({f(gx + SUMMARY_W / 2)},{f(y3)}) {{summary}};")
        if kept:
            squares(gx + SUMMARY_W + 0.1, y3, kept,
                    **(dict(fill="polA!35", draw="polA") if p == "A" else dict(fill="polK!25", draw="polK", lw="1pt")))
        note(xc, yn, {"S": "summary only", "A": "summary + new steps", "K": "summary + kept steps"}[p])
    # later steps and repayment
    right = X0 + 4 * W + 3 * GAP
    o(f"\\draw[black!35,fill=black!4,rounded corners=2pt] ({f(X0)},{f(BAN[0])}) rectangle ({f(right)},{f(BAN[1])});")
    o(f"\\node[text width=14.1cm,align=center] at ({f((X0 + right) / 2)},{f(sum(BAN) / 2)}) "
      "{Later steps carry this context until the next summary. A summary pays if cumulative savings before the next "
      "summary exceed the cost of summarizing and switching.};")
    o("\\end{tikzpicture}")
    return out


def write_tex():
    lines = [BEGIN, "% Generated by figures/overview/build_figure.py (schematic, no data).",
             "\\begin{figure*}[t]", "\\centering", *draw(), f"\\caption{{{CAPTION}}}", "\\label{fig:overview}",
             "\\end{figure*}", END]
    tex = "\n".join(lines) + "\n"
    (HERE / "overview-tikz.tex").write_text(tex)
    return tex


def paste(tex):
    main = PAPER / "main.tex"
    s = main.read_text()
    if BEGIN not in s or END not in s:
        raise SystemExit("markers not found in main.tex")
    a, b = s.index(BEGIN), s.index(END) + len(END) + 1
    main.write_text(s[:a] + tex + s[b:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paste", action="store_true")
    args = parser.parse_args()
    tex = write_tex()
    if args.paste:
        paste(tex)
    print(f"wrote {HERE / 'overview-tikz.tex'}")


if __name__ == "__main__":
    main()
