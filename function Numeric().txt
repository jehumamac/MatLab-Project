import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import re

# ─────────────────────────── Modernized Colour Palette ───────────────────────
C = {
    "bg":         "#F8FAFC",  # Slate 50 (Ultra clean background)
    "card":       "#FFFFFF",  # Pure White Cards
    "primary":    "#0F172A",  # Slate 900 (Deep modern contrast tech header)
    "primary_d":  "#1E293B",  # Slate 800
    "accent":     "#2563EB",  # Royal Blue Accent
    "accent_l":   "#DBEAFE",  # Light Blue Tint
    "warn":       "#D97706",  # Amber 600
    "error":      "#DC2626",  # Red 600
    "border":     "#E2E8F0",  # Slate 200 (Clean structural lines)
    "text":       "#0F172A",  # Slate 900
    "sub":        "#64748B",  # Slate 500 (Subdued text)
    "row_green":  "#F0FDF4",  # Mint Tint
    "row_yellow": "#FEFCE8",  # Soft Yellow Tint
    "row_blue":   "#EFF6FF",  # Soft Blue Tint
    "row_red":    "#FEF2F2",  # Soft Red Tint
}

FONT       = ("Segoe UI", 10)
FONT_B     = ("Segoe UI", 10, "bold")
FONT_LG    = ("Segoe UI", 12, "bold")
FONT_TITLE = ("Segoe UI", 15, "bold")
FONT_MONO  = ("Consolas", 10)


# ══════════════════════════════════════════════════════════════════════════════
#   Helper: evaluate user expression safely
# ══════════════════════════════════════════════════════════════════════════════
def make_func(expr_str: str):
    """
    Turn a string like 'x^3 - x - 2' into a callable f(x).
    Supports: ^  sin cos tan exp log sqrt abs pi
    """
    s = expr_str.strip()
    s = re.sub(r'\^', '**', s)                     # x^n → x**n
    s = re.sub(r'(\d)(x)', r'\1*x', s)             # 2x  → 2*x
    allowed = {
        'x': None, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt,
        'abs': abs, 'pi': math.pi, 'e': math.e,
    }
    def f(x):
        env = dict(allowed)
        env['x'] = x
        return eval(s, {"__builtins__": {}}, env)
    f(1.0)           # quick validation
    return f, s


def num_deriv(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)


# ══════════════════════════════════════════════════════════════════════════════
#   Numerical Methods
# ══════════════════════════════════════════════════════════════════════════════
def solve_incremental(f, a, b, tol, max_it):
    dx0 = (b - a) / 10
    rows, xl, dx = [], a, dx0
    for _ in range(max_it * 30):
        xu = min(xl + dx, b)
        fxl, fxu = f(xl), f(xu)
        prod = fxl * fxu
        if prod > 0:
            rows.append((xl, dx, xu, fxl, fxu, prod, "Go to next interval", "green"))
            xl, dx = xu, dx0
        elif prod < 0:
            rows.append((xl, dx, xu, fxl, fxu, prod, "Revert — smaller interval", "yellow"))
            dx /= 10
            if dx < tol:
                return xl + dx, True, rows
        else:
            rows.append((xl, dx, xu, fxl, fxu, prod, "Exact root", "blue"))
            return xu, True, rows
        if len(rows) >= max_it * 20:
            break
    return xl, True, rows


def solve_bisection(f, a, b, tol, max_it):
    if f(a) * f(b) >= 0:
        raise ValueError(f"f(a)·f(b) must be negative.\nf({a:.4f})={f(a):.4f}  f({b:.4f})={f(b):.4f}")
    rows = []
    for i in range(1, max_it + 1):
        c = (a + b) / 2; fc = f(c)
        rows.append((i, a, b, c, fc, abs(b - a)))
        if abs(fc) < tol or abs(b - a) < tol:
            return c, True, rows
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    return c, False, rows


def solve_regula_falsi(f, a, b, tol, max_it):
    if f(a) * f(b) >= 0:
        raise ValueError(f"f(a)·f(b) must be negative.  f({a:.4f})={f(a):.4f}  f({b:.4f})={f(b):.4f}")
    rows, xL, xU, xR_prev = [], a, b, None
    for i in range(1, max_it + 1):
        fL, fU = f(xL), f(xU)
        xR = xU - fU * (xL - xU) / (fL - fU)
        fR = f(xR)
        ea = "" if xR_prev is None else abs((xR - xR_prev) / xR) * 100
        prod = fL * fR
        if prod < 0:
            remark, color = "x_R = x_U", "green"
        elif prod > 0:
            remark, color = "x_R = x_L", "yellow"
        else:
            remark, color = "Exact root", "blue"
        rows.append((i, xL, xU, xR, ea, fL, fU, fR, prod, remark, color))
        if prod < 0:
            xU = xR
        elif prod > 0:
            xL = xR
        else:
            return xR, True, rows
        if xR_prev is not None and isinstance(ea, float) and ea < tol * 100:
            return xR, True, rows
        if abs(fR) < tol:
            return xR, True, rows
        xR_prev = xR
    return xR_prev if xR_prev else (a + b) / 2, False, rows


def solve_newton(f, x0, tol, max_it):
    rows, x = [], x0
    for i in range(1, max_it + 1):
        fx = f(x); dfx = num_deriv(f, x)
        if abs(dfx) < 1e-14:
            raise ValueError("Derivative is near zero — Newton-Raphson stopped.")
        xn = x - fx / dfx
        rows.append((i, x, fx, dfx, xn, abs(xn - x)))
        if abs(xn - x) < tol:
            return xn, True, rows
        x = xn
    return x, False, rows


def solve_secant(f, x0, x1, tol, max_it):
    rows = []
    for i in range(1, max_it + 1):
        f0, f1 = f(x0), f(x1)
        if abs(f1 - f0) < 1e-14:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        rows.append((i, x0, x1, f0, f1, x2, abs(x2 - x1)))
        if abs(x2 - x1) < tol:
            return x2, True, rows
        x0, x1 = x1, x2
    return x1, False, rows


# ══════════════════════════════════════════════════════════════════════════════
#   Matrix helpers
# ══════════════════════════════════════════════════════════════════════════════
def adjoint(M):
    n = M.shape[0]
    C = np.zeros_like(M, dtype=float)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(M, i, 0), j, 1)
            C[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
    return C.T


def fmt_matrix(label, M):
    lines = [f"  {label}:", ""]
    for row in M:
        lines.append("    │  " + "  ".join(f"{v:12.6f}" for v in row) + "  │")
    lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#   Main Application
# ══════════════════════════════════════════════════════════════════════════════
class NumericLabApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("NumericLab Solver Pro")
        self.geometry("1180x780")  # Fixed bad URL-style geometry specifier
        self.minsize(1020, 700)
        self.configure(bg=C["bg"])
        self._style()
        self._build_ui()

    # ── ttk style ─────────────────────────────────────────────────────────
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".",            background=C["bg"],  foreground=C["text"],   font=FONT)
        s.configure("TFrame",       background=C["bg"])
        s.configure("Card.TFrame",  background=C["card"], relief="solid", borderwidth=1, darkcolor=C["border"])
        s.configure("TLabel",       background=C["bg"],  foreground=C["text"])
        s.configure("Card.TLabel",  background=C["card"])
        s.configure("TNotebook",    background=C["bg"],     borderwidth=0)
        s.configure("TNotebook.Tab", background=C["border"], foreground=C["sub"], padding=[16, 8], font=FONT_B)
        s.map("TNotebook.Tab",
              background=[("selected", C["accent"])],
              foreground=[("selected", "white")])
        s.configure("Primary.TButton",
                    background=C["accent"], foreground="white",
                    font=FONT_B, padding=[16, 8], relief="flat", borderwidth=0)
        s.map("Primary.TButton",
              background=[("active", C["primary_d"]), ("pressed", C["primary_d"])])
        s.configure("Secondary.TButton",
                    background=C["card"], foreground=C["text"], relief="solid", borderwidth=1, font=FONT, padding=[12, 7])
        s.configure("TCombobox",    fieldbackground=C["card"], background=C["card"], arrowcolor=C["text"])
        s.configure("TEntry",       fieldbackground=C["card"], bordercolor=C["border"])
        s.configure("Treeview",     background=C["card"], fieldbackground=C["card"], rowheight=24, font=FONT_MONO, borderwidth=0)
        s.configure("Treeview.Heading", background=C["primary"], foreground="white", font=FONT_B, relief="flat", padding=5)
        s.map("Treeview", background=[("selected", C["accent"])])

    # ── top-level layout ──────────────────────────────────────────────────
    def _build_ui(self):
        # ── header ──
        hdr = tk.Frame(self, bg=C["primary"], height=64)
        hdr.pack(fill="x")
        tk.Label(hdr, text="NumericLab Solver", font=FONT_TITLE,
                 bg=C["primary"], fg="white").pack(side="left", padx=24, pady=14)
        tk.Label(hdr, text="Numerical Methods & Matrix Analysis Engine",
                 font=("Segoe UI", 9), bg=C["primary"], fg=C["sub"]).pack(side="left", pady=20)

        # ── notebook ──
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=16, pady=(12, 16))

        rf_tab = ttk.Frame(nb, style="TFrame")
        mx_tab = ttk.Frame(nb, style="TFrame")
        nb.add(rf_tab, text="  Root Finding Engine  ")
        nb.add(mx_tab, text="  Matrix Operations  ")

        self._build_root_tab(rf_tab)
        self._build_matrix_tab(mx_tab)

    # ══════════════════════════════════════════════════════════════════════
    #   ROOT FINDING TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_root_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)

        # ── row 0 — equation + method ──
        top = ttk.Frame(parent, style="Card.TFrame", padding=16)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=(12, 6))
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="f(x) = 0", font=FONT_LG, style="Card.TLabel", foreground=C["accent"]).grid(row=0, column=0, sticky="w", padx=(0,12))
        self.rf_eq = ttk.Entry(top, font=("Consolas", 12))
        self.rf_eq.insert(0, "x**3 - x - 2")
        self.rf_eq.grid(row=0, column=1, sticky="ew", ipady=5)
        ttk.Label(top, text="Method:", style="Card.TLabel", font=FONT_B).grid(row=0, column=2, padx=(18, 6))
        self.rf_method = ttk.Combobox(top, width=20, state="readonly", font=FONT,
                                       values=["Incremental","Bisection","Regula-Falsi",
                                              "Newton-Raphson","Secant"])
        self.rf_method.current(0)
        self.rf_method.grid(row=0, column=3, ipady=2)
        self.rf_method.bind("<<ComboboxSelected>>", self._rf_toggle_params)

        hint = "Syntax guidelines:  x**n  |  sin(x)  cos(x)  tan(x)  exp(x)  log(x)  sqrt(x)  abs(x)  |  pi  e"
        ttk.Label(top, text=hint, foreground=C["sub"], font=("Segoe UI", 9),
                  style="Card.TLabel").grid(row=1, column=0, columnspan=4, sticky="w", pady=(8,0))

        # ── row 1 — parameters ──
        par = ttk.Frame(parent, style="Card.TFrame", padding=(16, 12))
        par.grid(row=1, column=0, sticky="ew", padx=8, pady=4)

        def lbl(text, col, padl=0):
            return ttk.Label(par, text=text, style="Card.TLabel", font=FONT_B)

        def ent(default, col, width=9):
            e = ttk.Entry(par, width=width, font=FONT_MONO)
            e.insert(0, str(default))
            e.grid(row=0, column=col, padx=(0, 16), ipady=4)
            return e

        self._rf_a_lbl = lbl("a (lower):", 0); self._rf_a_lbl.grid(row=0, column=0, sticky="w", padx=(0,4))
        self.rf_a    = ent(-1,   1)
        self._rf_b_lbl = lbl("b (upper):", 2); self._rf_b_lbl.grid(row=0, column=2, sticky="w", padx=(0,4))
        self.rf_b    = ent(2,    3)
        self._rf_x0_lbl = lbl("x₀ (init):", 4); self._rf_x0_lbl.grid(row=0, column=4, sticky="w", padx=(0,4))
        self.rf_x0   = ent(1,    5)
        
        lbl("Tolerance:", 6).grid(row=0, column=6, sticky="w", padx=(0,4))
        self.rf_tol  = ent(1e-6, 7, 12)
        lbl("Max Iter:", 8).grid(row=0, column=8, sticky="w", padx=(0,4))
        self.rf_maxn = ent(50,   9, 7)
        
        ttk.Button(par, text="  RUN SOLVER  ", style="Primary.TButton",
                   command=self._rf_solve).grid(row=0, column=10, padx=(8, 0))

        self._rf_toggle_params(None)

        # ── row 2 — result banner ──
        self.rf_result_var = tk.StringVar(value="Enter configuration properties, then click Run Solver.")
        self.rf_result_lbl = tk.Label(parent, textvariable=self.rf_result_var,
                                      font=("Segoe UI", 11, "bold"),
                                      bg=C["bg"], fg=C["sub"], anchor="w", padx=12, pady=6)
        self.rf_result_lbl.grid(row=2, column=0, sticky="ew", padx=8)

        # ── row 3 — table + plot layout (Perfect 50/50 Split) ──
        bot = ttk.Frame(parent)
        bot.grid(row=3, column=0, sticky="nsew", padx=8, pady=(4, 12))
        bot.columnconfigure(0, weight=1, uniform="group1")  # Clean half-and-half configuration
        bot.columnconfigure(1, weight=1, uniform="group1")
        bot.rowconfigure(0, weight=1)

        # table container pane
        tbl_frame = ttk.Frame(bot, style="Card.TFrame")
        tbl_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tbl_frame.rowconfigure(0, weight=1)
        tbl_frame.columnconfigure(0, weight=1)

        self.rf_tree = ttk.Treeview(tbl_frame, show="headings", selectmode="browse")
        vsb = ttk.Scrollbar(tbl_frame, orient="vertical",   command=self.rf_tree.yview)
        hsb = ttk.Scrollbar(tbl_frame, orient="horizontal", command=self.rf_tree.xview)
        self.rf_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.rf_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        self.rf_tree.tag_configure("green",  background=C["row_green"])
        self.rf_tree.tag_configure("yellow", background=C["row_yellow"])
        self.rf_tree.tag_configure("blue",   background=C["row_blue"])
        self.rf_tree.tag_configure("red",    background=C["row_red"])

        # plot container pane
        plot_frame = ttk.Frame(bot, style="Card.TFrame")
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        # Adjusted aspect ratio to fit the wider 50% screen section beautifully
        self.rf_fig, self.rf_ax = plt.subplots(figsize=(5.5, 4.0), dpi=100, facecolor=C["card"])
        self.rf_ax.set_facecolor("#FAFAFA")
        self.rf_canvas = FigureCanvasTkAgg(self.rf_fig, master=plot_frame)
        self.rf_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        self._rf_blank_plot()

    def _rf_toggle_params(self, _event):
        m = self.rf_method.get()
        bracket = m in ("Incremental", "Bisection", "Regula-Falsi")
        secant  = (m == "Secant")
        newton  = (m == "Newton-Raphson")

        show_ab = bracket or secant
        for w in (self._rf_a_lbl, self.rf_a, self._rf_b_lbl, self.rf_b):
            w.grid() if show_ab else w.grid_remove()
        for w in (self._rf_x0_lbl, self.rf_x0):
            w.grid() if newton else w.grid_remove()

        if secant:
            self._rf_a_lbl.configure(text="x₀:")
            self._rf_b_lbl.configure(text="x₁:")
        else:
            self._rf_a_lbl.configure(text="a (lower):")
            self._rf_b_lbl.configure(text="b (upper):")

    def _rf_blank_plot(self):
        ax = self.rf_ax; ax.clear()
        ax.set_title("Functional Curve Mapping View f(x)", fontsize=10, fontweight="bold", color=C["text"], pad=10)
        ax.set_xlabel("x Coordinate System", fontsize=9, color=C["sub"])
        ax.set_ylabel("f(x) Value Scale", fontsize=9, color=C["sub"])
        ax.tick_params(labelsize=8, colors=C["sub"])
        ax.grid(True, alpha=0.2, color="#94A3B8")
        self.rf_fig.tight_layout(pad=2.0)
        self.rf_canvas.draw()

    def _rf_set_status(self, msg, kind="ok"):
        colours = {"ok": C["accent"], "warn": C["warn"], "error": C["error"], "sub": C["sub"]}
        self.rf_result_var.set(msg)
        self.rf_result_lbl.configure(fg=colours.get(kind, C["text"]))

    # ── solve dispatcher ──────────────────────────────────────────────────
    def _rf_solve(self):
        expr = self.rf_eq.get().strip()
        if not expr:
            self._rf_set_status("Please enter an equation.", "error"); return
        try:
            f, clean = make_func(expr)
        except Exception as e:
            self._rf_set_status(f"Syntax error: {e}", "error"); return

        method = self.rf_method.get()
        try:
            tol   = float(self.rf_tol.get())
            maxn  = int(float(self.rf_maxn.get()))
            a     = float(self.rf_a.get())  if method != "Newton-Raphson" else None
            b     = float(self.rf_b.get())  if method != "Newton-Raphson" else None
            x0    = float(self.rf_x0.get()) if method == "Newton-Raphson" else None
        except ValueError as e:
            self._rf_set_status(f"Invalid parameter: {e}", "error"); return

        try:
            if method == "Incremental":
                root, conv, rows = solve_incremental(f, a, b, tol, maxn)
                self._rf_show_incremental(rows)
            elif method == "Bisection":
                root, conv, rows = solve_bisection(f, a, b, tol, maxn)
                self._rf_show_generic(
                    rows,
                    ("i", "a", "b", "c = (a+b)/2", "f(c)", "|b-a|"),
                    (1, 6, 6, 10, 10, 12),
                    color_fn=lambda r: "blue" if abs(r[4]) < tol else "green"
                )
            elif method == "Regula-Falsi":
                root, conv, rows = solve_regula_falsi(f, a, b, tol, maxn)
                self._rf_show_rf(rows)
            elif method == "Newton-Raphson":
                root, conv, rows = solve_newton(f, x0, tol, maxn)
                self._rf_show_generic(
                    rows,
                    ("i", "xₙ", "f(xₙ)", "f'(xₙ)", "xₙ₊₁", "|error|"),
                    (1, 11, 12, 12, 11, 12),
                    color_fn=lambda r: "blue" if r[5] < tol else "green"
                )
            elif method == "Secant":
                root, conv, rows = solve_secant(f, a, b, tol, maxn)
                self._rf_show_generic(
                    rows,
                    ("i", "xₙ₋₁", "xₙ", "f(xₙ₋₁)", "f(xₙ)", "xₙ₊₁", "|error|"),
                    (1, 11, 11, 12, 12, 11, 12),
                    color_fn=lambda r: "blue" if r[6] < tol else "green"
                )
        except ValueError as e:
            self._rf_set_status(str(e), "error"); return
        except Exception as e:
            self._rf_set_status(f"Computation error: {e}", "error"); return

        n_iter = len(rows)
        if conv:
            self._rf_set_status(
                f"[{method}]   Root Found ≈ {root:.8f}   │   f(root) = {f(root):.3e}"
                f"   │   {n_iter} Evaluation Steps", "ok")
        else:
            self._rf_set_status(
                f"[{method}]   Did not fully converge — {n_iter} iterations. "
                "Adjust bounds or tolerance configuration parameters.", "warn")

        self._rf_plot(f, method, expr, root, a, b, x0, rows, conv)

    # ── table helpers ─────────────────────────────────────────────────────
    def _rf_clear_tree(self, cols, widths):
        for c in self.rf_tree["columns"]:
            self.rf_tree.heading(c, text="")
        self.rf_tree.delete(*self.rf_tree.get_children())
        self.rf_tree["columns"] = cols
        for c, w in zip(cols, widths):
            self.rf_tree.heading(c, text=c, anchor="center")
            self.rf_tree.column(c, width=w*8, anchor="center", stretch=True)

    def _rf_show_incremental(self, rows):
        cols   = ("Iter", "xₗ", "Δx", "xᵤ", "f(xₗ)", "f(xᵤ)", "f(xₗ)·f(xᵤ)", "Remark")
        widths = (4,       9,    9,    9,    11,      11,       13,              20)
        self._rf_clear_tree(cols, widths)
        for i, r in enumerate(rows):
            vals = (i+1, f"{r[0]:.6f}", f"{r[1]:.2e}", f"{r[2]:.6f}",
                    f"{r[3]:.6f}", f"{r[4]:.6f}", f"{r[5]:.4e}", r[6])
            self.rf_tree.insert("", "end", values=vals, tags=(r[7],))

    def _rf_show_rf(self, rows):
        cols   = ("#", "xₗ", "xᵤ", "xᵣ", "Eₐ%", "f(xₗ)", "f(xᵤ)", "f(xᵣ)", "f(xₗ)·f(xᵣ)", "Remark")
        widths = (3,    9,    9,    9,    9,      11,      11,      12,      13,              12)
        self._rf_clear_tree(cols, widths)
        for r in rows:
            ea_s = f"{r[4]:.4f}" if isinstance(r[4], float) else ""
            vals = (r[0], f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.6f}", ea_s,
                    f"{r[5]:.6f}", f"{r[6]:.6f}", f"{r[7]:.4e}", f"{r[8]:.4e}", r[9])
            self.rf_tree.insert("", "end", values=vals, tags=(r[10],))

    def _rf_show_generic(self, rows, cols, widths, color_fn=None):
        self._rf_clear_tree(cols, widths)
        for r in rows:
            fmtted = []
            for v in r:
                if isinstance(v, int):
                    fmtted.append(str(v))
                elif isinstance(v, float):
                    fmtted.append(f"{v:.8g}")
                else:
                    fmtted.append(str(v))
            tag = color_fn(r) if color_fn else "green"
            self.rf_tree.insert("", "end", values=fmtted, tags=(tag,))

    # ── plot ──────────────────────────────────────────────────────────────
    def _rf_plot(self, f, method, expr, root, a, b, x0, rows, conv):
        ax = self.rf_ax; ax.clear()

        if method == "Newton-Raphson":
            cx = x0; xlo, xhi = cx - 3, cx + 3
        elif method == "Secant":
            pts = [r[1] for r in rows] + [r[2] for r in rows]
            xlo, xhi = min(pts) - 0.75, max(pts) + 0.75
        elif a is not None and b is not None:
            margin = abs(b - a) * 0.2
            xlo, xhi = a - margin, b + margin
        else:
            xlo, xhi = root - 3, root + 3

        xv = np.linspace(xlo, xhi, 750)
        try:
            yv = np.array([f(xi) for xi in xv], dtype=float)
        except:
            yv = np.full_like(xv, np.nan)

        ax.plot(xv, yv, color=C["accent"], linewidth=2.5, zorder=3, label="f(x)")
        ax.axhline(0, color="#475569", linewidth=1.2, zorder=2)
        ax.axvline(0, color="#475569", linewidth=0.8, linestyle=":", zorder=2)

        cmap_c = plt.get_cmap("winter")
        cmap_a = plt.get_cmap("autumn")
        cmap_s = plt.get_cmap("plasma")
        n = max(len(rows), 1)

        if method == "Bisection":
            for k, r in enumerate(rows):
                col = cmap_c(k / n)
                ax.plot([r[1], r[2]], [0, 0], "-", color=col, lw=2.5, alpha=0.5)
                ax.plot([r[3], r[3]], [0, r[4]], ":", color=col, lw=1.2)
                ax.plot(r[3], 0, "o", color=col, ms=4, zorder=4)

        elif method == "Regula-Falsi":
            for k, r in enumerate(rows):
                col = cmap_c(k / n)
                fxL, fxU = f(r[1]), f(r[2])
                ax.plot([r[1], r[2]], [fxL, fxU], "--", color=col, lw=1.3, alpha=0.6)
                ax.plot([r[3], r[3]], [0, r[7]], ":", color=col, lw=1)
                ax.plot(r[3], 0, "v", color=col, ms=5, zorder=4)
            ax.plot(a, f(a), "gs", ms=7, zorder=5, label="Initial a")
            ax.plot(b, f(b), "rs", ms=7, zorder=5, label="Initial b")

        elif method == "Newton-Raphson":
            for k, r in enumerate(rows):
                col = cmap_a(k / n)
                xk, fk, dk = r[1], r[2], r[3]
                span = 1.0
                xt = np.linspace(xk - span, xk + span, 40)
                yt = fk + dk * (xt - xk)
                ax.plot(xt, yt, "-", color=col, lw=1.2, alpha=0.6)
                ax.plot([xk, xk], [0, fk], ":", color=col, lw=1)
                ax.plot(xk, fk, "o", color=col, ms=4, zorder=4)

        elif method == "Secant":
            for k, r in enumerate(rows):
                col = cmap_s(k / n)
                f0k, f1k = r[3], r[4]
                ax.plot([r[1], r[5]], [f0k, f0k + (f1k - f0k) / (r[2] - r[1]) * (r[5] - r[1])],
                        "--", color=col, lw=1.2, alpha=0.6)
                ax.plot([r[5], r[5]], [0, f(r[5])], ":", color=col, lw=1)
                ax.plot(r[5], 0, "v", color=col, ms=5, zorder=4)

        elif method == "Incremental":
            for k, r in enumerate(rows):
                col = "#4ADE80" if r[7] == "green" else "#FBBF24"
                ax.axvline(r[0], color=col, lw=0.8, linestyle="--", alpha=0.5)

        if conv and not np.isnan(root):
            ax.axvline(root, color=C["error"], lw=1.2, linestyle="-.", alpha=0.7)
            ax.plot(root, f(root), "X", color=C["error"], ms=10, zorder=6, markeredgecolor="white")
            ax.plot(root, 0, "o", color="#F59E0B", ms=9, zorder=7, markeredgecolor=C["error"], markeredgewidth=1.5)
            
            yf = yv[np.isfinite(yv)]
            off = max(abs(yf)) * 0.12 if len(yf) and max(abs(yf)) > 0 else 0.2
            ax.text(root, off, f" Root Convergence\n x = {root:.5f}", fontsize=8.5,
                    fontweight="bold", color=C["error"], va="bottom", ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C["border"], alpha=0.85))

        ax.set_title(f"{method} Convergence Mapping Pattern: f(x) = {expr}", fontsize=10, fontweight="bold", color=C["text"], pad=10)
        ax.set_xlabel("x Domain Range Space", fontsize=9, color=C["sub"])
        ax.set_ylabel("f(x) Value Frame", fontsize=9, color=C["sub"])
        ax.tick_params(labelsize=8, colors=C["sub"])
        ax.grid(True, alpha=0.2, color="#94A3B8")
        
        yf = yv[np.isfinite(yv)]
        if len(yf) and max(abs(yf)) > 0:
            ax.set_ylim(-max(abs(yf)) * 1.3, max(abs(yf)) * 1.3)
            
        self.rf_fig.tight_layout(pad=2.0)
        self.rf_canvas.draw()

    # ══════════════════════════════════════════════════════════════════════
    #   MATRIX TAB
    # ══════════════════════════════════════════════════════════════════════
    def _build_matrix_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        # ── controls bar ──
        ctrl = ttk.Frame(parent, style="Card.TFrame", padding=14)
        ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=(12, 6))

        ttk.Label(ctrl, text="Rows:", font=FONT_B, style="Card.TLabel").pack(side="left")
        self.mx_rows = ttk.Spinbox(ctrl, from_=1, to=6, width=4, font=FONT)
        self.mx_rows.set(3); self.mx_rows.pack(side="left", padx=(4, 16))

        ttk.Label(ctrl, text="Cols:", font=FONT_B, style="Card.TLabel").pack(side="left")
        self.mx_cols = ttk.Spinbox(ctrl, from_=1, to=6, width=4, font=FONT)
        self.mx_cols.set(3); self.mx_cols.pack(side="left", padx=(4, 16))

        ttk.Label(ctrl, text="Operation:", font=FONT_B, style="Card.TLabel").pack(side="left")
        self.mx_op = ttk.Combobox(ctrl, width=24, state="readonly", font=FONT, values=[
            "Addition (A + B)", "Multiplication (A × B)",
            "Transpose of A", "Determinant of A",
            "Inverse of A", "Adjoint of A",
            "Power of A  (Aⁿ)", "Linear Equations (Ax = b)"
        ])
        self.mx_op.current(0); self.mx_op.pack(side="left", padx=(4, 16))
        self.mx_op.bind("<<ComboboxSelected>>", self._mx_op_changed)

        ttk.Label(ctrl, text="n =", font=FONT_B, style="Card.TLabel").pack(side="left")
        self.mx_n = ttk.Spinbox(ctrl, from_=1, to=20, width=4, font=FONT)
        self.mx_n.set(2); self.mx_n.pack(side="left", padx=(4, 18))

        ttk.Button(ctrl, text="Set Structure", style="Secondary.TButton",
                   command=self._mx_build).pack(side="left", padx=(0, 8))
        ttk.Button(ctrl, text="  EXECUTE  ", style="Primary.TButton",
                   command=self._mx_calc).pack(side="left")

        # ── matrix entry grid ──
        grid_area = ttk.Frame(parent, style="Card.TFrame", padding=16)
        grid_area.grid(row=1, column=0, sticky="ew", padx=8, pady=4)

        self.mx_lbl_a = ttk.Label(grid_area, text="Matrix A Matrix Source", font=FONT_LG, style="Card.TLabel", foreground=C["accent"])
        self.mx_lbl_a.grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.mx_lbl_b = ttk.Label(grid_area, text="Matrix B Matrix Source", font=FONT_LG, style="Card.TLabel", foreground=C["accent"])
        self.mx_lbl_b.grid(row=0, column=1, sticky="w", padx=(48, 0), pady=(0, 8))

        self.mx_frame_a = ttk.Frame(grid_area, style="Card.TFrame")
        self.mx_frame_a.grid(row=1, column=0, sticky="nw")
        self.mx_frame_b = ttk.Frame(grid_area, style="Card.TFrame")
        self.mx_frame_b.grid(row=1, column=1, sticky="nw", padx=(48, 0))

        MAX = 6; CW = 58; CH = 28
        self.mx_ea = [[None]*MAX for _ in range(MAX)]
        self.mx_eb = [[None]*MAX for _ in range(MAX)]

        for i in range(MAX):
            for j in range(MAX):
                va = "1" if i == j else "0"
                ea = ttk.Entry(self.mx_frame_a, width=8, font=FONT_MONO, justify="center")
                ea.insert(0, va); ea.grid(row=i, column=j, padx=3, pady=3, ipady=3)
                self.mx_ea[i][j] = ea

                eb = ttk.Entry(self.mx_frame_b, width=8, font=FONT_MONO, justify="center")
                eb.insert(0, va); eb.grid(row=i, column=j, padx=3, pady=3, ipady=3)
                self.mx_eb[i][j] = eb

        self._mx_build()

        # ── output area ──
        out_frame = ttk.Frame(parent, style="Card.TFrame", padding=(12, 10))
        out_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 16))
        out_frame.rowconfigure(0, weight=1); out_frame.columnconfigure(0, weight=1)

        self.mx_out = scrolledtext.ScrolledText(
            out_frame, font=FONT_MONO, bg=C["card"], fg=C["text"],
            relief="flat", borderwidth=0, state="disabled", wrap="none",
            height=10
        )
        self.mx_out.grid(row=0, column=0, sticky="nsew")

        self._mx_write("Evaluated linear algebraic data arrays will be printed onto this logging frame console workspace block.")

    def _mx_op_changed(self, _):
        self._mx_build()

    def _mx_build(self):
        try:
            r = int(self.mx_rows.get())
            c = int(self.mx_cols.get())
        except:
            r, c = 3, 3
        r = max(1, min(6, r)); c = max(1, min(6, c))
        op = self.mx_op.get()
        need_b = op in ("Addition (A + B)", "Multiplication (A × B)")
        need_v = op == "Linear Equations (Ax = b)"

        for i in range(6):
            for j in range(6):
                if i < r and j < c:
                    self.mx_ea[i][j].grid()
                else:
                    self.mx_ea[i][j].grid_remove()
                show_bj = j < (1 if need_v else c)
                show_b  = (need_b or need_v) and i < r and show_bj
                if show_b:
                    self.mx_eb[i][j].grid()
                else:
                    self.mx_eb[i][j].grid_remove()

        self.mx_lbl_b.configure(text="System Vector b Target Space" if need_v else "Matrix B Matrix Source")
        self.mx_lbl_b.grid() if (need_b or need_v) else self.mx_lbl_b.grid_remove()
        self.mx_frame_b.grid() if (need_b or need_v) else self.mx_frame_b.grid_remove()

    def _mx_read(self, grid, rows, cols):
        M = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                try:
                    M[i, j] = float(grid[i][j].get())
                except:
                    M[i, j] = 0
        return M

    def _mx_write(self, text):
        self.mx_out.configure(state="normal")
        self.mx_out.delete("1.0", "end")
        self.mx_out.insert("end", text)
        self.mx_out.configure(state="disabled")

    def _mx_calc(self):
        try:
            r = int(self.mx_rows.get()); c = int(self.mx_cols.get())
        except:
            self._mx_write("ERROR: Invalid rows/cols structure bounds inputs."); return
        r = max(1, min(6, r)); c = max(1, min(6, c))
        op = self.mx_op.get()
        A = self._mx_read(self.mx_ea, r, c)
        lines = []

        try:
            if op == "Addition (A + B)":
                B = self._mx_read(self.mx_eb, r, c)
                lines.append(fmt_matrix("Sum Array Matrix A + B Output", A + B))

            elif op == "Multiplication (A × B)":
                B = self._mx_read(self.mx_eb, r, c)
                if A.shape[1] != B.shape[0]:
                    self._mx_write("ERROR: Dimensional inner boundary mismatch! Columns of A must equate Rows of B."); return
                lines.append(fmt_matrix("Dot Product Matrix A × B Output", A @ B))

            elif op == "Transpose of A":
                lines.append(fmt_matrix("Transposed Structure Aᵀ", A.T))

            elif op == "Determinant of A":
                if r != c:
                    self._mx_write("ERROR: Scalar Determinant evaluation requires an equivalent square data matrix space form."); return
                d = np.linalg.det(A)
                lines.append(f"  Matrix Determinant Evaluator Scalar det(A)  =  {d:.10f}\n")

            elif op == "Inverse of A":
                if r != c:
                    self._mx_write("ERROR: Structural matrix invert functions require uniform square geometry configurations."); return
                if abs(np.linalg.det(A)) < 1e-12:
                    self._mx_write("ERROR: Singular Matrix error state triggered! Inverse execution structurally locked (Det ≈ 0)."); return
                lines.append(fmt_matrix("Inverted Matrix Form A⁻¹", np.linalg.inv(A)))

            elif op == "Adjoint of A":
                if r != c:
                    self._mx_write("ERROR: Classical Adjugate matrices require standard square dimension properties."); return
                lines.append(fmt_matrix("Classical Adjoint Formulation adj(A)", adjoint(A)))

            elif op == "Power of A  (Aⁿ)":
                if r != c:
                    self._mx_write("ERROR: Structural power scaling functions request uniform square arrays."); return
                try:
                    n = int(self.mx_n.get())
                except:
                    n = 2
                result = np.linalg.matrix_power(A, n)
                lines.append(fmt_matrix(f"Matrix Structural Power Factorization Output A^{n}", result))

            elif op == "Linear Equations (Ax = b)":
                if r != c:
                    self._mx_write("ERROR: System requires a square coefficient matrix mapping framework."); return
                b_vec = self._mx_read(self.mx_eb, r, 1).flatten()
                if abs(np.linalg.det(A)) < 1e-12:
                    self._mx_write("ERROR: Singular system — no unique solution."); return
                x = np.linalg.solve(A, b_vec)
                lines.append("  System Solution Vectors tracking space (Ax = b):\n")
                for i, xi in enumerate(x):
                    lines.append(f"    x{i+1} Vector Component Coordinate Base  =  {xi:14.8f}")
                residual = np.linalg.norm(A @ x - b_vec)
                lines.append(f"\n  System Calibration Identity Vector Check Verification  ‖Ax − b‖  =  {residual:.4e}\n")

        except Exception as e:
            self._mx_write(f"Structural execution fault occurred: {e}"); return

        self._mx_write("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = NumericLabApp()
    app.mainloop()