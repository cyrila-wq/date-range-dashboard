"""
Dynamic date-range dashboard
============================
Upload any Excel / CSV that has a start-date column, an end-date column and one
or more value columns. Pick the columns you want to see, add filters, choose the
date range and months, then click "Generate Dashboard". The app spreads each
row's value across every day from start to end:

    SKU ID     4-Nov-2026  5-Nov-2026  6-Nov-2026 ...
    40325899   656.25      656.25      -          ...

Everything is chosen by you from the headers of the uploaded file:
    - Start date column and End date column (dates are read as DD-MM-YYYY)
    - Columns to show on the left (SKU ID, Brand, Manufacturer ...)
    - Value column(s) (margin, cost, discount ...)
    - Filters on any column (applied BEFORE any date expansion)
    - Date range and which months to show (applied BEFORE any date expansion)
    - Download of the dashboard as Excel / CSV

How it stays fast on big files (e.g. 167,901 rows):
    - The whole file is read once and kept in session state.
    - Filters, date range and months are applied first, using small NumPy masks
      (no copy of the 37-column table is made).
    - Only the rows/days that survive are expanded, in chunks, and are folded
      straight into the final table (rows x dates). The full day-by-day
      table is never held in memory.
    - Nothing is recalculated until you click "Generate Dashboard".

Run:
    pip install -r requirements.txt
    python -m streamlit run Dasboard.py
"""

import io

import numpy as np
import pandas as pd

AGG_FUNCS = {
    "Average": "mean",
    "Sum": "sum",
    "Max": "max",
    "Min": "min",
    "First": "first",
    "Last": "last",
}
BLANK = "(blank)"

# Safety limits (the app warns clearly instead of crashing when they are exceeded)
MAX_EXPANDED_ROWS = 30_000_000   # day-level records processed in one run (records x value columns)
MAX_MATRIX_CELLS = 8_000_000     # dashboard rows x date columns x value columns
MAX_DATE_COLUMNS = 200_000       # hard stop for absurd date ranges
CHUNK_RECORDS = 1_000_000        # day-level records expanded at one time (keeps memory small)
EXCEL_MAX_CELLS = 1_000_000      # Excel is only built up to this many cells (CSV has no such limit)
DISPLAY_MAX_CELLS = 400_000      # on-screen preview limit (downloads always contain everything)


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def parse_dates(series: pd.Series) -> pd.Series:
    """Read dates as day-first (DD-MM-YYYY). Handles real Excel dates, Excel serial numbers and text with a time."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.normalize()

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        # Excel serial numbers (e.g. 46330 = 4-Nov-2026) -- only if they look like real serials
        nums = pd.to_numeric(series, errors="coerce")
        plausible = nums.between(20000, 80000)
        if plausible.sum() >= max(1, int(nums.notna().sum() * 0.9)):
            return pd.to_datetime(nums.where(plausible), unit="D", origin="1899-12-30", errors="coerce").dt.normalize()

    text = series.astype(str).str.strip()
    out = pd.to_datetime(text, format="%d-%m-%Y", errors="coerce")

    rest = out.isna() & series.notna()
    if rest.any():
        try:
            out[rest] = pd.to_datetime(text[rest], dayfirst=True, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            out[rest] = pd.to_datetime(text[rest], dayfirst=True, errors="coerce")
    return out.dt.normalize()


def clean_numeric(series: pd.Series) -> pd.Series:
    """Turn '10%', ' 15 %', '1,200' into numbers. Anything unreadable becomes blank."""
    if not pd.api.types.is_numeric_dtype(series):
        series = (
            series.astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
    return pd.to_numeric(series, errors="coerce")


def to_label(series: pd.Series) -> pd.Series:
    """Make a column safe to use as a row label (IDs stay as 40325899, not 40325899.0)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        out = series.dt.strftime("%d-%b-%Y")
    elif pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if len(non_null) and (non_null % 1 == 0).all():
            out = series.astype("Int64").astype(str)
        else:
            out = series.astype(str)
    else:
        out = series.astype(str).str.strip()
    out = out.where(series.notna(), BLANK)
    return out.replace({"": BLANK, "nan": BLANK, "<NA>": BLANK, "NaT": BLANK, "None": BLANK})


def guess_column(columns, patterns, exclude=()):
    """Pick the first column whose name contains one of the patterns (patterns in priority order)."""
    for pat in patterns:
        for col in columns:
            if col not in exclude and str(col).strip().lower() == pat.strip():
                return col
    for pat in patterns:
        for col in columns:
            if col not in exclude and pat in str(col).lower():
                return col
    return None


# --------------------------------------------------------------------------- #
# Small date helpers (work on whole-day integers, which is fast and exact)
# --------------------------------------------------------------------------- #
def _to_days(values):
    """datetime64 array -> (whole-day integers, valid mask). NaT rows are marked invalid."""
    values = np.asarray(values)
    valid = ~np.isnat(values)
    days = np.zeros(len(values), dtype="int64")
    if valid.any():
        days[valid] = values[valid].astype("datetime64[D]").astype("int64")
    return days, valid


def _ts_to_day(ts) -> int:
    """Timestamp -> whole-day integer."""
    return int(pd.Timestamp(ts).normalize().to_datetime64().astype("datetime64[D]").astype("int64"))


def _clip_to_range(lo_d, hi_d, good, range_lo, range_hi):
    """Cut every row's [start, end] to the selected range. Rows fully outside are dropped."""
    a = np.maximum(lo_d, range_lo)
    b = np.minimum(hi_d, range_hi)
    keep = good & (a <= b)
    return a, b, keep


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #
def date_bounds(df, start_col, end_col, start_dates=None, end_dates=None, row_mask=None):
    """Return (earliest start, latest end, number of readable rows).

    start_dates / end_dates: already-parsed dates (optional, avoids parsing again).
    row_mask: boolean array, True for the rows that passed the filters (optional).
    """
    s = start_dates if start_dates is not None else parse_dates(df[start_col])
    e = end_dates if end_dates is not None else parse_dates(df[end_col])
    sd, sv = _to_days(s.to_numpy())
    ed, ev = _to_days(e.to_numpy())
    if row_mask is not None:
        sd, sv, ed, ev = sd[row_mask], sv[row_mask], ed[row_mask], ev[row_mask]
    good = sv & ev
    if not good.any():
        return None, None, 0
    lo = min(sd[good].min(), ed[good].min())
    hi = max(sd[good].max(), ed[good].max())
    return pd.Timestamp(int(lo), unit="D"), pd.Timestamp(int(hi), unit="D"), int(good.sum())


def expansion_size(start_dates, end_dates, range_start, range_end, row_mask=None):
    """Cheap estimate: (rows that touch the date range, day-level records they will create)."""
    sd, sv = _to_days(start_dates.to_numpy())
    ed, ev = _to_days(end_dates.to_numpy())
    if row_mask is not None:
        sd, sv, ed, ev = sd[row_mask], sv[row_mask], ed[row_mask], ev[row_mask]
    good = sv & ev
    a, b, keep = _clip_to_range(np.minimum(sd, ed), np.maximum(sd, ed), good,
                                _ts_to_day(range_start), _ts_to_day(range_end))
    if not keep.any():
        return 0, 0
    return int(keep.sum()), int((b[keep] - a[keep] + 1).sum())


def _accumulate(acc, cnt, flat, vals, agg):
    """Fold one chunk of day-level records into the running result (acc / cnt are flat arrays)."""
    grp = pd.DataFrame({"k": flat, "v": vals}).groupby("k", sort=False)["v"]
    if agg == "Average":
        g = grp.agg(["sum", "count"])
        k = g.index.to_numpy()
        acc[k] += g["sum"].to_numpy()
        cnt[k] += g["count"].to_numpy()
        return
    g = grp.agg(AGG_FUNCS[agg])
    k = g.index.to_numpy()
    gv = g.to_numpy(dtype="float64")
    cur = acc[k]
    if agg == "Sum":
        acc[k] = np.where(np.isnan(cur), gv, cur + gv)
    elif agg == "Max":
        acc[k] = np.fmax(cur, gv)
    elif agg == "Min":
        acc[k] = np.fmin(cur, gv)
    elif agg == "First":
        acc[k] = np.where(np.isnan(cur), gv, cur)   # keep the earliest value seen
    elif agg == "Last":
        acc[k] = gv                                  # later rows overwrite earlier ones


def build_matrix(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    value_cols: list,
    show_cols: list,
    agg: str = "Average",
    range_start=None,
    range_end=None,
    visible_months=None,
    start_dates=None,
    end_dates=None,
    row_mask=None,
    progress_cb=None,
):
    """
    One row per combination of the shown columns (plus 'Metric' if more than one
    value column), one column per date. Returns (matrix, number_of_skipped_rows).

    df             : the full table (it is NOT copied).
    row_mask       : boolean array, True for rows that passed the filters.
    start_dates /
    end_dates      : already-parsed date Series for df (optional, saves time).
    progress_cb    : optional function(fraction 0-1, message).

    Order of work: filters (row_mask) -> date range -> months -> expand only what is left.
    """

    def report(frac, msg):
        if progress_cb is not None:
            progress_cb(float(min(max(frac, 0.0), 1.0)), msg)

    if agg not in AGG_FUNCS:
        raise ValueError(f"Unknown aggregation: {agg}")

    # ---- 1. dates of the filtered rows (integers, no copies of df) ----------
    report(0.02, "Reading dates...")
    s = start_dates if start_dates is not None else parse_dates(df[start_col])
    e = end_dates if end_dates is not None else parse_dates(df[end_col])
    sd, sv = _to_days(s.to_numpy())
    ed, ev = _to_days(e.to_numpy())
    pos = np.arange(len(df)) if row_mask is None else np.flatnonzero(row_mask)
    sd, sv, ed, ev = sd[pos], sv[pos], ed[pos], ev[pos]

    good = sv & ev
    bad_rows = int((~good).sum())
    if not good.any():
        return pd.DataFrame(), bad_rows

    lo_d = np.minimum(sd, ed)   # reversed dates are swapped safely
    hi_d = np.maximum(sd, ed)

    # ---- 2. date range: cut every row to the selected range BEFORE expanding
    range_lo = _ts_to_day(range_start) if range_start is not None else int(lo_d[good].min())
    range_hi = _ts_to_day(range_end) if range_end is not None else int(hi_d[good].max())
    if range_lo > range_hi:
        raise ValueError("The 'from' date is after the 'to' date. Please fix the date range.")
    n_days = range_hi - range_lo + 1
    if n_days > MAX_DATE_COLUMNS:
        raise ValueError(f"The date range covers {n_days:,} days, which is too many. Please narrow it.")

    a, b, keep = _clip_to_range(lo_d, hi_d, good, range_lo, range_hi)
    if not keep.any():
        return pd.DataFrame(), bad_rows
    rows = pos[keep]              # positions in df of the rows we really need
    a_rel = a[keep] - range_lo    # first day (0-based) inside the range
    b_rel = b[keep] - range_lo    # last day (0-based) inside the range
    lengths = b_rel - a_rel + 1   # start and end dates are both included

    # ---- 3. months to show -> which day-columns exist ------------------------
    all_dates = pd.date_range(pd.Timestamp(range_lo, unit="D"), periods=n_days, freq="D")
    if visible_months is not None:
        vis_mask = np.asarray(pd.Index(all_dates.strftime("%b-%Y")).isin(list(visible_months)))
    else:
        vis_mask = np.ones(n_days, dtype=bool)
    D = int(vis_mask.sum())
    if D == 0:
        return pd.DataFrame(), bad_rows
    col_of_day = np.full(n_days, -1, dtype="int64")
    col_of_day[vis_mask] = np.arange(D)
    kept_dates = all_dates[vis_mask]

    M = len(value_cols)
    total_records = int(lengths.sum())
    if total_records * M > MAX_EXPANDED_ROWS:
        raise ValueError(
            f"This would need {total_records * M:,} day-level records "
            f"(limit {MAX_EXPANDED_ROWS:,}). Add filters or narrow the date range / months."
        )

    # ---- 4. one group id per combination of the shown columns ----------------
    report(0.05, "Grouping rows...")
    if show_cols:
        keys = pd.DataFrame({c: to_label(df[c].iloc[rows]).to_numpy() for c in show_cols})
        grouped = keys.groupby(list(show_cols), sort=True)
        gid = grouped.ngroup().to_numpy().astype("int64")
        group_frame = grouped.size().index.to_frame(index=False)
        del keys
        G = len(group_frame)
    else:
        gid = np.zeros(len(rows), dtype="int64")
        group_frame = None
        G = 1

    cells = G * M * D
    if cells > MAX_MATRIX_CELLS:
        raise ValueError(
            f"The dashboard would have {G * M:,} rows x {D:,} date columns = {cells:,} cells "
            f"(limit {MAX_MATRIX_CELLS:,}). Add filters, choose fewer columns on the left, "
            "or select fewer months."
        )

    # ---- 5. expand in chunks and fold straight into the result ---------------
    result = np.full((G, M, D), np.nan, dtype="float64")
    for m, v in enumerate(value_cols):
        vals_all = clean_numeric(df[v].iloc[rows]).to_numpy(dtype="float64", na_value=np.nan)
        ok_idx = np.flatnonzero(~np.isnan(vals_all))
        acc = np.zeros(G * D, dtype="float64") if agg == "Average" else np.full(G * D, np.nan)
        cnt = np.zeros(G * D, dtype="int64") if agg == "Average" else None

        if len(ok_idx):
            cum = np.cumsum(lengths[ok_idx])
            n_ok = len(ok_idx)
            start = 0
            while start < n_ok:
                base = int(cum[start - 1]) if start else 0
                end = int(np.searchsorted(cum, base + CHUNK_RECORDS, side="right"))
                end = min(max(end, start + 1), n_ok)

                sel = ok_idx[start:end]
                L = lengths[sel]
                tot = int(L.sum())
                rep = np.repeat(np.arange(len(sel)), L)
                offs = np.arange(tot, dtype="int64") - np.repeat(np.cumsum(L) - L, L)
                col = col_of_day[a_rel[sel][rep] + offs]
                flat = gid[sel][rep] * D + col
                vv = vals_all[sel][rep]
                if D != n_days:                      # some months are hidden
                    shown = col >= 0
                    flat, vv = flat[shown], vv[shown]
                if len(flat):
                    _accumulate(acc, cnt, flat, vv, agg)

                start = end
                report(0.08 + 0.82 * ((m + start / n_ok) / M),
                       f"Building dashboard: {v} ({start:,} of {n_ok:,} rows)")

        if agg == "Average":
            with np.errstate(divide="ignore", invalid="ignore"):
                acc = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        result[:, m, :] = acc.reshape(G, D)
        del acc, cnt, vals_all

    # ---- 6. assemble the final table ------------------------------------------
    report(0.92, "Assembling table...")
    flat2 = result.reshape(G * M, D)
    keep_rows = ~np.isnan(flat2).all(axis=1)
    if not keep_rows.any():
        return pd.DataFrame(), bad_rows

    if group_frame is not None:
        frame = group_frame.iloc[np.repeat(np.arange(G), M)].reset_index(drop=True)
    else:
        frame = pd.DataFrame(index=range(G * M))
    if M > 1:
        frame["Metric"] = pd.Categorical(
            np.tile(np.array(value_cols, dtype=object), G), categories=value_cols, ordered=True
        )
    frame = frame.loc[keep_rows]

    if frame.shape[1] == 0:
        index = pd.Index(["All"] * int(keep_rows.sum()), name="")
    elif frame.shape[1] == 1:
        index = pd.Index(frame.iloc[:, 0])
        index.name = frame.columns[0]
    else:
        index = pd.MultiIndex.from_frame(frame)

    pivot = pd.DataFrame(flat2[keep_rows], index=index, columns=kept_dates)
    return pivot, bad_rows


def date_label(d: pd.Timestamp) -> str:
    return f"{d.day}-{d.strftime('%b-%Y')}"


def _scale(pivot, fraction_to_percent):
    return pivot * 100 if fraction_to_percent else pivot


def format_for_display(pivot, as_percent_sign=False, fraction_to_percent=False):
    """Text version for on-screen display. Gaps show '-'."""
    out = _scale(pivot, fraction_to_percent)

    def fmt(v):
        if pd.isna(v):
            return "-"
        text = f"{round(float(v), 2):.2f}".rstrip("0").rstrip(".")
        return text + ("%" if as_percent_sign else "")

    out = out.apply(lambda col: col.map(fmt))
    out.columns = [date_label(d) for d in out.columns]
    return out.reset_index()


def format_for_export(pivot, as_percent_sign=False, fraction_to_percent=False):
    """Same table for Excel. Numbers stay real numbers (unless % sign is on); gaps are '-'."""
    out = _scale(pivot, fraction_to_percent).round(2)
    if as_percent_sign:
        out = out.apply(lambda col: col.map(lambda v: "-" if pd.isna(v) else f"{v:g}%"))
    else:
        out = out.astype(object).where(out.notna(), "-")
    out.columns = [date_label(d) for d in out.columns]
    return out.reset_index()


def to_csv_bytes(pivot, as_percent_sign=False, fraction_to_percent=False, block_rows=20_000):
    """Complete dashboard as CSV. Written in blocks of rows so memory stays small."""
    buf = io.StringIO()
    wrote_header = False
    for i in range(0, len(pivot), block_rows):
        block = format_for_export(pivot.iloc[i:i + block_rows], as_percent_sign, fraction_to_percent)
        block.to_csv(buf, index=False, header=not wrote_header)
        wrote_header = True
    return buf.getvalue().encode("utf-8")


def to_excel_bytes(export_df: pd.DataFrame, n_label_cols: int) -> bytes:
    """Excel file with bold header, frozen label columns + header row, shaded value cells."""
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook(write_only=True)          # streaming mode: fast and light on memory
    ws = wb.create_sheet("Dashboard")
    center = Alignment(horizontal="center")
    bold = Font(bold=True)
    fill = PatternFill("solid", start_color="E6F1FB", end_color="E6F1FB")

    ws.freeze_panes = f"{get_column_letter(n_label_cols + 1)}2"   # label columns + header row
    for i, col in enumerate(export_df.columns, start=1):
        if i <= n_label_cols:
            longest = max([len(str(col))] + [len(str(v)) for v in export_df[col].head(500)])
            ws.column_dimensions[get_column_letter(i)].width = min(max(longest + 2, 10), 45)
        else:
            ws.column_dimensions[get_column_letter(i)].width = 13

    header = []
    for name in export_df.columns:
        cell = WriteOnlyCell(ws, value=str(name))
        cell.font = bold
        cell.alignment = center
        header.append(cell)
    ws.append(header)

    for row in export_df.itertuples(index=False, name=None):
        out = []
        for j, v in enumerate(row):
            if j < n_label_cols:
                out.append(v)
            else:
                cell = WriteOnlyCell(ws, value=v)
                cell.alignment = center
                if v is not None and v != "-":
                    cell.fill = fill
                out.append(cell)
        ws.append(out)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def read_table(data: bytes, name: str, sheet, header_row: int):
    """Read the WHOLE file (no row limit)."""
    if name.lower().endswith((".csv", ".txt")):
        try:
            return pd.read_csv(io.BytesIO(data), header=header_row - 1)
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(data), header=header_row - 1, encoding="latin-1")
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=header_row - 1)


def sheet_names(data: bytes):
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


# --------------------------------------------------------------------------- #
# Streamlit app
# --------------------------------------------------------------------------- #
def main():
    import streamlit as st

    st.set_page_config(page_title="Date range dashboard", layout="wide")
    st.title("Date range dashboard")

    uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xlsm", "xls", "csv"])
    if uploaded is None:
        st.info("Upload a file, then choose its start date, end date, value and display columns.")
        return

    file_id = f"{uploaded.name}_{uploaded.size}"
    k = file_id
    is_csv = uploaded.name.lower().endswith((".csv", ".txt"))

    # A different file was uploaded: forget everything about the previous one.
    if st.session_state.get("_file_id") != file_id:
        for key in ("_df", "_data_key", "_cache", "_sheets", "dash"):
            st.session_state.pop(key, None)
        st.session_state["_file_id"] = file_id

    # ---- File / sheet / header row -------------------------------------------
    st.sidebar.header("File")
    sheet = None
    if not is_csv:
        if "_sheets" not in st.session_state:
            try:
                st.session_state["_sheets"] = sheet_names(uploaded.getvalue())
            except Exception as exc:  # noqa: BLE001
                st.error(f"Couldn't open the Excel file: {exc}")
                return
        names = st.session_state["_sheets"]
        sheet = st.sidebar.selectbox("Sheet", names, key=f"{k}_sheet") if len(names) > 1 else names[0]
    header_row = st.sidebar.number_input("Header row (1 = first row)", 1, 50, 1, key=f"{k}_hdr")

    # ---- Read the whole file once and keep it in session state ---------------
    data_key = (file_id, str(sheet), int(header_row))
    if st.session_state.get("_data_key") != data_key:
        try:
            with st.spinner("Reading the whole file... (large Excel files can take a minute)"):
                new_df = read_table(uploaded.getvalue(), uploaded.name, sheet, int(header_row))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Couldn't read the file: {exc}")
            return
        new_df.columns = [str(c).strip() for c in new_df.columns]
        st.session_state["_df"] = new_df
        st.session_state["_data_key"] = data_key
        st.session_state["_cache"] = {}      # parsed dates / labels / numbers for this table
        st.session_state.pop("dash", None)

    df = st.session_state["_df"]
    cache = st.session_state["_cache"]
    cols = list(df.columns)

    def cached(key, fn):
        if key not in cache:
            cache[key] = fn()
        return cache[key]

    with st.expander(f"Uploaded data preview ({len(df):,} rows, {len(cols)} columns)"):
        st.dataframe(df.head(200), hide_index=True)

    def idx(guess):
        return cols.index(guess) if guess in cols else 0

    promo_cols = [c for c in cols if "promo" in c.lower()]

    # ---- 1. Dates -------------------------------------------------------------
    st.sidebar.header("1. Dates")
    start_guess = guess_column(cols, ["start date", "from date", "start", "from"], exclude=promo_cols)
    end_guess = guess_column(cols, ["end date", "to date", "end", "till", "to "], exclude=promo_cols)
    start_col = st.sidebar.selectbox("Start date column", cols, index=idx(start_guess), key=f"{k}_sd")
    end_col = st.sidebar.selectbox("End date column", cols, index=idx(end_guess), key=f"{k}_ed")

    # ---- 2. Values ------------------------------------------------------------
    st.sidebar.header("2. Values")
    value_guess = guess_column(
        cols,
        ["discount", "margin", "landing", "value", "promo", "cost", "price", "mrp", "qty", "amount"],
        exclude=[start_col, end_col],
    )
    value_cols = st.sidebar.multiselect(
        "Value column(s)",
        [c for c in cols if c not in (start_col, end_col)],
        default=[value_guess] if value_guess else [],
        key=f"{k}_val",
    )
    agg = st.sidebar.selectbox("If date ranges overlap", list(AGG_FUNCS.keys()), key=f"{k}_agg")

    # ---- 3. Columns shown -----------------------------------------------------
    st.sidebar.header("3. Columns shown")
    other_cols = [c for c in cols if c not in (start_col, end_col)]
    sku_guess = guess_column(other_cols, ["sku id", "sku", "item", "product"])
    show_cols = st.sidebar.multiselect(
        "Columns on the left (e.g. SKU, Brand)",
        other_cols,
        default=[sku_guess] if sku_guess else other_cols[:1],
        key=f"{k}_show",
    )

    # ---- 4. Filters (a True/False mask over the full table -- no copies) ------
    st.sidebar.header("4. Filters")
    filter_cols = st.sidebar.multiselect("Filter on columns", cols, key=f"{k}_fcols")
    mask = np.ones(len(df), dtype=bool)
    active_filters = []
    for fc in filter_cols:
        labels = cached(("labels", fc), lambda fc=fc: to_label(df[fc]))
        options = sorted(pd.unique(labels.to_numpy()[mask]).tolist())
        chosen = st.sidebar.multiselect(fc, options, key=f"{k}_f_{fc}")
        if chosen:
            mask &= labels.isin(chosen).to_numpy()
            active_filters.append((fc, tuple(chosen)))
    n_filtered = int(mask.sum())
    st.sidebar.caption(f"Rows after filters: {n_filtered:,} of {len(df):,}")

    # ---- Dates of the whole file are parsed ONCE and remembered ---------------
    with st.spinner("Reading dates..."):
        S = cached(("dates", start_col), lambda: parse_dates(df[start_col]))
        E = cached(("dates", end_col), lambda: parse_dates(df[end_col]))

    lo, hi, readable = date_bounds(df, start_col, end_col, S, E, mask)
    if readable == 0:
        st.error(
            f"Couldn't read dates from '{start_col}' or '{end_col}' (expected DD-MM-YYYY) "
            "in the rows that pass the filters. Check the date column choice or the filters."
        )
        return

    # ---- 5. Date range / months -----------------------------------------------
    st.sidebar.header("5. Date range")
    st.sidebar.caption(f"Detected: {lo:%d-%b-%Y} to {hi:%d-%b-%Y}")
    range_start, range_end = lo, hi
    if st.sidebar.checkbox("Override date range", key=f"{k}_ovr"):
        picked = st.sidebar.date_input("From / to", value=(lo.date(), hi.date()), key=f"{k}_rng")
        if isinstance(picked, tuple) and len(picked) == 2:
            range_start, range_end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])

    months = list(dict.fromkeys(
        d.strftime("%b-%Y") for d in pd.date_range(range_start.replace(day=1), range_end, freq="MS")
    ))
    if not months:
        st.error("The selected date range is empty. Please fix the date range.")
        return
    chosen_months = st.sidebar.multiselect(
        "Months shown", months, default=months, key=f"{k}_months_{months[0]}_{months[-1]}"
    )

    # ---- 6. Number format -----------------------------------------------------
    st.sidebar.header("6. Number format")
    looks_fraction = False
    if value_cols:
        mins, maxs = [], []
        for v in value_cols:
            arr = cached(
                ("num", v),
                lambda v=v: clean_numeric(df[v]).to_numpy(dtype="float64", na_value=np.nan),
            )[mask]
            arr = arr[~np.isnan(arr)]
            if len(arr):
                mins.append(arr.min())
                maxs.append(arr.max())
        looks_fraction = bool(mins) and min(mins) >= 0 and max(maxs) <= 1
    fraction_to_percent = st.sidebar.checkbox(
        "Convert fractions to percent (0.1 becomes 10)",
        value=looks_fraction,
        key=f"{k}_frac_{int(looks_fraction)}_{'|'.join(value_cols)}",
    )
    percent_sign = st.sidebar.checkbox("Show % sign", value=False, key=f"{k}_pct")

    # ---- 7. Generate ----------------------------------------------------------
    st.sidebar.header("7. Generate")
    problems = []
    if not value_cols:
        problems.append("Choose at least one value column.")
    if not chosen_months:
        problems.append("Choose at least one month to show.")
    if n_filtered == 0:
        problems.append("No rows are left after the filters.")

    if value_cols and chosen_months and n_filtered:
        rows_in_range, day_records = expansion_size(S, E, range_start, range_end, mask)
        work = day_records * len(value_cols)
        st.sidebar.caption(
            f"Rows touching the date range: {rows_in_range:,}. "
            f"Day-level records to process: {work:,} (limit {MAX_EXPANDED_ROWS:,})."
        )
        if work > MAX_EXPANDED_ROWS:
            problems.append(
                "Too much data for one run. Add filters, narrow the date range, or choose fewer value columns."
            )
    for p in problems:
        st.sidebar.warning(p)

    settings = {
        "file": data_key,
        "start_col": start_col,
        "end_col": end_col,
        "values": tuple(value_cols),
        "show": tuple(show_cols),
        "agg": agg,
        "filters": tuple(active_filters),
        "range": (str(range_start.date()), str(range_end.date())),
        "months": tuple(chosen_months),
        "fraction_to_percent": bool(fraction_to_percent),
        "percent_sign": bool(percent_sign),
    }
    generate = st.sidebar.button("Generate Dashboard", type="primary", disabled=bool(problems), key=f"{k}_gen")

    # ---- Generate only when the button is clicked -----------------------------
    if generate:
        st.session_state.pop("dash", None)
        progress = st.progress(0.0, text="Starting...")

        def on_progress(fraction, message):
            progress.progress(fraction, text=message)

        try:
            matrix, skipped = build_matrix(
                df, start_col, end_col, value_cols, show_cols, agg,
                range_start, range_end, set(chosen_months),
                start_dates=S, end_dates=E, row_mask=mask, progress_cb=on_progress,
            )
            dash = {
                "settings": settings,
                "skipped": skipped,
                "rows_after_filters": n_filtered,
                "n_rows": len(matrix),
                "n_dates": matrix.shape[1] if not matrix.empty else 0,
                "lo": lo,
                "hi": hi,
                "shown": None,
                "preview_note": "",
                "csv": None,
                "excel": None,
                "excel_note": "",
            }
            if not matrix.empty:
                n_cols = matrix.shape[1]
                n_preview = max(50, DISPLAY_MAX_CELLS // max(1, n_cols))
                on_progress(0.94, "Preparing on-screen table...")
                dash["shown"] = format_for_display(matrix.head(n_preview), percent_sign, fraction_to_percent)
                if len(matrix) > n_preview:
                    dash["preview_note"] = (
                        f"Showing the first {n_preview:,} of {len(matrix):,} rows on screen. "
                        "The downloads contain every row."
                    )
                on_progress(0.96, "Preparing CSV download...")
                dash["csv"] = to_csv_bytes(matrix, percent_sign, fraction_to_percent)

                cells = matrix.shape[0] * n_cols
                if cells <= EXCEL_MAX_CELLS and n_cols + len(show_cols) + 1 < 16000:
                    on_progress(0.98, "Preparing Excel download...")
                    export_df = format_for_export(matrix, percent_sign, fraction_to_percent)
                    n_label_cols = len(export_df.columns) - n_cols
                    dash["excel"] = to_excel_bytes(export_df, n_label_cols)
                else:
                    dash["excel_note"] = (
                        f"The Excel file was not built because the dashboard has {cells:,} cells "
                        f"(Excel limit in this app: {EXCEL_MAX_CELLS:,}). Use the CSV download, "
                        "or add filters / select fewer months to get an Excel file."
                    )
            st.session_state["dash"] = dash
        except ValueError as exc:
            st.error(str(exc))
        except MemoryError:
            st.error(
                "The app ran out of memory. Add filters, choose fewer columns on the left, "
                "or select fewer months, then click Generate Dashboard again."
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong while building the dashboard: {exc}")
        finally:
            progress.empty()

    # ---- Show the saved dashboard (nothing is recalculated here) --------------
    dash = st.session_state.get("dash")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows in file", f"{len(df):,}")
    c2.metric("Rows after filters", f"{n_filtered:,}")
    c3.metric("Dashboard rows", f"{dash['n_rows']:,}" if dash else "-")
    c4.metric("Date columns", f"{dash['n_dates']:,}" if dash else "-")
    st.caption(f"Date range in file (after filters): {lo:%d-%b-%Y} to {hi:%d-%b-%Y}")

    if dash is None:
        st.info("Choose your options in the sidebar, then click **Generate Dashboard**.")
        return

    if dash["settings"] != settings:
        st.warning("Your selections changed after this dashboard was generated. "
                   "Click **Generate Dashboard** to update it.")
    if dash["skipped"]:
        st.warning(f"{dash['skipped']} row(s) skipped because the start or end date couldn't be read.")
    if dash["shown"] is None:
        st.info("Nothing to show with the current filters and months.")
        return
    if dash["n_dates"] > 400:
        st.warning("That's a lot of date columns. Use 'Months shown' or the date range override to narrow it.")
    if dash["preview_note"]:
        st.caption(dash["preview_note"])

    shown = dash["shown"]
    try:
        st.dataframe(shown, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(shown, hide_index=True, use_container_width=True)

    d1, d2, _ = st.columns([1, 1, 4])
    if dash["excel"] is not None:
        d1.download_button(
            "Download Excel",
            dash["excel"],
            "dashboard.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    d2.download_button(
        "Download CSV",
        dash["csv"],
        "dashboard.csv",
        "text/csv",
    )
    if dash["excel_note"]:
        st.info(dash["excel_note"])


if __name__ == "__main__":
    main()