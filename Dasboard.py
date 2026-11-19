"""
Dynamic date-range dashboard
============================
Upload any Excel / CSV that has a start-date column, an end-date column and one
or more value columns. Pick the columns you want to see, add filters, and the
app spreads each row's value across every day from start to end:

    SKU ID     4-Nov-2026  5-Nov-2026  6-Nov-2026 ...
    40325899   656.25      656.25      -          ...

Everything is chosen by you from the headers of the uploaded file:
    - Start date column and End date column (dates are read as DD-MM-YYYY)
    - Columns to show on the left (SKU ID, Brand, Manufacturer ...)
    - Value column(s) (margin, cost, discount ...)
    - Filters on any column
    - Which months to show
    - Download of the dashboard as Excel / CSV

Run:
    pip install streamlit pandas openpyxl
    streamlit run date_range_dashboard.py
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
MAX_EXPANDED_ROWS = 5_000_000


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def parse_dates(series: pd.Series) -> pd.Series:
    """Read dates as day-first (DD-MM-YYYY). Handles real Excel dates and text with a time."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.normalize()

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
# Core logic
# --------------------------------------------------------------------------- #
def date_bounds(df, start_col, end_col):
    """Return (earliest start, latest end, number of readable rows)."""
    s = parse_dates(df[start_col])
    e = parse_dates(df[end_col])
    good = s.notna() & e.notna()
    if not good.any():
        return None, None, 0
    lo = min(s[good].min(), e[good].min())
    hi = max(s[good].max(), e[good].max())
    return lo, hi, int(good.sum())


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
):
    """
    One row per combination of the shown columns (plus 'Metric' if more than one
    value column), one column per date. Returns (matrix, number_of_skipped_rows).
    """
    work = df.copy()
    s = parse_dates(work[start_col])
    e = parse_dates(work[end_col])
    good = s.notna() & e.notna()
    bad_rows = int((~good).sum())
    work, s, e = work[good].copy(), s[good], e[good]
    if work.empty:
        return pd.DataFrame(), bad_rows

    swap = s > e
    s2 = s.where(~swap, e)
    e2 = e.where(~swap, s)

    for c in show_cols:
        work[c] = to_label(work[c])

    lengths = ((e2 - s2).dt.days + 1).to_numpy()
    total = int(lengths.sum())
    if total > MAX_EXPANDED_ROWS:
        raise ValueError(
            f"This would create {total:,} day-level records. "
            "Narrow the date range or add filters."
        )

    idx = np.repeat(np.arange(len(work)), lengths)
    offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    day = s2.iloc[idx].reset_index(drop=True) + pd.to_timedelta(offsets, unit="D")
    base = work.iloc[idx].reset_index(drop=True)

    frames = []
    for v in value_cols:
        part = base[show_cols].copy()
        part["__date__"] = day
        part["__value__"] = clean_numeric(base[v]).to_numpy()
        part["Metric"] = v
        frames.append(part)
    long = pd.concat(frames, ignore_index=True)
    long = long[long["__value__"].notna()]

    lo = pd.Timestamp(range_start) if range_start is not None else day.min()
    hi = pd.Timestamp(range_end) if range_end is not None else day.max()
    long = long[(long["__date__"] >= lo) & (long["__date__"] <= hi)]

    all_dates = pd.date_range(lo, hi, freq="D")
    if visible_months is not None:
        keep = [d for d in all_dates if d.strftime("%b-%Y") in visible_months]
        all_dates = pd.DatetimeIndex(keep)
        long = long[long["__date__"].dt.strftime("%b-%Y").isin(visible_months)]

    if long.empty or len(all_dates) == 0:
        return pd.DataFrame(), bad_rows

    index_cols = list(show_cols)
    if len(value_cols) > 1:
        long["Metric"] = pd.Categorical(long["Metric"], categories=value_cols, ordered=True)
        index_cols.append("Metric")
    else:
        long = long.drop(columns=["Metric"])
    if not index_cols:
        long["__all__"] = "All"
        index_cols = ["__all__"]

    pivot = long.pivot_table(
        index=index_cols,
        columns="__date__",
        values="__value__",
        aggfunc=AGG_FUNCS[agg],
        observed=True,
    )
    pivot = pivot.reindex(columns=all_dates)
    if index_cols == ["__all__"]:
        pivot.index.name = ""
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


def to_excel_bytes(export_df: pd.DataFrame, n_label_cols: int) -> bytes:
    """Excel file with bold header, frozen label columns, shaded value cells."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Dashboard")
        ws = writer.sheets["Dashboard"]
        fill = PatternFill("solid", start_color="E6F1FB", end_color="E6F1FB")

        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")
        for row in ws.iter_rows(min_row=2, min_col=n_label_cols + 1):
            for cell in row:
                cell.alignment = Alignment(horizontal="center")
                if cell.value not in (None, "-"):
                    cell.fill = fill
        ws.freeze_panes = ws.cell(row=2, column=n_label_cols + 1)

        for i, col in enumerate(export_df.columns, start=1):
            if i <= n_label_cols:
                longest = max([len(str(col))] + [len(str(v)) for v in export_df[col].head(500)])
                ws.column_dimensions[get_column_letter(i)].width = min(max(longest + 2, 10), 45)
            else:
                ws.column_dimensions[get_column_letter(i)].width = 13
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Streamlit app
# --------------------------------------------------------------------------- #
def main():
    import streamlit as st

    st.set_page_config(page_title="Date range dashboard", layout="wide")
    st.title("Date range dashboard")

    @st.cache_data(show_spinner="Reading file...")
    def read_table(data: bytes, name: str, sheet, header_row: int):
        if name.lower().endswith((".csv", ".txt")):
            try:
                return pd.read_csv(io.BytesIO(data), header=header_row - 1)
            except UnicodeDecodeError:
                return pd.read_csv(io.BytesIO(data), header=header_row - 1, encoding="latin-1")
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=header_row - 1)

    @st.cache_data
    def sheet_names(data: bytes):
        return pd.ExcelFile(io.BytesIO(data)).sheet_names

    uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xlsm", "xls", "csv"])
    if uploaded is None:
        st.info("Upload a file, then choose its start date, end date, value and display columns.")
        return

    raw = uploaded.getvalue()
    is_csv = uploaded.name.lower().endswith((".csv", ".txt"))
    k = uploaded.name

    st.sidebar.header("File")
    sheet = None
    if not is_csv:
        names = sheet_names(raw)
        sheet = st.sidebar.selectbox("Sheet", names, key=f"{k}_sheet") if len(names) > 1 else names[0]
    header_row = st.sidebar.number_input("Header row (1 = first row)", 1, 50, 1, key=f"{k}_hdr")

    try:
        df = read_table(raw, uploaded.name, sheet, int(header_row))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't read the file: {exc}")
        return
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    with st.expander(f"Uploaded data preview ({len(df):,} rows, {len(cols)} columns)"):
        st.dataframe(df.head(200), hide_index=True)

    def idx(guess):
        return cols.index(guess) if guess in cols else 0

    promo_cols = [c for c in cols if "promo" in c.lower()]

    st.sidebar.header("1. Dates")
    start_guess = guess_column(cols, ["start date", "from date", "start", "from"], exclude=promo_cols)
    end_guess = guess_column(cols, ["end date", "to date", "end", "till", "to "], exclude=promo_cols)
    start_col = st.sidebar.selectbox("Start date column", cols, index=idx(start_guess), key=f"{k}_sd")
    end_col = st.sidebar.selectbox("End date column", cols, index=idx(end_guess), key=f"{k}_ed")

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

    st.sidebar.header("3. Columns shown")
    other_cols = [c for c in cols if c not in (start_col, end_col)]
    sku_guess = guess_column(other_cols, ["sku id", "sku", "item", "product"])
    show_cols = st.sidebar.multiselect(
        "Columns on the left (e.g. SKU, Brand)",
        other_cols,
        default=[sku_guess] if sku_guess else other_cols[:1],
        key=f"{k}_show",
    )

    st.sidebar.header("4. Filters")
    filter_cols = st.sidebar.multiselect("Filter on columns", cols, key=f"{k}_fcols")
    filtered = df
    for fc in filter_cols:
        labels = to_label(filtered[fc])
        options = sorted(labels.unique().tolist())
        chosen = st.sidebar.multiselect(fc, options, key=f"{k}_f_{fc}")
        if chosen:
            filtered = filtered[labels.isin(chosen)]

    if not value_cols:
        st.info("Choose at least one value column in the sidebar.")
        return

    lo, hi, readable = date_bounds(filtered, start_col, end_col)
    if readable == 0:
        st.error(
            f"Couldn't read dates from '{start_col}' or '{end_col}' (expected DD-MM-YYYY). "
            "Check the date column choice."
        )
        return

    st.sidebar.header("5. Date range")
    st.sidebar.caption(f"Detected: {lo:%d-%b-%Y} to {hi:%d-%b-%Y}")
    range_start, range_end = lo, hi
    if st.sidebar.checkbox("Override date range", key=f"{k}_ovr"):
        picked = st.sidebar.date_input("From / to", value=(lo.date(), hi.date()), key=f"{k}_rng")
        if isinstance(picked, tuple) and len(picked) == 2:
            range_start, range_end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])

    months = list(dict.fromkeys(d.strftime("%b-%Y") for d in pd.date_range(range_start, range_end, freq="D")))
    chosen_months = st.sidebar.multiselect(
        "Months shown", months, default=months, key=f"{k}_months_{months[0]}_{months[-1]}"
    )

    st.sidebar.header("6. Number format")
    sample = pd.concat([clean_numeric(filtered[v]).dropna() for v in value_cols])
    looks_fraction = len(sample) > 0 and sample.min() >= 0 and sample.max() <= 1
    fraction_to_percent = st.sidebar.checkbox(
        "Convert fractions to percent (0.1 becomes 10)", value=bool(looks_fraction), key=f"{k}_frac"
    )
    percent_sign = st.sidebar.checkbox("Show % sign", value=False, key=f"{k}_pct")

    try:
        matrix, skipped = build_matrix(
            filtered, start_col, end_col, value_cols, show_cols, agg,
            range_start, range_end, set(chosen_months),
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows in file", f"{len(df):,}")
    c2.metric("Rows after filters", f"{len(filtered):,}")
    c3.metric("Dashboard rows", f"{len(matrix):,}")
    c4.metric("Date columns", f"{matrix.shape[1] if not matrix.empty else 0:,}")
    st.caption(f"Date range in file: {lo:%d-%b-%Y} to {hi:%d-%b-%Y}")

    if skipped:
        st.warning(f"{skipped} row(s) skipped because the start or end date couldn't be read.")
    if matrix.empty:
        st.info("Nothing to show with the current filters and months.")
        return
    if matrix.shape[1] > 400:
        st.warning("That's a lot of date columns. Use 'Months shown' or the date range override to narrow it.")

    shown = format_for_display(matrix, percent_sign, fraction_to_percent)
    try:
        st.dataframe(shown, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(shown, hide_index=True, use_container_width=True)

    export_df = format_for_export(matrix, percent_sign, fraction_to_percent)
    n_label_cols = len(export_df.columns) - matrix.shape[1]

    d1, d2, _ = st.columns([1, 1, 4])
    d1.download_button(
        "Download Excel",
        to_excel_bytes(export_df, n_label_cols),
        "dashboard.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    d2.download_button(
        "Download CSV",
        export_df.to_csv(index=False).encode("utf-8"),
        "dashboard.csv",
        "text/csv",
    )


if __name__ == "__main__":
    main()

    "python -m streamlit run Dasboard.py "