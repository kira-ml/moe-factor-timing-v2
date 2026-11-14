"""Ken French Data Library ingestion.

The benchmark uses the Fama-French five factors plus momentum. The source
files are monthly returns in percent, reconstructed by CRSP. They are converted
to decimal units and frozen in ``data/reference`` so the default run is
offline and reproducible.

Important limitation: the Ken French files are rebuilt from the current CRSP
database. They are a research benchmark, not a literal point-in-time vintage.
The data contract records this explicitly.
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..utils.io import ensure_dir, sha256_file, write_json
from .schema import CMA, FACTOR_COLUMNS, HML, MKT_RF, MOM, RMW, RF, SMB

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

_ROW = re.compile(r"^\s*(\d{6})\s*,(.*)$")
_USER_AGENT = "moe-factor-timing/2.0 (research benchmark)"


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _read_first_csv(zip_bytes: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise ValueError("downloaded archive contains no CSV file")
        return archive.read(names[0]).decode("latin-1")


def parse_monthly_section(text: str, expected_columns: tuple[str, ...]) -> pd.DataFrame:
    """Parse the monthly block of a Ken French CSV.

    The files contain commentary, a monthly section, a blank line, and an
    annual section. Only the first block of six-digit YYYYMM rows is read.
    """
    header_index: int | None = None
    for i, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith(",") and all(name in stripped for name in expected_columns):
            header_index = i
            break
    if header_index is None:
        raise ValueError(f"could not find header for columns {expected_columns}")

    rows: list[tuple[str, list[str]]] = []
    started = False
    for line in text.splitlines()[header_index + 1 :]:
        match = _ROW.match(line)
        if match:
            started = True
            rows.append((match.group(1), [part.strip() for part in match.group(2).split(",")]))
        elif started:
            break
    if not rows:
        raise ValueError("no monthly rows found")

    index = pd.to_datetime([row[0] for row in rows], format="%Y%m") + pd.offsets.MonthEnd(0)
    values = [[float(value) for value in row[1][: len(expected_columns)]] for row in rows]
    frame = pd.DataFrame(values, index=index, columns=list(expected_columns))
    frame.index.name = "date"
    return frame


def fetch_reference_data(output_dir: str | Path, force: bool = False) -> dict:
    """Download and freeze the reference factor panel.

    Returns a metadata dictionary describing exactly what was written.
    """
    output = ensure_dir(Path(output_dir))
    factors_path = output / "factors.csv"
    metadata_path = output / "metadata.json"
    if factors_path.exists() and not force:
        raise FileExistsError(
            f"{factors_path} already exists; pass force=True to replace the reference snapshot"
        )

    ff5_text = _read_first_csv(_download(FF5_URL))
    mom_text = _read_first_csv(_download(MOM_URL))

    ff5 = parse_monthly_section(ff5_text, ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"))
    mom = parse_monthly_section(mom_text, ("Mom",))
    ff5 = ff5.rename(columns={"Mkt-RF": MKT_RF})
    mom = mom.rename(columns={"Mom": MOM})

    panel = ff5.join(mom, how="inner")
    missing = [column for column in (*FACTOR_COLUMNS, RF) if column not in panel.columns]
    if missing:
        raise ValueError(f"missing expected columns: {missing}")
    panel = panel.loc[:, [*FACTOR_COLUMNS, RF]]
    panel = panel.div(100.0)  # percent -> decimal
    panel.index.name = "date"
    panel.to_csv(factors_path, float_format="%.10f")

    metadata = {
        "source": "Ken French Data Library",
        "source_urls": [FF5_URL, MOM_URL],
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "frequency": "monthly",
        "units": "decimal excess returns (0.01 == 1%); RF is a decimal total return",
        "columns": [*FACTOR_COLUMNS, RF],
        "factor_definitions": {
            MKT_RF: "Market excess return",
            SMB: "Small-minus-big size factor",
            HML: "High-minus-low value factor",
            RMW: "Robust-minus-weak profitability factor",
            CMA: "Conservative-minus-aggressive investment factor",
            MOM: "Momentum factor (12-2 prior return)",
            RF: "One-month Treasury bill return",
        },
        "start": panel.index.min().strftime("%Y-%m-%d"),
        "end": panel.index.max().strftime("%Y-%m-%d"),
        "n_rows": int(len(panel)),
        "file_sha256": sha256_file(factors_path),
        "citation": (
            "Fama, E. F., & French, K. R. (1993). Common risk factors in the returns "
            "on stocks and bonds. Journal of Financial Economics, 33(1), 3-56. "
            "Data: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
        ),
        "point_in_time_caveat": (
            "Ken French factors are reconstructed from the current CRSP database and "
            "are not a literal point-in-time vintage. This benchmark therefore tests "
            "model behaviour on a research panel, not live tradability."
        ),
    }
    write_json(metadata_path, metadata)
    return metadata
