"""Prediction tool — score a registered model on a registered dataset.

Looks up the model from the session registry, reconstructs the design
matrix via patsy on the formula RHS, applies the appropriate inverse
link (or threshold for classification), and returns one row per
non-dropped input row with the source ``row_index`` preserved for
SQL joinability.

See ``docs/proposals/model_registry.md`` §``predict`` and proposal
§"Decisions — resolved from v1 open questions" for the locked semantics.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from data_analyst_mcp import session
from data_analyst_mcp.errors import build_error
from data_analyst_mcp.formatting import truncate_rows
from data_analyst_mcp.recorder import get_recorder

logger = logging.getLogger(__name__)


def _materialize_dataframe(name: str) -> Any:
    """Materialize a registered dataset as a pandas DataFrame via DuckDB."""
    con = session.get_connection()
    return con.execute(f'SELECT * FROM "{name}"').df()


def _patsy() -> Any:
    """Return ``patsy`` as untyped — same convention as the other tools."""
    import patsy as _patsy_mod  # type: ignore[reportMissingTypeStubs]

    return _patsy_mod


def _patsy_dmatrix(rhs: str, df: Any) -> Any:
    """Return a patsy design matrix as untyped."""
    return _patsy().dmatrix(rhs, df, return_type="dataframe")


def _formula_rhs(formula: str) -> str:
    """Return the right-hand-side of a Wilkinson formula (everything after ``~``)."""
    if "~" not in formula:
        return formula.strip()
    return formula.split("~", 1)[1].strip()


class PredictInput(BaseModel):
    """Inputs for ``predict`` — see proposal §Input."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(
        ...,
        description=(
            "Registry handle for a previously-fitted model. Use list_models "
            "to discover available names."
        ),
    )
    dataset: str = Field(
        ...,
        description=(
            "Registered dataset name to score. Must contain every predictor "
            "referenced by the model's formula; the outcome column is optional."
        ),
    )
    output: Literal["link", "response", "class"] = Field(
        default="response",
        description=(
            "Prediction mode. 'link' = linear predictor η; 'response' = "
            "inverse-link applied (probability for logistic, expected count "
            "for Poisson / negbin, identity for OLS); 'class' = thresholded "
            "class label, logistic-only."
        ),
    )
    threshold: float = Field(
        default=0.5,
        description=(
            "Decision threshold for output='class'. Must be in the open "
            "interval (0, 1); endpoints rejected."
        ),
    )
    limit: int = Field(
        default=50,
        description="Page size — same pagination contract as `query`.",
    )
    cursor: str | None = Field(
        default=None,
        description=(
            "Pagination cursor — stringified row index in the source dataset. "
            "Pass back the cursor from a previous response to fetch the next page."
        ),
    )
    include_se: bool = Field(
        default=False,
        description=(
            "OLS only: return per-row prediction standard errors and 95% "
            "confidence intervals (se_mean / mean_ci_lower / mean_ci_upper). "
            "GLM prediction intervals are deferred — proper delta-method "
            "implementations are their own proposal."
        ),
    )


def _record_predict(payload: PredictInput, result: dict[str, Any]) -> None:
    """Append a markdown + code cell pair for the predict call.

    The code cell rehydrates predictions via ``<model_name>.predict(...)``
    on the dataframe materialized in the setup cell — no statsmodels
    Results pickle, just a fresh in-notebook predict call against the
    re-fit model object."""
    if not result.get("ok"):
        return
    md_lines = [
        f"### Predicted `{payload.model_name}` on `{payload.dataset}`",
        f"- Output mode: {result['output_mode']}",
        f"- {result['total_rows']} predictions, {result['dropped_rows']} dropped",
    ]
    md = "\n".join(md_lines)
    out_mode = result["output_mode"]
    # Use the kind from the registered model to emit the right snippet.
    entry = session.get_model(payload.model_name)
    if entry is None:  # pragma: no cover - guarded earlier in the entry point
        return
    if out_mode == "link":
        which = ', which="linear"' if entry.kind == "logistic" else ""
        code = f"preds = {payload.model_name}.predict({payload.dataset}_df{which})\npreds.head()"
    elif out_mode == "class":
        code = (
            f"probs = {payload.model_name}.predict({payload.dataset}_df)\n"
            f"preds = (probs >= {payload.threshold}).astype(int)\n"
            f"preds.head()"
        )
    else:  # response
        code = f"preds = {payload.model_name}.predict({payload.dataset}_df)\npreds.head()"
    get_recorder().record(markdown=md, code=code, tool_name="predict")


def predict(payload: PredictInput) -> dict[str, Any]:
    """Score a registered model on a registered dataset."""
    # Threshold validation — must lie in the *open* interval (0, 1).
    if not (0.0 < payload.threshold < 1.0):
        return build_error(
            type="threshold_out_of_range",
            message=(f"threshold must be in the open interval (0, 1); got {payload.threshold!r}."),
            hint="Use a value strictly between 0 and 1 (e.g. 0.5).",
        )

    entry = session.get_model(payload.model_name)
    if entry is None:
        known = sorted(session.get_models().keys())
        return build_error(
            type="model_not_found",
            message=f"No model named {payload.model_name!r} registered.",
            hint=f"Known model names: {known}." if known else "Registry is empty.",
        )

    datasets = session.get_datasets()
    if payload.dataset not in datasets:
        return build_error(
            type="dataset_not_found",
            message=f"No dataset named {payload.dataset!r} registered.",
            hint="Call list_datasets to see what is available.",
        )

    # Output-mode dispatch validation: class is logistic-only, include_se
    # is OLS-only. Both checks before any patsy work so error types stay
    # deterministic.
    if payload.output == "class" and entry.kind != "logistic":
        return build_error(
            type="class_output_requires_logistic",
            message=(
                f"output='class' requires a logistic model; "
                f"{payload.model_name!r} is kind={entry.kind!r}."
            ),
            hint="Use output='response' for non-logistic models.",
        )
    if payload.include_se and entry.kind != "ols":
        return build_error(
            type="include_se_requires_ols",
            message=(
                f"include_se=True requires an OLS model; "
                f"{payload.model_name!r} is kind={entry.kind!r}."
            ),
            hint=(
                "Prediction SEs are only meaningful for OLS in this proposal; "
                "defer GLM intervals to a follow-up."
            ),
        )

    df = _materialize_dataframe(payload.dataset)

    # Predictor presence check — friendlier than the patsy NameError. Parse
    # the RHS conservatively: any token that's an identifier and not a
    # patsy keyword is treated as a column reference.
    missing = _missing_predictor_columns(entry.formula, df.columns)
    if missing:
        return build_error(
            type="missing_predictors",
            message=(f"Dataset {payload.dataset!r} is missing predictors: {missing}."),
            hint="Add the missing columns or use a dataset that contains every predictor.",
        )

    import numpy as np

    rhs = _formula_rhs(entry.formula)
    try:
        # NA_action='drop' is the patsy default — rows with NaN predictors
        # are dropped. We capture which rows survived via the design info's
        # row index.
        design: Any = _patsy_dmatrix(rhs, df)
    except Exception as exc:
        return build_error(
            type="formula_error",
            message=f"Could not build design matrix for predict: {exc}",
            hint="Check that the dataset's predictor columns match the model's formula.",
        )

    # patsy preserves the original DataFrame index on the resulting design
    # matrix; that's the source-row index we want to surface.
    kept_idx: list[int] = [int(i) for i in design.index]
    dropped_rows = int(len(df) - len(kept_idx))

    m: Any = entry._result  # type: ignore[reportPrivateUsage]
    # Subset the dataframe to patsy-kept rows so predictions align 1:1
    # with kept_idx; statsmodels' predict() either silently drops NaN
    # rows (output count < len(df), order undefined) or fills with NaN
    # depending on the kind — neither is convenient, so we filter ourselves.
    df_kept: Any = df.loc[kept_idx]
    y: Any
    if payload.output == "link":
        # statsmodels Logit.predict accepts which="linear"; for OLS / GLM
        # families the linear predictor equals the exog × params dot product.
        if entry.kind == "logistic":
            y = np.asarray(m.predict(df_kept, which="linear"))
        elif entry.kind in ("poisson", "negbin"):
            # Discrete-model GLM-family results don't always accept
            # ``linear=True`` cleanly — take exog @ params for safety.
            y = np.asarray(design.values @ m.params.values)
        else:  # ols → linear == response
            y = np.asarray(m.predict(df_kept))
    elif payload.output == "class":
        probs: Any = np.asarray(m.predict(df_kept))
        y = (probs >= payload.threshold).astype(int)
    else:  # response
        y = np.asarray(m.predict(df_kept))

    # Build the rows. patsy preserved the DataFrame index → the order of
    # `y` matches `kept_idx`.
    rows: list[dict[str, Any]] = []
    if payload.output == "class":
        for src_i, val in zip(kept_idx, y, strict=True):
            rows.append({"row_index": src_i, "y_pred": float(val), "y_class": int(val)})
    else:
        for src_i, val in zip(kept_idx, y, strict=True):
            rows.append({"row_index": src_i, "y_pred": float(val)})

    # OLS prediction SE block — get_prediction is statsmodels' canonical
    # API and gives a tidy summary_frame.
    if payload.include_se and entry.kind == "ols":
        pred: Any = m.get_prediction(df).summary_frame()
        # summary_frame's index aligns with the original DataFrame index;
        # restrict to the rows patsy kept.
        pred_kept: Any = pred.loc[kept_idx]
        se_mean: Any = pred_kept["mean_se"].to_numpy()
        ci_lo: Any = pred_kept["mean_ci_lower"].to_numpy()
        ci_hi: Any = pred_kept["mean_ci_upper"].to_numpy()
        for row, se, lo, hi in zip(rows, se_mean, ci_lo, ci_hi, strict=True):
            row["se_mean"] = float(se)
            row["mean_ci_lower"] = float(lo)
            row["mean_ci_upper"] = float(hi)

    # Cursor: stringified row index in the source dataset. We slice on
    # *predictions*, not source rows, because dropped rows are not in the
    # prediction list at all.
    if payload.cursor is not None:
        try:
            cursor_idx = int(payload.cursor)
        except ValueError:
            return build_error(
                type="invalid_cursor",
                message=f"cursor must be a stringified integer; got {payload.cursor!r}.",
                hint="Pass the cursor value returned by a previous predict call.",
            )
        # Skip every prediction whose source row_index is strictly less
        # than the cursor. This is the semantics callers want — the cursor
        # echoes a source-dataset row index and downstream calls resume
        # from "the next surviving row after that index".
        rows = [r for r in rows if r["row_index"] >= cursor_idx]

    truncated = truncate_rows(rows, payload.limit)
    # ``truncated`` carries `cursor: <limit_int>` when a tail was clipped;
    # convert to the stringified source-row index of the first clipped row.
    next_cursor: str | None
    if truncated["truncated"]:
        # The first row beyond the limit window — its row_index is the
        # next cursor (caller fetches "everything from row_index N onward").
        first_clipped = rows[payload.limit]
        next_cursor = str(first_clipped["row_index"])
    else:
        next_cursor = None

    response = {
        "ok": True,
        "model_name": payload.model_name,
        "dataset": payload.dataset,
        "output_mode": payload.output,
        "predictions": truncated["rows"],
        "dropped_rows": dropped_rows,
        "total_rows": truncated["total_rows"],
        "truncated": truncated["truncated"],
        "cursor": next_cursor,
    }
    _record_predict(payload, response)
    return response


def _missing_predictor_columns(formula: str, columns: Any) -> list[str]:
    """Return predictor column names referenced by the formula but missing.

    Conservative tokenizer: walks the formula RHS character-by-character,
    extracts identifier-like tokens (``[A-Za-z_][A-Za-z0-9_]*``), and
    filters out patsy keywords (``C``, ``I``, ``Q``, ``Treatment``,
    ``Sum``, ``Helmert``, ``Diff``, ``Poly``, ``Center``, ``standardize``,
    ``scale``) plus numeric and arithmetic noise. Any remaining identifier
    not in ``columns`` is reported as missing.
    """
    import re

    cols_set = set(str(c) for c in columns)
    rhs = _formula_rhs(formula)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rhs)
    patsy_keywords = {
        "C",
        "I",
        "Q",
        "Treatment",
        "Sum",
        "Helmert",
        "Diff",
        "Poly",
        "Center",
        "standardize",
        "scale",
        "np",
        "log",
        "exp",
        "sqrt",
        "log2",
        "log10",
    }
    missing: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if tok in patsy_keywords or tok in seen:
            continue
        seen.add(tok)
        if tok not in cols_set:
            missing.append(tok)
    return missing
