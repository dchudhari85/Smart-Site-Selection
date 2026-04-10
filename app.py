from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="SmartSite Select", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
if not DATA_DIR.exists():
    # fallback for when the app file is copied outside the packaged folder
    for probe in [Path.cwd() / "data", Path(__file__).parent.parent / "data"]:
        if probe.exists():
            DATA_DIR = probe
            break

CONFIG_PATH = DATA_DIR / "README.md"
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"

SITE_ACTION_COLUMNS = [
    "site_id",
    "manual_select",
    "preferred",
    "final_status_override",
    "selection_justification",
    "cda_status_override",
    "cra_flag_override",
    "cra_comment",
    "notification_ack",
    "last_updated",
]
SITE_ACTION_TEXT_COLUMNS = [
    "site_id",
    "final_status_override",
    "selection_justification",
    "cda_status_override",
    "cra_flag_override",
    "cra_comment",
    "last_updated",
]
SITE_ACTION_BOOL_COLUMNS = ["manual_select", "preferred", "notification_ack"]

SURVEY_TRACKING_COLUMNS = [
    "site_id",
    "response_received",
    "survey_sent",
    "survey_sent_at",
    "response_received_at",
    "reminder_count",
    "days_open",
    "survey_template",
    "secure_link",
    "last_updated",
]
SURVEY_TRACKING_TEXT_COLUMNS = [
    "site_id",
    "survey_sent_at",
    "response_received_at",
    "survey_template",
    "secure_link",
    "last_updated",
]
SURVEY_TRACKING_BOOL_COLUMNS = ["survey_sent", "response_received"]
SURVEY_TRACKING_NUMERIC_SPEC = {
    "reminder_count": {"default": 0, "dtype": "int"},
    "days_open": {"default": 0, "dtype": "int"},
}

NOTIFICATION_COLUMNS = ["notification_id", "site_id", "type", "priority", "message", "created_at", "acknowledged"]
NOTIFICATION_TEXT_COLUMNS = ["notification_id", "site_id", "type", "priority", "message", "created_at"]
NOTIFICATION_BOOL_COLUMNS = ["acknowledged"]

USER_COLUMNS = ["username", "password", "full_name", "role", "is_active"]
USER_TEXT_COLUMNS = ["username", "password", "full_name", "role"]
USER_BOOL_COLUMNS = ["is_active"]

CHAT_USAGE_COLUMNS = [
    "usage_id",
    "username",
    "full_name",
    "role",
    "timestamp",
    "page_name",
    "prompt",
    "response",
    "used_local_llm",
    "success",
    "error_message",
]
CHAT_USAGE_TEXT_COLUMNS = ["usage_id", "username", "full_name", "role", "timestamp", "page_name", "prompt", "response", "error_message"]
CHAT_USAGE_BOOL_COLUMNS = ["used_local_llm", "success"]

DEFAULT_CHAT_GREETING = (
    "Ask me about site feasibility, qualification status, top sites in a region, "
    "or how the AI score was calculated."
)

TRUE_BOOL_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_BOOL_VALUES = {"0", "false", "f", "no", "n", ""}


def now_ts() -> str:
    return datetime.now().strftime(TIMESTAMP_FMT)


def load_json_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required data file: {path}")
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, name: str) -> None:
    (DATA_DIR / name).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / name, index=False)


def load_or_init(name: str, columns: list[str]) -> pd.DataFrame:
    path = DATA_DIR / name
    if path.exists():
        df = pd.read_csv(path)
        for col in columns:
            if col not in df.columns:
                df[col] = ""
        return df[columns]
    df = pd.DataFrame(columns=columns)
    save_csv(df, name)
    return df


def append_row(name: str, row: dict, columns: list[str]) -> None:
    df = load_or_init(name, columns)
    df.loc[len(df)] = {c: row.get(c, "") for c in columns}
    save_csv(df, name)


def append_audit(action: str, entity_type: str, entity_id: str, details: str) -> None:
    append_row(
        "audit_log.csv",
        {
            "timestamp": now_ts(),
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "details": details,
        },
        ["timestamp", "action", "entity_type", "entity_id", "details"],
    )


def _is_missing(value) -> bool:
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def normalize_text_value(value) -> str:
    if _is_missing(value):
        return ""
    return str(value)


def normalize_bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    if _is_missing(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in TRUE_BOOL_VALUES:
        return True
    if text in FALSE_BOOL_VALUES:
        return False
    return False


def normalize_text_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].apply(normalize_text_value)
    return out


def normalize_bool_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].apply(normalize_bool_value).astype(bool)
    return out


def normalize_numeric_columns(df: pd.DataFrame, spec: dict[str, dict]) -> pd.DataFrame:
    out = df.copy()
    for col, cfg in spec.items():
        default = cfg.get("default", 0)
        dtype = cfg.get("dtype", "float")
        if col not in out.columns:
            out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
        if dtype == "int":
            out[col] = out[col].round().astype(int)
        elif dtype == "float":
            out[col] = out[col].astype(float)
    return out


def normalize_site_actions(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_text_columns(df, SITE_ACTION_TEXT_COLUMNS)
    out = normalize_bool_columns(out, SITE_ACTION_BOOL_COLUMNS)
    return out[SITE_ACTION_COLUMNS]


def normalize_survey_tracking(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_text_columns(df, SURVEY_TRACKING_TEXT_COLUMNS)
    out = normalize_bool_columns(out, SURVEY_TRACKING_BOOL_COLUMNS)
    out = normalize_numeric_columns(out, SURVEY_TRACKING_NUMERIC_SPEC)
    return out[SURVEY_TRACKING_COLUMNS]


def normalize_notifications(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_text_columns(df, NOTIFICATION_TEXT_COLUMNS)
    out = normalize_bool_columns(out, NOTIFICATION_BOOL_COLUMNS)
    return out[NOTIFICATION_COLUMNS]


def default_site_action_row(site_id: str) -> dict:
    return {
        "site_id": site_id,
        "manual_select": False,
        "preferred": False,
        "final_status_override": "",
        "selection_justification": "",
        "cda_status_override": "",
        "cra_flag_override": "",
        "cra_comment": "",
        "notification_ack": False,
        "last_updated": "",
    }


def default_survey_tracking_row(site_id: str, response_received: bool = False) -> dict:
    sent = bool(response_received)
    return {
        "site_id": site_id,
        "response_received": bool(response_received),
        "survey_sent": sent,
        "survey_sent_at": "",
        "response_received_at": "",
        "reminder_count": 0,
        "days_open": 0,
        "survey_template": "",
        "secure_link": "",
        "last_updated": "",
    }


def _ensure_site_rows(df: pd.DataFrame, site_ids: list[str], row_builder) -> tuple[pd.DataFrame, bool]:
    out = df.copy()
    if "site_id" not in out.columns:
        out["site_id"] = ""
    out["site_id"] = out["site_id"].apply(normalize_text_value)
    existing = set(out["site_id"])
    missing = [sid for sid in site_ids if sid not in existing]
    if not missing:
        return out, False
    additions = pd.DataFrame([row_builder(sid) for sid in missing])
    out = pd.concat([out, additions], ignore_index=True)
    return out, True


def load_or_init_site_actions(site_ids: list[str]) -> pd.DataFrame:
    path = DATA_DIR / "site_actions.csv"
    changed = False
    if path.exists():
        df = pd.read_csv(path)
        changed = any(col not in df.columns for col in SITE_ACTION_COLUMNS)
    else:
        df = pd.DataFrame([default_site_action_row(sid) for sid in site_ids])
        save_csv(df, path.name)
        changed = True
    df = normalize_site_actions(df)
    df, added_rows = _ensure_site_rows(df, site_ids, default_site_action_row)
    if added_rows:
        changed = True
    df = normalize_site_actions(df.drop_duplicates(subset=["site_id"], keep="last"))
    if changed:
        save_csv(df, path.name)
    return df


def load_or_init_survey_tracking(site_ids: list[str], feasibility: pd.DataFrame) -> pd.DataFrame:
    path = DATA_DIR / "survey_tracking.csv"
    changed = False
    has_response = feasibility.groupby("site_id")["interest_level"].apply(
        lambda s: s.fillna("").astype(str).str.len().gt(0).any()
    )

    def survey_row_builder(site_id: str) -> dict:
        return default_survey_tracking_row(site_id, bool(has_response.get(site_id, False)))

    if path.exists():
        df = pd.read_csv(path)
        changed = any(col not in df.columns for col in SURVEY_TRACKING_COLUMNS)
    else:
        df = pd.DataFrame([survey_row_builder(sid) for sid in site_ids])
        save_csv(df, path.name)
        changed = True
    df = normalize_survey_tracking(df)
    df, added_rows = _ensure_site_rows(df, site_ids, survey_row_builder)
    if added_rows:
        changed = True
    df = normalize_survey_tracking(df.drop_duplicates(subset=["site_id"], keep="last"))
    if changed:
        save_csv(df, path.name)
    return df


def update_days_open(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_survey_tracking(df)
    sent = pd.to_datetime(out["survey_sent_at"], errors="coerce")
    out["days_open"] = ((pd.Timestamp.now() - sent).dt.days).fillna(0).clip(lower=0).astype(int)
    out.loc[~out["survey_sent"], "days_open"] = 0
    return out


def load_or_init_notifications() -> pd.DataFrame:
    path = DATA_DIR / "notifications.csv"
    changed = False
    if path.exists():
        df = pd.read_csv(path)
        changed = any(col not in df.columns for col in NOTIFICATION_COLUMNS)
    else:
        df = pd.DataFrame(columns=NOTIFICATION_COLUMNS)
        save_csv(df, path.name)
        changed = True
    df = normalize_notifications(df)
    if changed:
        save_csv(df, path.name)
    return df


def default_user_rows() -> list[dict]:
    return [
        {
            "username": "admin",
            "password": "admin123",
            "full_name": "Alex Morgan",
            "role": "Admin",
            "is_active": True,
        },
        {
            "username": "cra_user",
            "password": "cra123",
            "full_name": "Jordan Lee",
            "role": "CRA",
            "is_active": True,
        },
    ]


def load_or_init_users() -> pd.DataFrame:
    path = DATA_DIR / "users.csv"
    changed = False
    if path.exists():
        df = pd.read_csv(path)
        changed = any(col not in df.columns for col in USER_COLUMNS)
    else:
        df = pd.DataFrame(default_user_rows())
        save_csv(df, path.name)
        changed = True

    if df.empty:
        df = pd.DataFrame(default_user_rows())
        changed = True

    df = normalize_text_columns(df, USER_TEXT_COLUMNS)
    df = normalize_bool_columns(df, USER_BOOL_COLUMNS)
    for col in USER_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in USER_TEXT_COLUMNS else False
    df = df[USER_COLUMNS]
    if changed:
        save_csv(df, path.name)
    return df


def load_users() -> pd.DataFrame:
    return load_or_init_users()


def authenticate_user(username: str, password: str) -> dict | None:
    users = load_users().copy()
    user_name_key = normalize_text_value(username).strip().lower()
    password_key = normalize_text_value(password)
    if not user_name_key or not password_key:
        return None

    users["_username_key"] = users["username"].str.strip().str.lower()
    match = users[
        (users["_username_key"] == user_name_key)
        & (users["password"] == password_key)
        & (users["is_active"])
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "username": normalize_text_value(row["username"]),
        "full_name": normalize_text_value(row["full_name"]),
        "role": normalize_text_value(row["role"]),
    }


def load_or_init_chat_usage() -> pd.DataFrame:
    path = DATA_DIR / "chat_usage.csv"
    changed = False
    if path.exists():
        df = pd.read_csv(path)
        changed = any(col not in df.columns for col in CHAT_USAGE_COLUMNS)
    else:
        df = pd.DataFrame(columns=CHAT_USAGE_COLUMNS)
        save_csv(df, path.name)
        changed = True
    df = normalize_text_columns(df, CHAT_USAGE_TEXT_COLUMNS)
    df = normalize_bool_columns(df, CHAT_USAGE_BOOL_COLUMNS)
    df = df[CHAT_USAGE_COLUMNS]
    if changed:
        save_csv(df, path.name)
    return df


def truncate_for_storage(value: str, max_len: int = 2000) -> str:
    text = normalize_text_value(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def append_chat_usage(
    username: str,
    full_name: str,
    role: str,
    page_name: str,
    prompt: str,
    response: str,
    used_local_llm: bool,
    success: bool,
    error_message: str,
) -> None:
    try:
        usage = load_or_init_chat_usage()
        usage.loc[len(usage)] = {
            "usage_id": f"U{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "username": truncate_for_storage(username, 120),
            "full_name": truncate_for_storage(full_name, 200),
            "role": truncate_for_storage(role, 80),
            "timestamp": now_ts(),
            "page_name": truncate_for_storage(page_name, 120),
            "prompt": truncate_for_storage(prompt, 1600),
            "response": truncate_for_storage(response, 1600),
            "used_local_llm": bool(used_local_llm),
            "success": bool(success),
            "error_message": truncate_for_storage(error_message, 400),
        }
        save_csv(usage[CHAT_USAGE_COLUMNS], "chat_usage.csv")
    except Exception:
        # chat logging must never block user interactions
        pass


def reset_chat_history() -> None:
    st.session_state["chat_history"] = [{"role": "assistant", "content": DEFAULT_CHAT_GREETING}]


CONFIG = load_json_config(CONFIG_PATH)
SITES = load_csv("sites.csv")
PIS = load_csv("principal_investigators.csv")
PERF = load_csv("site_performance_history.csv")
FEAS = load_csv("feasibility_responses_new_trial.csv")
REC = load_csv("recommended_top_sites.csv")
ACTIONS = load_or_init_site_actions(SITES["site_id"].astype(str).tolist())
TRACK = update_days_open(load_or_init_survey_tracking(SITES["site_id"].astype(str).tolist(), FEAS))
save_csv(normalize_survey_tracking(TRACK), "survey_tracking.csv")
NOTES = load_or_init_notifications()
AUDIT = load_or_init("audit_log.csv", ["timestamp", "action", "entity_type", "entity_id", "details"])
USERS = load_or_init_users()
CHAT_USAGE = load_or_init_chat_usage()

TRIAL = CONFIG.get("new_trial", {})
TRIAL_TA = TRIAL.get("therapeutic_area", "Oncology")
TRIAL_IND = TRIAL.get("indication", "NSCLC")
TRIAL_PHASE = str(TRIAL.get("phase", "III"))
WEIGHTS = CONFIG.get("scoring_weights", {})


def build_best_pi_lookup(pis: pd.DataFrame, trial_ta: str, trial_indication: str) -> pd.DataFrame:
    required_text = ["site_id", "pi_name", "specialty_therapeutic_area", "specialty_indication"]
    required_numeric = ["years_experience", "completed_trials", "audit_findings_last_3y"]
    pi_df = pis.copy()

    for col in required_text:
        if col not in pi_df.columns:
            pi_df[col] = ""
    for col in required_numeric:
        if col not in pi_df.columns:
            pi_df[col] = 0

    pi_df = normalize_text_columns(pi_df, required_text)
    pi_df = normalize_numeric_columns(
        pi_df,
        {
            "years_experience": {"default": 0, "dtype": "float"},
            "completed_trials": {"default": 0, "dtype": "float"},
            "audit_findings_last_3y": {"default": 0, "dtype": "float"},
        },
    )
    pi_df = pi_df[(pi_df["site_id"] != "") & (pi_df["pi_name"] != "")].copy()
    if pi_df.empty:
        return pd.DataFrame(
            columns=[
                "site_id",
                "matched_pi_name",
                "pi_years_experience",
                "pi_completed_trials",
                "pi_audit_findings_last_3y",
            ]
        )

    trial_ta_norm = normalize_text_value(trial_ta).strip().lower()
    trial_ind_norm = normalize_text_value(trial_indication).strip().lower()
    pi_df["_ta_match"] = pi_df["specialty_therapeutic_area"].str.strip().str.lower().eq(trial_ta_norm)
    pi_df["_ind_match"] = pi_df["specialty_indication"].str.strip().str.lower().eq(trial_ind_norm)
    pi_df["_match_tier"] = 2
    pi_df.loc[pi_df["_ta_match"], "_match_tier"] = 1
    pi_df.loc[pi_df["_ta_match"] & pi_df["_ind_match"], "_match_tier"] = 0

    ranked = pi_df.sort_values(
        ["site_id", "_match_tier", "years_experience", "completed_trials", "pi_name"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    )
    best = ranked.drop_duplicates("site_id", keep="first").rename(
        columns={
            "pi_name": "matched_pi_name",
            "years_experience": "pi_years_experience",
            "completed_trials": "pi_completed_trials",
            "audit_findings_last_3y": "pi_audit_findings_last_3y",
        }
    )
    return best[
        ["site_id", "matched_pi_name", "pi_years_experience", "pi_completed_trials", "pi_audit_findings_last_3y"]
    ]


@st.cache_data(show_spinner=False)
def build_master(sites, pis, perf, feas, rec, actions, track):
    pi_match = build_best_pi_lookup(pis, TRIAL_TA, TRIAL_IND)

    perf_match = perf[(perf["therapeutic_area"] == TRIAL_TA) & (perf["indication"] == TRIAL_IND)].copy()
    perf_agg = perf_match.groupby("site_id", as_index=False).agg(
        avg_enroll_rate_per_month=("avg_enroll_rate_per_month", "mean"),
        screen_fail_rate=("screen_fail_rate", "mean"),
        protocol_deviation_rate=("protocol_deviation_rate", "mean"),
        data_entry_lag_days=("data_entry_lag_days", "mean"),
        retention_rate=("retention_rate", "mean"),
        competing_trials_same_ta=("competing_trials_same_ta", "mean"),
        site_startup_days_hist=("site_startup_days", "mean"),
        actual_enrollment=("actual_enrollment", "sum"),
        target_enrollment=("target_enrollment", "sum"),
    )

    df = sites.merge(pi_match[[c for c in pi_match.columns if c in [
        "site_id", "matched_pi_name", "pi_years_experience", "pi_completed_trials", "pi_audit_findings_last_3y"
    ]]], on="site_id", how="left")
    df = df.merge(perf_agg, on="site_id", how="left")
    df = df.merge(feas, on="site_id", how="left")
    df = df.merge(rec, on="site_id", how="left", suffixes=("", "_rec"))
    df = df.merge(actions, on="site_id", how="left")
    df = df.merge(track, on="site_id", how="left")

    for col in ["interest_level", "est_startup_days", "projected_enroll_rate_per_month", "central_irb_preferred"]:
        rec_col = f"{col}_rec"
        if rec_col in df.columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[rec_col])
            else:
                df[col] = df[rec_col]

    for col in [
        "pi_years_experience", "pi_completed_trials", "pi_audit_findings_last_3y",
        "avg_enroll_rate_per_month", "screen_fail_rate", "protocol_deviation_rate", "data_entry_lag_days",
        "retention_rate", "competing_trials_same_ta", "site_startup_days_hist", "actual_enrollment",
        "target_enrollment", "est_startup_days", "projected_enroll_rate_per_month", "site_selection_score"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["manual_select", "preferred", "survey_sent", "response_received"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_bool_value)
        else:
            df[col] = False

    reminder_series = pd.to_numeric(df["reminder_count"], errors="coerce") if "reminder_count" in df.columns else pd.Series(0, index=df.index)
    days_open_series = pd.to_numeric(df["days_open"], errors="coerce") if "days_open" in df.columns else pd.Series(0, index=df.index)
    central_series = pd.to_numeric(df["central_irb_preferred"], errors="coerce") if "central_irb_preferred" in df.columns else pd.Series(0, index=df.index)
    df["reminder_count"] = reminder_series.fillna(0).astype(int)
    df["days_open"] = days_open_series.fillna(0).astype(int)
    df["central_irb_preferred"] = central_series.fillna(0).astype(int)

    if "matched_pi_name" not in df.columns:
        df["matched_pi_name"] = "No PI on file"
    df["matched_pi_name"] = df["matched_pi_name"].apply(normalize_text_value).replace("", "No PI on file")
    interest_weight = {"High": 100, "Medium": 70, "Low": 35}
    df["interest_score"] = df["interest_level"].map(interest_weight).fillna(0)
    df["ai_rank_score"] = (df["site_selection_score"].fillna(0) * 300).clip(0, 100).round(0)
    df["feasibility_score"] = (
        df["interest_score"] * 0.35
        + df["projected_enroll_rate_per_month"].fillna(df["avg_enroll_rate_per_month"]).fillna(0).clip(0, 10) * 5.5
        + (100 - df["est_startup_days"].fillna(df["site_startup_days_hist"]).fillna(70).clip(0, 120)) * 0.18
        + df["retention_rate"].fillna(0.75) * 20
        + df["central_irb_preferred"] * 8
    ).clip(0, 100).round(0)
    df["qualification_score"] = (
        df["ai_rank_score"] * 0.45 + df["feasibility_score"] * 0.35 + df["pi_years_experience"].fillna(0) * 1.25 - df["pi_audit_findings_last_3y"].fillna(0) * 3.5
    ).clip(0, 100).round(0)

    def risk_bucket(r):
        score = 0
        score += int(r.get("screen_fail_rate", 0) > 0.22)
        score += int(r.get("protocol_deviation_rate", 0) > 0.08)
        score += int(r.get("data_entry_lag_days", 0) > 7)
        score += int(r.get("competing_trials_same_ta", 0) >= 3)
        score += int(r.get("pi_audit_findings_last_3y", 0) >= 2)
        return "High" if score >= 3 else "Medium" if score >= 1 else "Low"

    df["risk_level"] = df.apply(risk_bucket, axis=1)
    default_cda = pd.cut(df["ai_rank_score"], bins=[-1, 60, 84, 100], labels=["Pending", "In Review", "Executed"]).astype(str)
    cda_override = df["cda_status_override"] if "cda_status_override" in df.columns else pd.Series("", index=df.index)
    df["cda_status"] = cda_override.replace("", pd.NA).fillna(default_cda)

    def cra_flag(r):
        override = str(r.get("cra_flag_override", "") or "").strip()
        if override:
            return override
        if r["risk_level"] == "High":
            return "Risk"
        if r["survey_sent"] and (not r["response_received"]) and r["days_open"] > 7:
            return "Feasibility Delay"
        if int(r.get("central_irb_preferred", 0)) == 0:
            return "IRB Review"
        return "None"

    df["cra_flag"] = df.apply(cra_flag, axis=1)
    df["final_status"] = "Backup"
    df.loc[df["preferred"], "final_status"] = "Selected"
    df.loc[df["risk_level"] == "High", "final_status"] = "Rejected"
    override_series = df["final_status_override"] if "final_status_override" in df.columns else pd.Series("", index=df.index)
    override = override_series.replace("", pd.NA)
    df["final_status"] = override.fillna(df["final_status"])
    df["country_label"] = df["country"].replace({
        "US": "United States", "UK": "United Kingdom", "IN": "India", "DE": "Germany", "FR": "France",
        "ES": "Spain", "CN": "China", "JP": "Japan", "CA": "Canada", "AU": "Australia"
    })
    df = df.sort_values(["ai_rank_score", "feasibility_score", "qualification_score"], ascending=False).reset_index(drop=True)
    return df


MASTER = build_master(SITES, PIS, PERF, FEAS, REC, ACTIONS, TRACK)


def clear_and_rerun():
    build_master.clear()
    st.rerun()


def persist_site_action(site_id: str, **updates):
    df = normalize_site_actions(load_or_init_site_actions(SITES["site_id"].astype(str).tolist()))
    site_key = normalize_text_value(site_id)
    if not site_key:
        return

    idx = df.index[df["site_id"] == site_key]
    if len(idx) == 0:
        df = pd.concat([df, pd.DataFrame([default_site_action_row(site_key)])], ignore_index=True)
        i = df.index[-1]
    else:
        i = idx[0]

    for k, v in updates.items():
        if k not in SITE_ACTION_COLUMNS or k in {"site_id", "last_updated"}:
            continue
        if k in SITE_ACTION_BOOL_COLUMNS:
            df.at[i, k] = normalize_bool_value(v)
        else:
            df.at[i, k] = normalize_text_value(v)
    df.at[i, "site_id"] = site_key
    df.at[i, "last_updated"] = normalize_text_value(now_ts())
    save_csv(normalize_site_actions(df), "site_actions.csv")


def persist_bulk_site_action(site_ids: list[str], **updates):
    for sid in site_ids:
        persist_site_action(sid, **updates)


def persist_distribution(site_ids: list[str], template_name: str):
    df = normalize_survey_tracking(load_or_init_survey_tracking(SITES["site_id"].astype(str).tolist(), FEAS))
    ts = now_ts()
    for sid in site_ids:
        site_key = normalize_text_value(sid)
        idx = df.index[df["site_id"] == site_key]
        if len(idx) == 0:
            continue
        i = idx[0]
        df.at[i, "survey_sent"] = True
        df.at[i, "survey_sent_at"] = normalize_text_value(ts)
        df.at[i, "survey_template"] = normalize_text_value(template_name)
        df.at[i, "secure_link"] = f"https://secure-survey.local/{site_key.lower()}"
        df.at[i, "last_updated"] = normalize_text_value(ts)
        append_audit("survey_distributed", "site", site_key, template_name)
    save_csv(normalize_survey_tracking(update_days_open(df)), "survey_tracking.csv")


def persist_reminders(site_ids: list[str]):
    df = normalize_survey_tracking(load_or_init_survey_tracking(SITES["site_id"].astype(str).tolist(), FEAS))
    ts = now_ts()
    for sid in site_ids:
        site_key = normalize_text_value(sid)
        idx = df.index[df["site_id"] == site_key]
        if len(idx) == 0:
            continue
        i = idx[0]
        reminder_value = pd.to_numeric(df.at[i, "reminder_count"], errors="coerce")
        df.at[i, "reminder_count"] = int(0 if _is_missing(reminder_value) else reminder_value) + 1
        df.at[i, "last_updated"] = normalize_text_value(ts)
        append_audit("survey_reminder", "site", site_key, "Reminder sent")
    save_csv(normalize_survey_tracking(update_days_open(df)), "survey_tracking.csv")


def upsert_notification(site_id: str, note_type: str, priority: str, message: str):
    notes = normalize_notifications(load_or_init_notifications())
    existing_ids = pd.to_numeric(notes["notification_id"].str.replace("N", "", regex=False), errors="coerce").dropna()
    next_num = int(existing_ids.max()) + 1 if not existing_ids.empty else 1
    next_id = f"N{next_num:04d}"
    notes.loc[len(notes)] = {
        "notification_id": next_id,
        "site_id": normalize_text_value(site_id),
        "type": normalize_text_value(note_type),
        "priority": normalize_text_value(priority),
        "message": normalize_text_value(message),
        "created_at": normalize_text_value(now_ts()),
        "acknowledged": False,
    }
    save_csv(normalize_notifications(notes), "notifications.csv")


def acknowledge_notification(note_id: str):
    notes = normalize_notifications(load_or_init_notifications())
    idx = notes.index[notes["notification_id"] == normalize_text_value(note_id)]
    if len(idx):
        notes.at[idx[0], "acknowledged"] = True
        save_csv(normalize_notifications(notes), "notifications.csv")


def ranking_explanation(site_row: pd.Series) -> pd.DataFrame:
    mapping = {
        "avg_enroll_rate_per_month_scaled": "Enrollment rate",
        "screen_fail_rate_scaled": "Screen fail rate",
        "protocol_deviation_rate_scaled": "Protocol deviation rate",
        "data_entry_lag_days_scaled": "Data entry lag",
        "retention_rate_scaled": "Retention rate",
        "competing_trials_same_ta_scaled": "Competing trials",
    }
    rows = []
    for col, label in mapping.items():
        value = float(site_row.get(col, 0) or 0)
        weight = float(WEIGHTS.get(col, 0) or 0)
        rows.append({"Factor": label, "Scaled input": round(value, 3), "Weight": weight, "Contribution": round(value * weight, 3)})
    return pd.DataFrame(rows).sort_values("Contribution", ascending=False)


def style_app():
    st.markdown("""
    <style>
    :root {
      --page-bg:#EEF3F8;
      --sidebar-blue:#2F6DB5;
      --panel-dark:#163E73;
      --panel-dark-alt:#1F4E8C;
      --card-white:#FFFFFF;
      --text-dark:#1F2937;
      --text-muted:#6B7280;
      --border:#D7E1EC;
    }
    .stApp { background: var(--page-bg); color: var(--text-dark); }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, var(--sidebar-blue), #2B63A6 72%, #24558F);
      border-right: 1px solid rgba(255,255,255,.2);
    }
    [data-testid="stSidebar"] * { color:#FFFFFF !important; }
    [data-testid="stSidebar"] .stRadio label { background: transparent !important; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1280px; }
    .topbar {
      background: var(--sidebar-blue);
      border-radius: 16px;
      padding: 14px 18px;
      color: #FFFFFF;
      margin-bottom: 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
    }
    .crumb { font-size: 14px; opacity:.95; }
    .search-pill { background: rgba(255,255,255,.94); color: var(--text-muted); border-radius:12px; padding:10px 16px; min-width:260px; text-align:left; }
    .page-title { font-size: 28px; font-weight: 800; color: var(--text-dark); margin-bottom: 2px; }
    .page-sub { color: var(--text-muted); font-size: 15px; margin-bottom: 18px; }
    .metrics { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:14px; margin-bottom:18px; }
    .metric-card { background: var(--panel-dark-alt); color:#FFFFFF; border-radius:18px; padding:18px 20px; box-shadow: 0 2px 8px rgba(28,54,89,.08); }
    .metric-card * { color:#FFFFFF; }
    .metric-card.light { background: var(--card-white); color: var(--text-dark); border: 1px solid var(--border); }
    .metric-card.light * { color: var(--text-dark); }
    .metric-label { font-size:13px; text-transform:uppercase; letter-spacing:.04em; opacity:.92; }
    .metric-value { font-size:22px; font-weight:800; margin-top:6px; }
    .surface { background: var(--card-white); border:1px solid var(--border); border-radius:20px; padding:18px; margin-bottom:16px; box-shadow:0 1px 3px rgba(16,24,40,.04); color: var(--text-dark); }
    .surface * { color: var(--text-dark); }
    .surface .metric-card, .surface .metric-card * { color:#FFFFFF !important; }
    .surface .metric-card.light, .surface .metric-card.light * { color: var(--text-dark) !important; }
    .surface-dark { background: linear-gradient(160deg, var(--panel-dark), var(--panel-dark-alt)); border-radius:20px; padding:18px; color:#FFFFFF; margin-bottom:16px; }
    .surface-dark * { color:#FFFFFF; }
    .section-head { font-size:16px; font-weight:800; margin-bottom:10px; color: var(--text-dark); }
    .surface-dark .section-head { color:#FFFFFF !important; }
    .site-chip { display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:700; }
    .chip-success { background:#dcfce7; color:#166534; }
    .chip-warning { background:#fef3c7; color:#92400e; }
    .chip-danger { background:#fee2e2; color:#b91c1c; }
    .chip-info { background:#dbeafe; color:#1d4ed8; }
    .small-note, .stCaption, .page-sub, .footer-note, .search-pill { color: var(--text-muted); }
    .footer-note { font-size:12px; line-height:1.45; padding:14px 8px 2px 8px; color:#E8F1FE; }
    div[data-testid="stDataFrame"] div[role="table"], div[data-testid="stTable"] table { border-radius:16px; overflow:hidden; }
    div[data-testid="stDataFrame"] [role="columnheader"], div[data-testid="stTable"] th {
      background: #E7EEF7 !important;
      color: var(--text-dark) !important;
      border-bottom: 1px solid var(--border) !important;
    }
    div[data-testid="stDataFrame"] [role="gridcell"], div[data-testid="stTable"] td {
      background: #FFFFFF !important;
      color: var(--text-dark) !important;
      border-bottom: 1px solid #ECF1F7 !important;
    }
    .stTextInput label, .stTextArea label, .stSelectbox label, .stRadio label, .stSlider label,
    .stMultiSelect label, .stCheckbox label, .stNumberInput label {
      color: var(--text-dark) !important;
      font-weight: 600;
    }
    .stTextInput input, .stTextArea textarea, .stNumberInput input,
    .stSelectbox [data-baseweb="select"] > div, .stMultiSelect [data-baseweb="select"] > div {
      background: #FFFFFF !important;
      color: var(--text-dark) !important;
      border: 1px solid var(--border) !important;
    }
    .stRadio [role="radiogroup"] label, .stRadio [role="radiogroup"] p { color: var(--text-dark) !important; }
    .stButton > button, .stDownloadButton > button {
      background: #EDF3FB;
      color: var(--text-dark);
      border: 1px solid #BFD0E5;
      border-radius: 10px;
      font-weight: 700;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
      border-color: #99B5D4;
      color: #112236;
    }
    .stButton > button[kind="primary"] {
      background: var(--panel-dark-alt);
      color: #FFFFFF;
      border: 1px solid #10335D;
      box-shadow: 0 8px 20px rgba(20, 53, 97, .18);
    }
    .stButton > button[kind="primary"]:hover {
      background: #1B4680;
      border-color: #0F2E53;
    }
    .streamlit-expanderHeader, .streamlit-expanderContent, details, details * {
      color: var(--text-dark) !important;
    }
    @media (max-width: 980px) { .metrics { grid-template-columns:1fr 1fr; } }
    </style>
    """, unsafe_allow_html=True)


def render_topbar(title: str):
    st.markdown(
        f"<div class='topbar'><div class='crumb'>SmartSite Select &gt; {title}</div><div class='search-pill'>🔎 Search studies, sites, PIs...</div></div>",
        unsafe_allow_html=True,
    )


def metric_cards(items):
    html = "<div class='metrics'>"
    for i, (label, value, tone) in enumerate(items):
        klass = "metric-card light" if tone == "light" else "metric-card"
        html += f"<div class='{klass}'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def filtered_master(region: str, country: str, institution: str, interest: str, min_score: int) -> pd.DataFrame:
    df = MASTER.copy()
    if region != "All":
        df = df[df["region"] == region]
    if country != "All":
        df = df[df["country"] == country]
    if institution != "All":
        df = df[df["institution_type"] == institution]
    if interest != "All":
        df = df[df["interest_level"].fillna("Unknown") == interest]
    df = df[df["ai_rank_score"] >= min_score]
    return df.reset_index(drop=True)


def render_page_filters(master_df: pd.DataFrame, key_prefix: str) -> dict:
    st.markdown("**Study Filters**")
    region = st.selectbox(
        "Region",
        ["All"] + sorted(master_df["region"].dropna().unique().tolist()),
        key=f"{key_prefix}_region",
    )
    country = st.selectbox(
        "Country",
        ["All"] + sorted(master_df["country"].dropna().unique().tolist()),
        key=f"{key_prefix}_country",
    )
    institution = st.selectbox(
        "Institution",
        ["All"] + sorted(master_df["institution_type"].dropna().unique().tolist()),
        key=f"{key_prefix}_institution",
    )
    interest = st.selectbox(
        "Interest",
        ["All"] + sorted(master_df["interest_level"].dropna().unique().tolist()),
        key=f"{key_prefix}_interest",
    )
    min_ai_rank = st.slider("Min AI Match", 0, 100, 75, key=f"{key_prefix}_min_ai_rank")
    return {
        "region": region,
        "country": country,
        "institution": institution,
        "interest": interest,
        "min_ai_rank": min_ai_rank,
    }


def set_flash_message(message: str, level: str = "success") -> None:
    st.session_state["flash_message"] = {"message": message, "level": level}


def render_flash_message() -> None:
    flash = st.session_state.pop("flash_message", None)
    if not isinstance(flash, dict):
        return
    level = normalize_text_value(flash.get("level", "success")).lower()
    message = normalize_text_value(flash.get("message", "")).strip()
    if not message:
        return
    if level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    elif level == "info":
        st.info(message)
    else:
        st.success(message)


def initialize_auth_state() -> None:
    defaults = {
        "authenticated": False,
        "current_user": "",
        "current_full_name": "",
        "current_role": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def perform_logout() -> None:
    username = normalize_text_value(st.session_state.get("current_user", ""))
    if username:
        append_audit("logout", "user", username, "User logged out")
    st.session_state["authenticated"] = False
    st.session_state["current_user"] = ""
    st.session_state["current_full_name"] = ""
    st.session_state["current_role"] = ""
    st.session_state["page"] = "Study Setup"
    reset_chat_history()
    set_flash_message("Logged out successfully.")
    st.rerun()


def render_login_screen() -> None:
    render_topbar("Login")
    render_flash_message()
    st.markdown(
        "<div class='page-title'>SmartSite Select Login</div>"
        "<div class='page-sub'>Authenticate with a local account to access workflow pages and persistence actions.</div>",
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([1.0, 1.2, 1.0])
    with center:
        with st.container(border=True):
            st.markdown("### Sign in")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login", use_container_width=True, type="primary", key="login_button"):
                user = authenticate_user(username, password)
                if user is None:
                    st.error("Invalid credentials or inactive account.")
                else:
                    st.session_state["authenticated"] = True
                    st.session_state["current_user"] = user["username"]
                    st.session_state["current_full_name"] = user["full_name"]
                    st.session_state["current_role"] = user["role"]
                    reset_chat_history()
                    append_audit(
                        "login",
                        "user",
                        user["username"],
                        f"role={user['role']}; full_name={user['full_name']}",
                    )
                    set_flash_message(f"Welcome {user['full_name']}. Login successful.")
                    st.rerun()
            st.caption("Demo users: admin/admin123 (Admin), cra_user/cra123 (CRA)")


def build_chat_context(page_name: str, master_df: pd.DataFrame, view_df: pd.DataFrame) -> str:
    active_view = view_df if not view_df.empty else master_df
    top_rows = active_view.head(5)
    top_sites = []
    for row in top_rows.itertuples():
        top_sites.append(
            f"- {row.site_name} ({row.country_label}) | AI {int(row.ai_rank_score)} | Feasibility {int(row.feasibility_score)} | PI {row.matched_pi_name}"
        )
    if not top_sites:
        top_sites = ["- No sites available in current filtered view"]

    sent = int(master_df["survey_sent"].sum()) if "survey_sent" in master_df.columns else 0
    received = int(master_df["response_received"].sum()) if "response_received" in master_df.columns else 0
    pending = int(((master_df["survey_sent"]) & (~master_df["response_received"])).sum()) if {"survey_sent", "response_received"}.issubset(master_df.columns) else 0
    reminders = int(master_df["reminder_count"].sum()) if "reminder_count" in master_df.columns else 0
    selected = int((master_df["final_status"] == "Selected").sum()) if "final_status" in master_df.columns else 0
    backup = int((master_df["final_status"] == "Backup").sum()) if "final_status" in master_df.columns else 0
    rejected = int((master_df["final_status"] == "Rejected").sum()) if "final_status" in master_df.columns else 0
    high_match = int((master_df["ai_rank_score"] >= 85).sum()) if "ai_rank_score" in master_df.columns else 0
    high_risk = int((master_df["risk_level"] == "High").sum()) if "risk_level" in master_df.columns else 0
    avg_qualification = float(master_df["qualification_score"].mean()) if "qualification_score" in master_df.columns and not master_df.empty else 0.0

    return "\n".join(
        [
            f"Current workflow page: {page_name}",
            f"Trial context: TA={TRIAL_TA}; Indication={TRIAL_IND}; Phase={TRIAL_PHASE}",
            "Top ranked sites:",
            *top_sites,
            f"Feasibility: sent={sent}, received={received}, pending={pending}, reminders={reminders}",
            f"Final decisions: selected={selected}, backup={backup}, rejected={rejected}",
            f"Qualification metrics: avg_qualification={avg_qualification:.1f}, high_risk={high_risk}, high_match={high_match}",
        ]
    )


def query_local_llm(prompt: str, context: str) -> str:
    system_instruction = (
        "You are the SmartSite Select assistant. Use only the supplied app context. "
        "If information is unavailable, say so briefly and suggest checking the relevant workflow page. "
        "Answer in under 120 words unless the user asks for detail."
    )
    payload = {
        "model": "qwen2.5:7b",
        "stream": False,
        "prompt": (
            f"System instruction:\n{system_instruction}\n\n"
            f"App context:\n{context}\n\n"
            f"User question:\n{prompt}\n\n"
            "Assistant answer:"
        ),
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=25)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("Invalid Ollama response payload")
    answer = normalize_text_value(body.get("response", "")).strip()
    if not answer:
        raise ValueError("Empty Ollama response")
    return answer


def chatbot_answer_fallback(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["top", "best", "recommend"]):
        top = MASTER.iloc[0]
        return f"Top recommended site is {top['site_name']} in {top['country_label']} with AI match {int(top['ai_rank_score'])}% and feasibility {int(top['feasibility_score'])}/100."
    if "europe" in q:
        europe = MASTER[MASTER["region"].str.contains("Europe", case=False, na=False)].head(5)
        if europe.empty:
            return "No Europe sites are currently in the filtered view."
        lines = [f"{i+1}. {r.site_name} ({r.country_label}) – {int(r.ai_rank_score)}%" for i, r in enumerate(europe.itertuples())]
        return "Top European candidates right now:\n\n" + "\n".join(lines)
    if "qualification" in q or "cda" in q:
        sel = int((MASTER["final_status"] == "Selected").sum())
        return f"There are {sel} currently selected sites. CDA status, CRA flags, preferred flags, and comments all persist back to CSV sidecar files."
    if "feasibility" in q:
        sent = int(MASTER["survey_sent"].sum())
        recv = int(MASTER["response_received"].sum())
        return f"Feasibility dashboard shows {recv} received responses out of {sent} sent surveys, with {int(((MASTER['survey_sent']) & (~MASTER['response_received']) & (MASTER['days_open'] > 7)).sum())} SLA breaches."
    return "I can answer questions about site ranking, feasibility, qualification, final selection, navigation, and CSV persistence."


def chatbot_answer(query: str, page_name: str, view_df: pd.DataFrame) -> dict:
    context = build_chat_context(page_name, MASTER, view_df)
    try:
        answer = query_local_llm(query, context)
        return {
            "response": answer,
            "used_local_llm": True,
            "fallback_used": False,
            "success": True,
            "error_message": "",
        }
    except Exception as exc:
        return {
            "response": chatbot_answer_fallback(query),
            "used_local_llm": False,
            "fallback_used": True,
            "success": False,
            "error_message": truncate_for_storage(str(exc), 400),
        }


style_app()

initialize_auth_state()
if not bool(st.session_state.get("authenticated", False)):
    with st.sidebar:
        st.markdown("## SmartSite Select")
        st.caption("AI-driven smart site selection")
        st.markdown("---")
        st.caption("Sign in to access workflow pages and persisted actions.")
    render_login_screen()
    st.stop()

workflow_labels = {
    "Study Setup": "Protocol Definition",
    "Site Filtering": "AI Site Ranking & Filtering",
    "Feasibility Distribution": "Feasibility Distribution",
    "Feasibility Responses": "Feasibility Responses",
    "Feasibility Analysis": "Site Feasibility Analysis",
    "Qualification": "Site Qualification Review",
    "Final Selection": "Final Selection",
    "Chatbot Assistance": "Chatbot Assistance",
    "CRA Notifications": "Notification Center",
}
filter_enabled_pages = {
    "Site Filtering",
    "Feasibility Distribution",
    "Feasibility Responses",
    "Feasibility Analysis",
    "Qualification",
    "Final Selection",
}

workflow_pages = list(workflow_labels.keys())
if "page" not in st.session_state or st.session_state["page"] not in workflow_labels:
    st.session_state["page"] = workflow_pages[0]

filters = {"region": "All", "country": "All", "institution": "All", "interest": "All", "min_ai_rank": 0}
with st.sidebar:
    st.markdown("## SmartSite Select")
    st.caption("AI-driven smart site selection")
    st.caption(f"Signed in as {st.session_state['current_full_name']} ({st.session_state['current_role']})")
    if st.button("Logout", use_container_width=True, key="logout_button"):
        perform_logout()
    st.markdown("---")
    page = st.radio(
        "Workflow",
        workflow_pages,
        index=workflow_pages.index(st.session_state["page"]),
        label_visibility="visible",
    )
    st.session_state["page"] = page
    st.markdown("---")
    if page in filter_enabled_pages:
        page_key = normalize_text_value(page).lower().replace(" ", "_").replace("/", "_")
        filters = render_page_filters(MASTER, key_prefix=f"filters_{page_key}")
    else:
        st.caption("Study filters are hidden on this page.")
    st.markdown("---")
    st.markdown(f"**Therapeutic Area:** {TRIAL_TA}")
    st.markdown(f"**Indication:** {TRIAL_IND}")
    st.markdown(f"**Phase:** {TRIAL_PHASE}")
    st.markdown(f"**Dataset Version:** {CONFIG.get('version','v1')}")
    st.markdown("---")
    st.markdown("<div class='footer-note'>AI Status<br>Models are up to date. Last sync uses the local data folder and persisted app actions.</div>", unsafe_allow_html=True)

if page in filter_enabled_pages:
    VIEW = filtered_master(
        filters["region"],
        filters["country"],
        filters["institution"],
        filters["interest"],
        int(filters["min_ai_rank"]),
    )
else:
    VIEW = MASTER.copy().reset_index(drop=True)

render_topbar(workflow_labels[page])
render_flash_message()

if page == "Study Setup":
    st.markdown("<div class='page-title'>Protocol Definition</div><div class='page-sub'>Configure clinical trial parameters that guide AI-driven site ranking and feasibility projections.</div>", unsafe_allow_html=True)
    left, right = st.columns([1.65, 0.8])
    with left:
        with st.container(border=True):
            st.markdown("<div class='section-head'>General Information</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            study_title = c1.text_input("Study Title", value=f"Phase {TRIAL_PHASE} Evaluation of {TRIAL_IND} in {TRIAL_TA}")
            protocol_id = c2.text_input("Protocol ID", value=f"ST-{TRIAL_PHASE}-{TRIAL_TA[:3].upper()}-03")
            st.divider()
            st.markdown("<div class='section-head'>Clinical Parameters</div>", unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            ta = c3.selectbox(
                "Therapeutic Area",
                sorted(FEAS["new_trial_ta"].dropna().unique().tolist()),
                index=sorted(FEAS["new_trial_ta"].dropna().unique().tolist()).index(TRIAL_TA)
                if TRIAL_TA in sorted(FEAS["new_trial_ta"].dropna().unique().tolist())
                else 0,
            )
            indication = c4.selectbox(
                "Indication",
                sorted(FEAS["new_trial_indication"].dropna().unique().tolist()),
                index=sorted(FEAS["new_trial_indication"].dropna().unique().tolist()).index(TRIAL_IND)
                if TRIAL_IND in sorted(FEAS["new_trial_indication"].dropna().unique().tolist())
                else 0,
            )
            phase = st.radio("Study Phase", ["I", "I/II", "II", "III", "IV"], index=3, horizontal=True)
            st.divider()
            st.markdown("<div class='section-head'>Population & Geography</div>", unsafe_allow_html=True)
            c5, c6, c7, c8 = st.columns([1.25, 1.0, 1.0, 0.95])
            c5.number_input("Total Target Enrollment", value=450, step=10)
            c6.number_input("Min Age (in years)", value=18, step=1)
            c7.number_input("Max Age (in years)", value=85, step=1)
            c8.selectbox("Gender", ["All", "Male", "Female"], index=0)
            geos = st.multiselect(
                "Target Geographies",
                sorted(MASTER["region"].dropna().unique().tolist()),
                default=sorted(MASTER["region"].dropna().unique().tolist())[:3],
            )
            with st.expander("Advanced Feasibility Parameters", expanded=True):
                a1, a2 = st.columns(2)
                biomarker_required = a1.checkbox("Require Biomarker Testing", value=True)
                a1.caption("Prioritize sites with in-house genomic sequencing capabilities.")
                rare_disease_protocol = a1.checkbox("Rare Disease Protocol", value=False)
                a1.caption("Adjusts AI modeling for hyper-specific patient populations.")
                a2.selectbox("Competitive Trial Density Tolerance", ["Low", "Medium (Standard)", "High"], index=1)
                a2.selectbox("IRB Preference", ["Either", "Central Preferred", "Local Accepted"], index=1)
    with right:
        with st.container(border=True):
            st.markdown("<div class='section-head'>AI Feasibility Projection</div>", unsafe_allow_html=True)
            conf = int(min(99, max(75, VIEW["ai_rank_score"].head(30).mean() if not VIEW.empty else 92)))
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>Estimated Model Confidence</div><div class='metric-value' style='font-size:48px'>{conf}%</div><div>Based on similar historical trials</div></div>",
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.markdown(
                f"<div class='metric-card' style='margin-top:12px'><div class='metric-label'>Est. Sites Needed</div><div class='metric-value'>{max(12, int((450/ max(MASTER['projected_enroll_rate_per_month'].fillna(3).median(),1))*0.35))} - {max(18, int((450/ max(MASTER['projected_enroll_rate_per_month'].fillna(3).median(),1))*0.45))}</div></div>",
                unsafe_allow_html=True,
            )
            m2.markdown(
                "<div class='metric-card' style='margin-top:12px'><div class='metric-label'>Enrollment Time</div><div class='metric-value'>14.5 mo</div></div>",
                unsafe_allow_html=True,
            )
            if st.button("Generate AI Recommendations ⚡", use_container_width=True):
                append_audit("study_setup_generate", "protocol", protocol_id, f"TA={ta}; indication={indication}; phase={phase}")
                st.success("Generate AI Recommendations completed. Study setup parameters were captured.")

elif page == "Site Filtering":
    st.markdown("<div class='page-title'>AI Site Ranking & Filtering</div><div class='page-sub'>Review AI-recommended sites, adjust filters, and manually select final candidates for feasibility surveys.</div>", unsafe_allow_html=True)
    metric_cards([
        ("Total Sites Analyzed", f"{len(VIEW):,}", "dark"),
        ("High Match Candidates", int((VIEW["ai_rank_score"] >= 85).sum()), "light"),
        ("Avg. AI Match Score", f"{int(VIEW['ai_rank_score'].mean()) if not VIEW.empty else 0}%", "dark"),
        ("Est. Enrollment Reach", int(VIEW["projected_enroll_rate_per_month"].fillna(0).head(25).sum()), "dark"),
    ])
    left, right = st.columns([0.9, 1.25])
    with left:
        st.markdown("<div class='surface'><div class='section-head'>Global Distribution</div>", unsafe_allow_html=True)
        map_df = VIEW.dropna(subset=["latitude", "longitude"]).copy()
        if not map_df.empty:
            fig = px.scatter_geo(map_df.head(200), lat="latitude", lon="longitude", size="ai_rank_score", hover_name="site_name", color="region")
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='surface'><div class='section-head'>Advanced Filters</div>", unsafe_allow_html=True)
        search_term = st.text_input("Search sites", placeholder="Site name, PI, or ID")
        exp_filter = st.multiselect("PI experience", ["High (10+ years)", "Medium (5-10 years)", "Low (<5 years)"], default=["High (10+ years)"])
        if search_term:
            result = VIEW[
                VIEW["site_name"].str.contains(search_term, case=False, na=False)
                | VIEW["matched_pi_name"].str.contains(search_term, case=False, na=False)
                | VIEW["site_id"].str.contains(search_term, case=False, na=False)
            ]
        else:
            result = VIEW.copy()
        if exp_filter:
            keep = pd.Series(False, index=result.index)
            if "High (10+ years)" in exp_filter:
                keep |= result["pi_years_experience"].fillna(0) >= 10
            if "Medium (5-10 years)" in exp_filter:
                keep |= result["pi_years_experience"].fillna(0).between(5, 9.999)
            if "Low (<5 years)" in exp_filter:
                keep |= result["pi_years_experience"].fillna(0) < 5
            result = result[keep]
        st.caption(f"Showing {len(result)} sites from the current filtered cohort.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        display = result[["site_id", "site_name", "country_label", "matched_pi_name", "pi_years_experience", "ai_rank_score", "risk_level", "manual_select"]].copy()
        display.columns = ["Site ID", "Site Details", "Location", "PI Info", "PI Experience", "AI Match Score", "Risk", "Select"]
        edited = st.data_editor(
            display,
            use_container_width=True,
            hide_index=True,
            disabled=[c for c in display.columns if c != "Select"],
            column_config={
                "Select": st.column_config.CheckboxColumn("Select"),
                "AI Match Score": st.column_config.ProgressColumn("AI Match Score", min_value=0, max_value=100, format="%d%%"),
            },
            key="site_filter_editor",
        )
        if st.button("Proceed to Feasibility →", use_container_width=True, type="primary"):
            selected_ids = edited.loc[edited["Select"], "Site ID"].tolist()
            persist_bulk_site_action(MASTER["site_id"].tolist(), manual_select=False)
            if selected_ids:
                persist_bulk_site_action(selected_ids, manual_select=True)
                for sid in selected_ids:
                    append_audit("manual_select", "site", sid, "Selected from Site Filtering Dashboard")
            set_flash_message(f"Proceed to Feasibility completed. Saved {len(selected_ids)} selected candidate sites.")
            st.session_state["page"] = "Feasibility Distribution"
            clear_and_rerun()
        if not result.empty:
            st.markdown("<div class='surface' style='margin-top:14px'><div class='section-head'>Explainable AI for top site</div>", unsafe_allow_html=True)
            st.dataframe(ranking_explanation(result.iloc[0]), hide_index=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "Feasibility Distribution":
    st.markdown("<div class='page-title'>Feasibility Distribution</div><div class='page-sub'>Review the AI-ranked site list and configure survey distribution parameters with persisted send/reminder tracking.</div>", unsafe_allow_html=True)
    selected = MASTER[MASTER["manual_select"]].copy()
    if selected.empty:
        selected = VIEW.head(12).copy()
    metric_cards([
        ("Total Selected", len(selected), "dark"),
        ("Pending Send", int((~selected["survey_sent"]).sum()), "dark"),
        ("Delivered", int(selected["survey_sent"].sum()), "dark"),
        ("Reminders", int(selected["reminder_count"].sum()), "light"),
    ])
    left, right = st.columns([0.65, 1.45])
    with left:
        st.markdown("<div class='surface'><div class='section-head'>Auto-Select Rules</div>", unsafe_allow_html=True)
        threshold = st.slider("Min. AI Match Score", 50, 100, 85)
        min_trials = st.slider("PI Experience (years)", 0, 20, 5)
        auto_include_top_10 = st.checkbox("Auto-include Top 10%", value=True)
        template = st.text_input("Survey Template", value=f"{TRIAL_TA}-{TRIAL_IND}-feasibility")
        if st.button("Apply Rules", use_container_width=True):
            base = MASTER[(MASTER["ai_rank_score"] >= threshold) & (MASTER["pi_years_experience"].fillna(0) >= min_trials)]
            if auto_include_top_10:
                base = MASTER.head(max(1, int(len(MASTER) * 0.10))).combine_first(base)
            persist_bulk_site_action(MASTER["site_id"].tolist(), manual_select=False)
            persist_bulk_site_action(base["site_id"].tolist(), manual_select=True)
            set_flash_message(f"Apply Rules completed. Updated manual selections for {len(base)} sites.")
            clear_and_rerun()
        chosen = st.multiselect("Distribution list", options=selected["site_name"].tolist(), default=selected["site_name"].head(8).tolist())
        if st.button("Send Feasibility Surveys", use_container_width=True):
            ids = selected[selected["site_name"].isin(chosen)]["site_id"].tolist()
            persist_distribution(ids, template)
            for sid in ids:
                upsert_notification(sid, "Feasibility Survey Submitted", "Medium", f"Survey distributed using template {template}")
            set_flash_message(f"Send Feasibility Surveys completed. Distribution persisted for {len(ids)} sites.")
            clear_and_rerun()
        pending_ids = selected[(selected["survey_sent"]) & (~selected["response_received"])]["site_id"].tolist()
        if st.button("Send Reminders", use_container_width=True):
            persist_reminders(pending_ids)
            for sid in pending_ids:
                upsert_notification(sid, "SLA Breach Warning", "High", "Reminder triggered for pending feasibility survey")
            set_flash_message(f"Send Reminders completed. Reminder counts updated for {len(pending_ids)} sites.")
            clear_and_rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='surface-dark'><div class='section-head' style='color:#fff'>Ranked Sites for Feasibility</div>", unsafe_allow_html=True)
        dist = selected[["site_name", "site_id", "matched_pi_name", "country_label", "ai_rank_score", "survey_sent", "response_received", "reminder_count"]].copy()
        dist["Survey Status"] = dist.apply(lambda r: "Delivered" if r["response_received"] else "Sent" if r["survey_sent"] else "Pending", axis=1)
        dist["AI Match"] = dist["ai_rank_score"].astype(int)
        dist = dist[["site_name", "site_id", "matched_pi_name", "country_label", "AI Match", "Survey Status", "reminder_count"]]
        dist.columns = ["Site Name", "Site ID", "Principal Investigator", "Location", "AI Match", "Survey Status", "Reminders"]
        st.dataframe(dist, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Feasibility Responses":
    st.markdown("<div class='page-title'>Feasibility Responses</div><div class='page-sub'>Monitor survey response rates and identify bottlenecks in site evaluation.</div>", unsafe_allow_html=True)
    sent = int(MASTER["survey_sent"].sum())
    recv = int(MASTER["response_received"].sum())
    breaches = ((MASTER["survey_sent"]) & (~MASTER["response_received"]) & (MASTER["days_open"] > 7)).sum()
    metric_cards([
        ("Total Surveys Sent", sent, "dark"),
        ("Response Rate", f"{round((recv/sent)*100) if sent else 0}%", "dark"),
        ("Pending Responses", int(((MASTER["survey_sent"]) & (~MASTER["response_received"])).sum()), "dark"),
        ("SLA Breaches", int(breaches), "light"),
    ])
    c1, c2 = st.columns([0.8, 1.4])
    with c1:
        st.markdown("<div class='surface'><div class='section-head'>Outreach Breakdown</div>", unsafe_allow_html=True)
        pie = px.pie(values=[recv, max(sent - recv, 0), int(breaches)], names=["Received", "Pending", "Overdue (SLA)"], hole=0.62)
        pie.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='surface'><div class='section-head'>Response Trends</div>", unsafe_allow_html=True)
        trend = pd.DataFrame({"Day": [f"D{i}" for i in range(1, 9)], "Received": [max(0, recv - 7 + i*2) for i in range(8)], "Pending": [max(0, sent - recv - i) for i in range(8)]})
        fig = go.Figure()
        fig.add_bar(x=trend["Day"], y=trend["Received"], name="Received")
        fig.add_bar(x=trend["Day"], y=trend["Pending"], name="Pending")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), barmode="stack")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    tracking = MASTER[["site_name", "country_label", "response_received", "feasibility_score", "days_open", "reminder_count"]].copy()
    tracking["Status"] = tracking["response_received"].map({True: "Received", False: "Pending"})
    tracking.loc[(tracking["Status"] == "Pending") & (tracking["days_open"] > 7), "Status"] = "Overdue"
    tracking["Last Contact"] = tracking["days_open"].map(lambda d: f"{int(d)} days ago" if d else "Today")
    tracking = tracking[["site_name", "country_label", "Status", "feasibility_score", "Last Contact", "reminder_count"]]
    tracking.columns = ["Site Name", "Location", "Status", "Feasibility Score", "Last Contact", "Reminders"]
    st.markdown("<div class='surface-dark'><div class='section-head' style='color:#fff'>Site Response Tracking</div>", unsafe_allow_html=True)
    st.dataframe(tracking.head(25), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Feasibility Analysis":
    st.markdown("<div class='page-title'>Site Feasibility Analysis</div><div class='page-sub'>Drill into site-level detail with AI score explainability, PI context, and operational readiness signals.</div>", unsafe_allow_html=True)
    site_name = st.selectbox("Choose Site", VIEW["site_name"].tolist() if not VIEW.empty else MASTER["site_name"].tolist())
    row = MASTER[MASTER["site_name"] == site_name].iloc[0]
    st.markdown(f"<div class='surface-dark'><div style='display:flex;justify-content:space-between;align-items:center'><div><div style='font-size:22px;font-weight:800'>{row['site_name']}</div><div>{row['city']}, {row['country_label']}  •  PI: {row['matched_pi_name']}  •  Status: Feasibility {('Completed' if row['response_received'] else 'Pending')}</div></div><div style='font-size:46px;font-weight:800'>{int(row['ai_rank_score'])}<span style='font-size:18px'>/100</span></div></div></div>", unsafe_allow_html=True)
    a, b = st.columns([1.35, 1.0])
    with a:
        st.markdown("<div class='surface'><div class='section-head'>Site Selection Parameters</div>", unsafe_allow_html=True)
        score_df = pd.DataFrame({
            "Dimension": [
                "Investigator and Site Qualification", "Patient recruitment feasibility", "Patient recruitment and management",
                "Site infrastructure and technologies", "Lab and operational capability", "Regulatory / IRB readiness", "Budgetary consideration"
            ],
            "Score": [
                round((row['pi_years_experience'] or 0) / 2, 1), round((row['projected_enroll_rate_per_month'] or 0), 1),
                round((row['retention_rate'] or 0) * 10, 1), round(10 - min((row['data_entry_lag_days'] or 0), 10), 1),
                round(10 - min((row['screen_fail_rate'] or 0) * 20, 10), 1), round(8 + row['central_irb_preferred'] * 1.5, 1), round(8.0, 1)
            ]
        })
        st.dataframe(score_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='surface'><div class='section-head'>AI Score Breakdown</div>", unsafe_allow_html=True)
        g = go.Figure(go.Indicator(mode="gauge+number", value=float(row["feasibility_score"]), title={"text": "Feasibility Score"}, gauge={"axis": {"range": [0,100]}}))
        g.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(g, use_container_width=True)
        st.dataframe(ranking_explanation(row), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Qualification":
    st.markdown("<div class='page-title'>Site Qualification Review</div><div class='page-sub'>Review top-level site details, CDA state, CRA flags, and preferred or backup decisions.</div>", unsafe_allow_html=True)
    selected_pool = MASTER.head(10).copy()
    current = selected_pool.iloc[0]
    st.markdown(f"<div class='surface-dark'><div style='display:flex;justify-content:space-between;align-items:center;gap:12px'><div><div style='font-size:18px'>Score {int(current['qualification_score'])}</div><div style='font-size:20px;font-weight:800'>{current['site_name']}</div><div>{current['city']}, {current['country_label']}  •  {current['matched_pi_name']}  •  CDA {current['cda_status']}</div></div><div style='min-width:340px'></div></div></div>", unsafe_allow_html=True)
    q_df = MASTER[["site_id", "site_name", "matched_pi_name", "cda_status", "cra_flag", "qualification_score", "preferred"]].copy()
    q_df.columns = ["Site ID", "Facility Name", "Investigator", "CDA Status", "CRA Assigned", "Overall Score", "Preferred"]
    edited_q = st.data_editor(
        q_df,
        use_container_width=True,
        hide_index=True,
        disabled=["Site ID", "Facility Name", "Investigator", "Overall Score"],
        column_config={
            "Preferred": st.column_config.CheckboxColumn("Mark Preferred"),
            "CDA Status": st.column_config.SelectboxColumn("CDA Status", options=["Pending", "In Review", "Executed"]),
            "CRA Assigned": st.column_config.SelectboxColumn("CRA Flag", options=["None", "Risk", "IRB Review", "Feasibility Delay", "Documentation"]),
        },
        key="qualification_edit",
    )
    c1, c2 = st.columns([1.3, 0.9])
    with c1:
        if st.button("Save Qualification Updates", use_container_width=True):
            for _, r in edited_q.iterrows():
                persist_site_action(r["Site ID"], preferred=bool(r["Preferred"]), cda_status_override=r["CDA Status"], cra_flag_override=r["CRA Assigned"])
                append_audit("qualification_update", "site", r["Site ID"], f"CDA={r['CDA Status']}; CRA={r['CRA Assigned']}; preferred={r['Preferred']}")
            set_flash_message("Save Qualification Updates completed. Qualification changes were persisted.")
            clear_and_rerun()
    with c2:
        target_site = st.selectbox("CRA comment site", MASTER["site_name"].tolist())
        sid = MASTER.loc[MASTER["site_name"] == target_site, "site_id"].iloc[0]
        default_comment = MASTER.loc[MASTER["site_id"] == sid, "cra_comment"].fillna("").iloc[0]
        comment = st.text_area("CRA Comment", value=default_comment, height=110)
        if st.button("Save CRA Comment", use_container_width=True):
            persist_site_action(sid, cra_comment=comment)
            upsert_notification(sid, "Action Required: Qualification Review", "Medium", comment[:140] or "Qualification review updated")
            append_audit("cra_comment", "site", sid, comment[:140])
            set_flash_message("Save CRA Comment completed. Comment and notification were saved.")
            clear_and_rerun()

elif page == "Final Selection":
    st.markdown("<div class='page-title'>Final Selection</div><div class='page-sub'>Review and finalize the AI-recommended roster, capture justifications, and export stakeholder-ready outputs.</div>", unsafe_allow_html=True)
    metric_cards([
        ("Selected Sites", f"{int((MASTER['final_status'] == 'Selected').sum())}/{max(1, len(MASTER.head(35)))}", "dark"),
        ("Projected Enrollment", int(MASTER[MASTER['final_status'] != 'Rejected']['projected_enroll_rate_per_month'].fillna(0).sum()), "dark"),
        ("Est. Study Duration", "18 months", "dark"),
        ("Overall AI Confidence", f"{int(MASTER['ai_rank_score'].head(20).mean()) if not MASTER.empty else 94}%", "dark"),
    ])
    left, right = st.columns([1.65, 0.8])
    with left:
        st.markdown("<div class='surface-dark'><div class='section-head' style='color:#fff'>Selected & Backup Sites</div>", unsafe_allow_html=True)
        final = MASTER[["site_id", "site_name", "country_label", "qualification_score", "final_status", "selection_justification"]].copy()
        final.columns = ["Site ID", "Name", "Location", "Final Score", "Status", "AI Justification"]
        st.dataframe(final.head(20), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
        target = st.selectbox("Site for final decision", MASTER["site_name"].tolist())
        sid = MASTER.loc[MASTER["site_name"] == target, "site_id"].iloc[0]
        reason = st.text_area("Selection justification", value=MASTER.loc[MASTER["site_id"] == sid, "selection_justification"].fillna("").iloc[0] or "Strong protocol fit, favorable feasibility, and operational readiness.")
        status = st.selectbox("Final status override", ["", "Selected", "Backup", "Rejected"])
        if st.button("Approve Roster / Save Decision", use_container_width=True):
            persist_site_action(sid, selection_justification=reason, final_status_override=status)
            append_audit("final_decision", "site", sid, f"status={status or 'auto'}")
            set_flash_message("Approve Roster / Save Decision completed. Final site decision was saved.")
            clear_and_rerun()
    with right:
        st.markdown("<div class='surface'><div class='section-head'>Recent Activity</div>", unsafe_allow_html=True)
        audit_tail = load_or_init("audit_log.csv", ["timestamp", "action", "entity_type", "entity_id", "details"]).tail(8).iloc[::-1]
        if audit_tail.empty:
            st.caption("No activity yet.")
        else:
            for _, r in audit_tail.iterrows():
                st.markdown(f"**{r['action'].replace('_', ' ').title()}**  ")
                st.caption(f"{r['entity_id']} • {r['timestamp']}")
        st.markdown("</div>", unsafe_allow_html=True)
        export_df = MASTER[["site_id", "site_name", "country_label", "ai_rank_score", "feasibility_score", "qualification_score", "final_status", "selection_justification"]]
        st.download_button("Export Excel Matrix (CSV)", export_df.to_csv(index=False).encode("utf-8"), file_name="selected_sites_export.csv", mime="text/csv", use_container_width=True)
        st.download_button("Export Audit Log", load_or_init("audit_log.csv", ["timestamp", "action", "entity_type", "entity_id", "details"]).to_csv(index=False).encode("utf-8"), file_name="audit_log.csv", mime="text/csv", use_container_width=True)

elif page == "Chatbot Assistance":
    st.markdown("<div class='page-title'>Chatbot Assistance</div><div class='page-sub'>Use the assistant for navigation guidance, site queries, and score explanations based on the current dashboard state.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([0.65, 1.75])
    with c1:
        st.markdown("<div class='surface'><div class='section-head'>Recent Sessions</div>", unsafe_allow_html=True)
        for item in ["Mass General Feasibility", "European Oncology Sites", "SLA Breach Analysis", "Protocol ST-492 Setup"]:
            st.markdown(f"- {item}")
        if st.button("New Chat Session", use_container_width=True):
            reset_chat_history()
            st.success("New Chat Session started. Chat history has been reset.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        if "chat_history" not in st.session_state:
            reset_chat_history()
        st.markdown("<div class='surface-dark' style='min-height:520px'>", unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        prompt = st.chat_input("Ask about site feasibility, qualification status, or navigate the app...")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            answer_payload = chatbot_answer(prompt, page, VIEW)
            answer_text = normalize_text_value(answer_payload.get("response", "")).strip() or chatbot_answer_fallback(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": answer_text})
            with st.chat_message("assistant"):
                st.write(answer_text)
            append_chat_usage(
                username=normalize_text_value(st.session_state.get("current_user", "")),
                full_name=normalize_text_value(st.session_state.get("current_full_name", "")),
                role=normalize_text_value(st.session_state.get("current_role", "")),
                page_name=page,
                prompt=prompt,
                response=answer_text,
                used_local_llm=bool(answer_payload.get("used_local_llm", False)),
                success=bool(answer_payload.get("success", False)),
                error_message=normalize_text_value(answer_payload.get("error_message", "")),
            )
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "CRA Notifications":
    st.markdown("<div class='page-title'>Notification Center</div><div class='page-sub'>Manage feasibility submissions, pending actions, and acknowledgement status from the persisted notifications dataset.</div>", unsafe_allow_html=True)
    notes = load_or_init_notifications().merge(SITES[["site_id", "site_name"]], on="site_id", how="left")
    open_notes = notes[~notes["acknowledged"].fillna(False)] if not notes.empty else notes
    metric_cards([
        ("Total Notifications", len(notes), "dark"),
        ("Pending Actions", len(open_notes), "light"),
        ("Email Delivered", int((notes["type"].astype(str).str.contains("Survey|Warning|Action", na=False)).sum()) if not notes.empty else 0, "light"),
        ("Acknowledged", int(notes["acknowledged"].fillna(False).sum()) if not notes.empty else 0, "light"),
    ])
    if notes.empty:
        st.info("No notifications yet. Send surveys or save qualification decisions to generate alerts.")
    else:
        status_filter = st.radio("View", ["All", "Pending", "Acknowledged"], horizontal=True)
        show = notes.copy()
        if status_filter == "Pending":
            show = show[~show["acknowledged"].fillna(False)]
        elif status_filter == "Acknowledged":
            show = show[show["acknowledged"].fillna(False)]
        for _, note in show.sort_values("created_at", ascending=False).iterrows():
            with st.container(border=True):
                a, b, c = st.columns([2.3, 0.8, 0.9])
                a.markdown(f"**{note['type']}**")
                a.caption(f"{note.get('site_name', note['site_id'])} has an update. {note['message']}")
                b.markdown(f"<span class='site-chip {'chip-danger' if note['priority']=='High' else 'chip-warning' if note['priority']=='Medium' else 'chip-info'}'>{note['priority'].upper()}</span>", unsafe_allow_html=True)
                if bool(note.get("acknowledged", False)):
                    c.caption("Acknowledged")
                else:
                    if c.button("Acknowledge", key=f"ack_{note['notification_id']}"):
                        acknowledge_notification(note["notification_id"])
                        append_audit("notification_ack", "notification", note["notification_id"], note["type"])
                        set_flash_message("Acknowledge completed. Notification marked as acknowledged.")
                        clear_and_rerun()
                st.caption(str(note["created_at"]))

st.caption(f"Using local data folder: {DATA_DIR}")
