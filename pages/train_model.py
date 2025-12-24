# pages/5_🤖_Train_Model.py
import streamlit as st
import pandas as pd
import plotly.express as px
import time
import math
from modules.data_utils import read_dataset
from modules.model_utils import train_models

st.set_page_config(page_title="BiasScope | Train Model", layout="wide")
st.title("🤖 Train Model")

# Load dataset path from session
cleaned_path = st.session_state.get("cleaned_path", "") or st.session_state.get("dataset_path", "")
if not cleaned_path:
    st.warning("⚠️ Please upload and preprocess your dataset first.")
    st.stop()

df_full = read_dataset(cleaned_path)
st.markdown(f"**Loaded processed dataset:** `{cleaned_path.split('/')[-1]}` — Shape: {df_full.shape[0]} × {df_full.shape[1]}")

col1, col2 = st.columns([1, 2])
with col1:
    target_col = st.selectbox("🎯 Select Target Column", options=df_full.columns, index=len(df_full.columns)-1)

with col2:
    st.info("BiasScope will auto-detect the task and train candidate models.")
    mode = st.radio("Training Mode:", ["⚡ Quick Mode (fast)", "🔥 Full Mode (all models)"], index=0)
    tune_toggle = st.checkbox("Run hyperparameter tuning (SLOW)", value=False)
    if tune_toggle:
        st.warning("Tuning will significantly increase runtime.")

# Fast-test options
st.markdown("---")
st.subheader("⚙️ Quick test options")
use_sample = st.checkbox("Train on sample first (fast sanity check)", value=True)
sample_frac = 0.1
if use_sample:
    sample_frac = st.slider("Sample fraction (fraction of rows to use)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

# Cancel button (reset before each run)
if "cancel_train" not in st.session_state:
    st.session_state["cancel_train"] = False

# UI area for progress + control
st.markdown("---")
st.subheader("📈 Training Control & Progress")

col_a, col_b = st.columns([3, 1])
with col_b:
    if st.button("⏹️ Cancel training"):
        st.session_state["cancel_train"] = True
        st.warning("Cancel requested — training will stop after current model finishes.")

status_text = st.empty()
progress = st.progress(0)
timings_container = st.empty()
eta_text = st.empty()

# storage for timing info
model_timings = {}   # name -> seconds
timing_order = []    # ordered list of (name, seconds)

# progress callback will be passed into train_models
def _progress_cb(percent: int, message: str):
    """
    Called from train_models between model runs.
    We use message strings emitted by model_utils to track start/finish.
    If user requested cancel, raise an exception to abort the training loop.
    """
    # Check cancel flag and raise to abort
    if st.session_state.get("cancel_train", False):
        raise RuntimeError("CANCEL_REQUESTED")

    # Update progress UI
    try:
        progress.progress(min(100, max(0, int(percent))))
    except Exception:
        pass
    # Update status
    try:
        status_text.info(message)
    except Exception:
        pass

    # Message parsing: detect "Training ..." vs "Finished ..." patterns
    # We expect messages like: "Training {name} (i/N)..." and "Finished i/N: {name}"
    msg = str(message)
    if msg.lower().startswith("training"):
        # Extract model name between "Training " and " ("
        try:
            name = msg.split("Training", 1)[1].split("(")[0].strip()
        except Exception:
            name = msg
        # record start time
        model_timings[name] = time.time()
        if name not in timing_order:
            timing_order.append(name)
    elif msg.lower().startswith("finished") or msg.lower().startswith("finished!"):
        # For finished messages we expect "Finished i/N: name" or "Finished! Best Model -> name"
        # try to extract name from the message; if name exists in model_timings, compute elapsed.
        try:
            # attempt to find ': name' pattern
            if ":" in msg:
                name = msg.split(":", 1)[1].strip()
            elif "finished" in msg.lower():
                # fallback to last started model
                name = timing_order[-1] if timing_order else "unknown"
            else:
                name = msg
        except Exception:
            name = timing_order[-1] if timing_order else "unknown"

        if name in model_timings:
            elapsed = time.time() - model_timings[name]
            # overwrite with elapsed seconds (float)
            model_timings[name] = elapsed
            # also ensure timing_order contains name
            if name not in timing_order:
                timing_order.append(name)

    # update timings display & ETA
    try:
        # Build a small dataframe showing per-model times for finished ones
        rows = []
        finished_count = 0
        total_models_est = None
        # compute finished entries from model_timings that are numeric elapsed times
        for nm in timing_order:
            v = model_timings.get(nm)
            if isinstance(v, (int, float)) and v > 0 and v < 1e6:  # finished
                rows.append({"Model": nm, "Time (s)": round(v, 2)})
                finished_count += 1
        # estimate remaining time:
        total = len(timing_order) if timing_order else None
        if total is None or total == 0:
            eta_text.text("")
        else:
            avg = (sum([v for v in model_timings.values() if isinstance(v, (int, float)) and v > 0]) / max(1, finished_count)) if finished_count else None
            if avg:
                # estimate assuming total model count equals the number of models discovered so far or use percent
                # If percent provided, use it to estimate remaining:
                try:
                    pct = progress._value  # streamlit progress value (0..100)
                except Exception:
                    pct = None
                if pct and pct > 0:
                    remaining = max(0, 100 - pct) / 100.0 * (avg * max(1, total - finished_count))
                else:
                    remaining = avg * max(0, (total - finished_count))
                eta_text.info(f"Estimated remaining time: ~{int(remaining)}s (avg per finished model: {round(avg,1)}s)")
            else:
                eta_text.text("")
        if rows:
            df_t = pd.DataFrame(rows)
            timings_container.dataframe(df_t, use_container_width=True)
        else:
            timings_container.write("No finished model timings yet.")
    except Exception:
        pass


# Run training
if st.button("🚀 Start Training", use_container_width=True):
    # reset cancel flag
    st.session_state["cancel_train"] = False
    progress.progress(0)
    status_text.info("Starting...")

    # choose dataset (sample or full)
    if use_sample:
        df = df_full.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        st.info(f"Using sample: {df.shape[0]} rows ({int(sample_frac*100)}% of dataset) for fast test.")
    else:
        df = df_full.copy()

    mode_param = "quick" if "Quick Mode" in mode else "full"

    start_all = time.time()
    try:
        result = train_models(
            df,
            target_col,
            tune=tune_toggle,
            mode=mode_param,
            progress_callback=_progress_cb
        )
    except RuntimeError as e:
        # Cancelled or other runtime abort
        if "CANCEL_REQUESTED" in str(e) or "CANCEL" in str(e).upper():
            status_text.error("Training cancelled by user.")
            progress.progress(0)
            st.session_state["cancel_train"] = False
            result = {"error": "Training cancelled by user."}
        else:
            status_text.error(f"Training aborted: {e}")
            result = {"error": str(e)}
    except Exception as e:
        status_text.error(f"Training failed: {e}")
        result = {"error": str(e)}
    end_all = time.time()

    # Save result to session
    st.session_state["trained_model"] = result

    if result.get("error"):
        st.error(f"Result: {result.get('error')}")
    else:
        st.success(f"✅ Best Model: **{result.get('best_model')}** ({result.get('task')})")
        st.write(f"Model saved at: `{result.get('model_path')}`")

        # Final timing summary
        total_time = round(end_all - start_all, 2)
        st.info(f"Total training wall-time: {total_time}s")

        # Show results table
        st.subheader("📊 Model Performance Comparison")
        res_df = pd.DataFrame(result.get("results", []))
        st.dataframe(res_df, use_container_width=True)

        # Chart
        if result.get("task") == "classification":
            y_cols = []
            if "Accuracy" in res_df.columns: y_cols.append("Accuracy")
            if "F1 Score" in res_df.columns: y_cols.append("F1 Score")
            if y_cols:
                fig = px.bar(res_df, x="Model", y=y_cols, barmode="group", title="Classification Model Performance", text_auto=".2f")
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)
        else:
            if "R2 Score" in res_df.columns:
                fig = px.bar(res_df, x="Model", y=["R2 Score"], title="Regression Model Performance", text_auto=".2f")
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)

    # reset cancel flag at end
    st.session_state["cancel_train"] = False
