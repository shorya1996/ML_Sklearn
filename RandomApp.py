# app.py
import streamlit as st
import pandas as pd
import gc
from typing import Optional
from utils.data_preprocessing import preprocess_data
from utils.fake_data import generate_realistic_synthetic_dataset
from utils.rulefit_model import RuleFit
from utils.statistical_analysis import perform_statistical_analysis
from utils.logger import setup_logger

logger = setup_logger(__name__)

st.set_page_config(page_title="Rule Extractor", layout="wide")
st.title("✨ Rule Extractor Web App")

st.sidebar.header("📂 Data Options")
data_option = st.sidebar.radio("Choose an option:", ("Upload CSV", "Generate Fake Data to learn"))

# ---------- helpers ----------
def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric dtypes and convert low-cardinality objects to category to save memory."""
    df = df.copy()
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        try:
            if pd.api.types.is_integer_dtype(df[col].dropna()):
                df[col] = pd.to_numeric(df[col], downcast="signed")
            else:
                df[col] = pd.to_numeric(df[col], downcast="float")
        except Exception:
            # if conversion fails, leave as-is
            continue
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            if df[col].nunique(dropna=False) / max(1, len(df)) < 0.5:
                df[col] = df[col].astype("category")
        except Exception:
            continue
    return df

def clear_uploaded_dataframe():
    """Remove dataframe from session state, clear caches, and force GC."""
    if "data" in st.session_state:
        try:
            del st.session_state["data"]
            logger.info("Deleted st.session_state['data']")
        except Exception:
            st.session_state.pop("data", None)
    if "uploader_name" in st.session_state:
        st.session_state.pop("uploader_name", None)
    # clear any cached Streamlit data functions (if used)
    try:
        st.cache_data.clear()
    except Exception:
        try:
            st.legacy_caching.clear_cache()
        except Exception:
            pass
    gc.collect()
    logger.info("Cleared caches and ran gc.collect()")

def load_uploaded_csv(uploader) -> Optional[pd.DataFrame]:
    """Read uploaded CSV into memory, downcast, and return DataFrame. Returns None on failure."""
    try:
        # read in a memory-friendly way; pandas handles chunks internally with low_memory
        df = pd.read_csv(uploader)
        df = downcast_df(df)
        return df
    except Exception as e:
        logger.exception("Failed to read uploaded CSV: %s", e)
        st.error(f"Failed to read uploaded CSV: {e}")
        return None

# ---------- Data selection / loading ----------
# File uploader widget (Streamlit handles file persistence across reruns)
uploader = st.sidebar.file_uploader("Upload your CSV file", type=["csv"], key="uploader")

# If uploader is None but we previously had an uploaded file, clear stored data
if uploader is None and st.session_state.get("uploader_name") is not None:
    st.info("Uploaded file removed — clearing dataset from memory.")
    clear_uploaded_dataframe()

# If uploader provided, load it (but only if not already loaded)
if uploader is not None:
    # store name to detect future removal (uploader -> None)
    st.session_state["uploader_name"] = getattr(uploader, "name", "uploaded_file")
    if "data" not in st.session_state:
        with st.spinner("Reading CSV (downcasting to save memory)..."):
            df = load_uploaded_csv(uploader)
            if df is None:
                st.stop()
            st.session_state["data"] = df
            st.success(f"Loaded {len(df):,} rows, {len(df.columns):,} cols")

# Generate synthetic data option
if data_option == "Generate Fake Data to learn":
    rows = st.sidebar.slider("Number of rows", 10, 20000, 500)
    # if user generated fake data, store it in session_state['data'] (so subsequent actions can use it)
    if "data" not in st.session_state or st.session_state.get("_fake_rows") != rows:
        with st.spinner("Generating synthetic data..."):
            df_fake = generate_realistic_synthetic_dataset(num_transactions=rows)
            df_fake = downcast_df(df_fake)
            st.session_state["data"] = df_fake
            st.session_state["_fake_rows"] = rows
            st.success(f"Generated {rows:,} synthetic rows")

# If no data available, early stop
if "data" not in st.session_state:
    st.warning("No dataset loaded. Upload a CSV or generate fake data from the sidebar.")
    st.stop()

data = st.session_state["data"]

# ---------- UI: safe preview and actions ----------
st.markdown("### 🔍 Data Preview ")
# Always show only a small preview to avoid freezing the browser
preview_n = 100
try:
    st.dataframe(data.head(preview_n))
except Exception:
    # fallback minimal preview
    st.write(data.head(10))

# Show memory usage and provide a clear button
try:
    mem_mb = data.memory_usage(deep=True).sum() / (1024 ** 2)
    st.write(f"Approx. memory usage: **{mem_mb:.2f} MB** (dataframe in server memory)")
except Exception:
    st.write("Memory usage: unknown")



# ---------- Main actions ----------
option = st.selectbox("Select action:", ["Exploratory Data Analysis", "Generate Rules"])

if option == "Exploratory Data Analysis":
    drop_columns = st.multiselect("Columns to exclude:", data.columns.tolist())
    if st.button("Perform Analysis"):
        with st.spinner("Running statistical analysis..."):
            res = perform_statistical_analysis(data, drop_columns=drop_columns)
            st.markdown("#### Numerical Summary")
            st.json(res["Numerical Summary"])
            st.markdown("#### Categorical Summary")
            st.json(res["Categorical Summary"])
            st.markdown("#### Outliers (first 200 rows)")
            st.dataframe(res["Outliers Dataset"].head(200))

elif option == "Generate Rules":
    drop_columns_from_rules = st.multiselect("Drop columns (ids, tx ids):", data.columns.tolist())
    categorical_custom_columns = st.multiselect("Force categorical:", data.columns.tolist())
    target_col = st.selectbox("Select target column:", data.columns.tolist())
    sample_cap = st.number_input("Sample size cap for RF (interactive):", min_value=1000, max_value=500000, value=20000, step=1000)
    max_rules = st.number_input("Max rules to extract:", min_value=10, max_value=2000, value=200)

    if st.button("Generate Rules"):
        try:
            # Preprocess (do not drop uploaded df; preprocess_data returns X, y views)
            X, y = preprocess_data(data, target_col, categorical_custom_columns, drop_columns_from_rules, dropna_rows=False)
            rf = RuleFit(tree_size=4, max_rules=int(max_rules), min_support=0.01,
                         sample_size_cap=int(sample_cap), top_n_for_cats=30, n_estimators_cap=100, random_state=42)
            st.info("Fitting rule extractor (this may take a little while)...")
            with st.spinner("Fitting the model..."):
                rf.fit(X, y)
            rules = rf.get_rules()
            st.markdown("### 📜 Generated Rules")
            # show head only to avoid massive rendering
            st.dataframe(rules.head(500))
            csv = rules.to_csv(index=False)
            st.download_button("Download rules CSV", csv, file_name="rules.csv", mime="text/csv")
            # optionally show rule coverage counts without loading full df into UI:
            counts = rf.apply_rules_to_dataframe(data, chunk_size=100_000)
            st.markdown("Rule coverage (first 50 rules):")
            st.dataframe(pd.DataFrame(counts).head(50))
        except Exception as e:
            logger.exception("Error during rule generation: %s", e)
            st.error(f"Error during rule generation: {e}")
