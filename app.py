import streamlit as st
import pandas as pd
import json
import heapq
import os
import time
from io import BytesIO

from rank import score_candidate

# ---------------- CONFIG ---------------- #

st.set_page_config(
    page_title="AI Candidate Ranker",
    page_icon="🤖",
    layout="wide"
)

DATA_FILE = "candidates.jsonl"

st.title("🤖 Senior AI Engineer Candidate Ranker")

st.write("Ranks candidates directly from the local **candidates.jsonl** dataset.")

TOP_N = st.sidebar.slider(
    "Top Candidates",
    10,
    500,
    100,
    10
)

TOTAL_CANDIDATES = st.sidebar.number_input(
    "Expected Candidates",
    value=100000,
    step=1000
)

# ---------------- CACHE ---------------- #

@st.cache_data(show_spinner=False)
def dataframe_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# ---------------- MAIN ---------------- #

if not os.path.exists(DATA_FILE):
    st.error(f"{DATA_FILE} not found.")
    st.stop()

if st.button("🚀 Run Ranking", use_container_width=True):

    start = time.time()

    progress = st.progress(0)

    status = st.empty()

    heap = []

    processed = 0

    with open(DATA_FILE, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            candidate = json.loads(line)

            score, reasoning = score_candidate(candidate)

            item = (
                score,
                candidate["candidate_id"],
                reasoning
            )

            if len(heap) < TOP_N:
                heapq.heappush(heap, item)
            else:
                if score > heap[0][0]:
                    heapq.heapreplace(heap, item)

            processed += 1

            if processed % 500 == 0:

                elapsed = time.time() - start

                cps = processed / elapsed if elapsed else 0

                eta = (
                    (TOTAL_CANDIDATES - processed) / cps
                    if cps > 0 else 0
                )

                progress.progress(
                    min(processed / TOTAL_CANDIDATES, 1.0)
                )

                status.info(
                    f"""
Processed: **{processed:,}**

Speed: **{cps:.0f} candidates/sec**

ETA: **{eta:.1f} sec**
"""
                )

    progress.progress(1.0)

    top = sorted(heap, key=lambda x: x[0], reverse=True)

    rows = []

    for rank, (score, cid, reasoning) in enumerate(top, start=1):

        rows.append({
            "Rank": rank,
            "Candidate ID": cid,
            "Score": round(score, 4),
            "Reasoning": reasoning
        })

    df = pd.DataFrame(rows)

    runtime = time.time() - start

    st.success("✅ Ranking Completed")

    c1, c2, c3 = st.columns(3)

    c1.metric("Candidates Processed", f"{processed:,}")

    c2.metric("Top Candidates", len(df))

    c3.metric("Runtime", f"{runtime:.2f} sec")

    st.subheader("🏆 Top Ranked Candidates")

    st.dataframe(
        df,
        use_container_width=True,
        height=600
    )

    st.subheader("📊 Score Distribution")

    st.bar_chart(df["Score"])

    csv = dataframe_to_csv(df)

    with open("submission.csv", "wb") as f:
        f.write(csv)

    st.download_button(
        "⬇ Download submission.csv",
        csv,
        file_name="submission.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.success("submission.csv saved locally.")

    st.subheader("🔍 Search Candidate")

    search = st.text_input("Candidate ID")

    if search:

        result = df[
            df["Candidate ID"].astype(str) == search
        ]

        if len(result):

            st.dataframe(
                result,
                use_container_width=True
            )

        else:

            st.warning("Candidate not found in Top Rankings.")