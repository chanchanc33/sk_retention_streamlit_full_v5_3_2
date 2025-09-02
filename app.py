
# -*- coding: utf-8 -*-
"""
SK-Branded HR Retention Dashboard (Mapping-Stable Edition)
- Robust CSV/XLSX loader (multi-encoding, multi-separator, Excel fallback)
- Column mapping with live previews, save/load mapping JSON
- Normalization to canonical columns -> all filters/metrics/charts use the normalized frame
- Priority list fixed to mapped columns; safer typing; label normalization
"""

import os, io, json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from PIL import Image, ImageDraw, ImageFont, ImageCms

def _placeholder_logo_img():
    w, h = 420, 160
    img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)
    red = (226, 35, 26, 255)      # #E2231A
    orange = (240, 90, 34, 255)   # #F05A22
    d.polygon([(0, h), (120, 0), (220, 0), (90, h)], fill=red)
    d.polygon([(140, h), (260, 0), (360, 0), (240, h)], fill=orange)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 90)
    except Exception:
        font = ImageFont.load_default()
    text = "SK"
    tw, th = d.textsize(text, font=font)
    d.text(((w - tw) // 2, (h - th) // 2), text, fill=(255, 255, 255, 255), font=font)
    # composite to white background to remove alpha
    bg = Image.new("RGBA", img.size, (255,255,255,255))
    return Image.alpha_composite(bg, img).convert("RGB")

def load_logo_image():
    # Return PIL.Image ready for st.image without relying on file paths.
    try:
        if os.path.exists(LOGO_PATH):
            im = Image.open(LOGO_PATH)
        else:
            im = _placeholder_logo_img()
        # Normalize color space / alpha
        if im.mode in ("LA","RGBA","P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255,255,255,255))
            im = Image.alpha_composite(bg, im).convert("RGB")
        elif im.mode not in ("RGB","L"):
            im = im.convert("RGB")
        # ICC -> sRGB if embedded
        try:
            icc = im.info.get("icc_profile")
            if icc:
                src_prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                dst_prof = ImageCms.createProfile("sRGB")
                im = ImageCms.profileToProfile(im, src_prof, dst_prof, outputMode="RGB")
        except Exception:
            pass
        return im
    except Exception:
        try:
            return _placeholder_logo_img()
        except Exception:
            return None

# ---------------- Brand & Logo ----------------
DEFAULT_THEME = {"name":"SK Group","primary":"#E2231A","secondary":"#F05A22","accent":"#FFB000"}
ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "sk_group_logo.png")


# ---------------- Synthetic Contact (Demo) ----------------
import hashlib
def _synth_contact(name:str, org:str):
    key = f"{name}|{org}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    num1 = int(h[:4], 16) % 9000 + 1000
    num2 = int(h[4:8], 16) % 9000 + 1000
    phone = f"010-{num1:04d}-{num2:04d}"
    # Use reserved test domain to avoid real mails
    email = f"user{h[:8]}@example.com"
    return phone, email

st.set_page_config(page_title="SK Retention Dashboard", layout="wide", page_icon="🔥")

def ensure_logo():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    if os.path.exists(LOGO_PATH):
        return
    try:
        from PIL import Image, ImageDraw, ImageFont
        w, h = 420, 160
        img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
        d = ImageDraw.Draw(img)
        red = (226, 35, 26, 255)      # #E2231A
        orange = (240, 90, 34, 255)   # #F05A22
        d.polygon([(0, h), (120, 0), (220, 0), (90, h)], fill=red)
        d.polygon([(140, h), (260, 0), (360, 0), (240, h)], fill=orange)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 90)
        except:
            font = ImageFont.load_default()
        text = "SK"
        tw, th = d.textsize(text, font=font)
        d.text(((w - tw) // 2, (h - th) // 2), text, fill=(255, 255, 255, 255), font=font)
        img.save(LOGO_PATH)
    except Exception:
        pass

def load_logo():
    # Return a path to an sRGB, non-transparent PNG for consistent SK colors.
    import io as _io
    import os
    from pathlib import Path as _Path
    try:
        from PIL import Image, ImageCms
    except Exception:
        ensure_logo()
        return LOGO_PATH
    ensure_logo()
    src_path = LOGO_PATH
    try:
        im = Image.open(src_path)
        # Convert to RGB with white background if needed
        if im.mode in ("LA","RGBA","P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255,255,255,255))
            im = Image.alpha_composite(bg, im).convert("RGB")
        elif im.mode not in ("RGB","L"):
            im = im.convert("RGB")
        # ICC -> sRGB
        try:
            icc = im.info.get("icc_profile")
            if icc:
                src_prof = ImageCms.ImageCmsProfile(_io.BytesIO(icc))
                dst_prof = ImageCms.createProfile("sRGB")
                im = ImageCms.profileToProfile(im, src_prof, dst_prof, outputMode="RGB")
        except Exception:
            pass
        fixed_path = os.path.join(ASSETS_DIR, "sk_group_logo_srgb.png")
        im.save(fixed_path, format="PNG")
        return fixed_path if _Path(fixed_path).exists() else src_path
    except Exception:
        return src_path



ensure_logo()

# ---------------- Helpers ----------------
def try_read_table_from_bytes(data: bytes, name: str):
    last_err = ""
    encs = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]
    seps = [None, ",", ";", "\t", "|"]
    for enc in encs:
        for sep in seps:
            try:
                bio = io.BytesIO(data)
                df = pd.read_csv(bio, encoding=enc, sep=sep, engine="python")
                return df, f"CSV loaded (encoding={enc}, sep={'auto' if sep is None else sep})"
            except Exception as e:
                last_err = f"[csv enc={enc} sep={sep}] {e}"
    try:
        bio = io.BytesIO(data)
        df = pd.read_excel(bio)
        return df, "Excel loaded (.xlsx/.xls)"
    except Exception as e:
        last_err = f"[excel] {e}"
    return None, last_err

@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    debug = {}
    if uploaded_file is not None:
        raw = uploaded_file.read()
        df, msg = try_read_table_from_bytes(raw, uploaded_file.name)
        debug["source"] = f"uploaded: {uploaded_file.name}"
        debug["load_msg"] = msg
        if df is None:
            st.error("❌ 업로드 파일을 읽을 수 없습니다. CSV(UTF-8 권장) 또는 XLSX로 업로드 해주세요.")
            st.caption(f"마지막 에러: {msg}")
            st.stop()
        return df, debug
    if os.path.exists("hr_analysis_results.csv"):
        with open("hr_analysis_results.csv", "rb") as f:
            raw = f.read()
        df, msg = try_read_table_from_bytes(raw, "hr_analysis_results.csv")
        debug["source"] = "local: hr_analysis_results.csv"
        debug["load_msg"] = msg
        if df is None:
            st.error("로컬 hr_analysis_results.csv 를 읽을 수 없습니다. CSV 또는 XLSX 업로드를 시도하세요.")
            st.caption(f"마지막 에러: {msg}")
            st.stop()
        return df, debug
    st.error("파일이 없습니다. CSV 또는 XLSX 파일을 업로드 해주세요.")
    st.stop()

def find_column(cols, cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    for c in cands:
        for k,raw in low.items():
            if c.lower() in k: return raw
    return None

def auto_guess_map(columns):
    cols = list(columns)
    return {
        "EMP_NAME":   find_column(cols, ["성명","이름","name"]) or "없음",
        "ORG":        find_column(cols, ["본부","조직","org","division"]) or "없음",
        "TEAM":       find_column(cols, ["팀","부서","team","dept"]) or "없음",
        "GRADE":      find_column(cols, ["성과등급","grade"]) or "없음",
        "LEVEL":      find_column(cols, ["직급레벨","레벨","직급","level"]) or "없음",
        "AGE":        find_column(cols, ["나이","age"]) or "없음",
        "TENURE":     find_column(cols, ["근속연수(년)","근속연수","tenure","years at company"]) or "없음",
        "SALARY":     find_column(cols, ["연봉(원)","연봉","salary","annual salary"]) or "없음",
        "TALENT":     find_column(cols, ["인재등급","talent"]) or "없음",
        "RISK":       find_column(cols, ["퇴직위험도","위험도","risk","leave risk"]) or "없음",
        "RISK_PROB":  find_column(cols, ["퇴직위험예측확률","예측확률","prob"]) or "없음",
        "RISK_REASON":find_column(cols, ["위험요인","reason","factor"]) or "없음",
        "PHONE":      find_column(cols, ["휴대폰","전화","phone"]) or "없음",
        "EMAIL":      find_column(cols, ["이메일","email"]) or "없음",
    }

def to_num_series(s):
    return pd.to_numeric(s.astype(str).str.replace(",","").str.strip(), errors="coerce")

def normalize_labels(dfN):
    # TALENT normalization
    if "TALENT" in dfN.columns:
        m = {
            "critical":"Critical", "high":"High", "standard":"Standard", "development":"Development",
            "핵심":"Critical","핵심인재":"Critical","상":"High","중":"Standard","개발":"Development"
        }
        dfN["TALENT"] = dfN["TALENT"].astype(str).str.strip()
        dfN["TALENT"] = dfN["TALENT"].replace({k:v for k,v in m.items()})
        dfN["TALENT"] = dfN["TALENT"].str.title()
    # GRADE to upper S/A/B/C/D if looks like a grade
    if "GRADE" in dfN.columns:
        dfN["GRADE"] = dfN["GRADE"].astype(str).str.strip().str.upper()

def build_normalized(df, colmap):
    N = pd.DataFrame(index=df.index)
    for std, src in colmap.items():
        if src and src != "없음" and src in df.columns:
            N[std] = df[src]
    # numerics
    for k in ["LEVEL","AGE","TENURE","SALARY","RISK","RISK_PROB"]:
        if k in N.columns:
            N[k] = to_num_series(N[k])
    # clip risk
    if "RISK" in N.columns:
        N["RISK"] = N["RISK"].clip(lower=0, upper=100)
    # normalize labels
    normalize_labels(N)
    # ensure strings for key categoricals
    for k in ["EMP_NAME","ORG","TEAM","TALENT","GRADE","RISK_REASON","PHONE","EMAIL"]:
        if k in N.columns:
            N[k] = N[k].astype(str)
    return N

def fmt_int(n): return "-" if n is None else f"{int(round(n)):,}"
def fmt_float(n,d=2): return "-" if n is None else f"{float(n):.{d}f}"

# ---------------- Sidebar: Upload & Mapping Save/Load ----------------
with st.sidebar:
    st.markdown("### 📤 데이터 업로드")
    upl = st.file_uploader("CSV 또는 Excel(xlsx)", type=["csv","xlsx","xls"])
    st.markdown("---")
    st.markdown("### 💾 매핑 저장/불러오기")
    if "colmap_json" not in st.session_state:
        st.session_state["colmap_json"] = None
    load_map = st.file_uploader("매핑 JSON 불러오기", type=["json"], key="map_loader")
    if load_map is not None:
        try:
            st.session_state["colmap_json"] = json.loads(load_map.read().decode("utf-8"))
            st.success("매핑 JSON을 불러왔습니다.")
        except Exception as e:
            st.error(f"매핑 JSON 로드 실패: {e}")
    st.caption("이 버전은 SK Group 고정 테마입니다.")

# ---------------- Load Data ----------------
df, dbg = load_data(upl)
df.columns = [c.strip() for c in df.columns]

# ---------------- Header ----------------
left, right = st.columns([1,5])
with left:
    logo_img = load_logo_image()
    st.image(logo_img, width=120) if logo_img is not None else st.write('SK')
with right:
    st.markdown(f"""
    <div style="padding:12px 16px;border-radius:16px;background:linear-gradient(90deg,{DEFAULT_THEME['primary']}22,{DEFAULT_THEME['secondary']}22);border:1px solid {DEFAULT_THEME['primary']}33">
      <div style="font-weight:800;color:{DEFAULT_THEME['primary']};font-size:24px;line-height:1">실행형 핵심인재 리텐션 대시보드</div>
      <div style="color:#555">CSV/XLSX 업로드 → 컬럼 매핑 → 필터/차트 → 리텐션 패키지</div>
    </div>
    """, unsafe_allow_html=True)

with st.expander("🔧 로딩 정보 / 디버그", expanded=False):
    st.write(dbg)
    st.write("데이터 shape:", df.shape)
    st.dataframe(df.head(8), use_container_width=True)

# ---------------- Column Mapping UI with Previews ----------------

## ---------------- Column Mapping UI with Previews ----------------
st.subheader("1) 컬럼 매핑 (필수)")

def _col_preview(col):
    if col == "없음" or col not in df.columns:
        return "—"
    vals = df[col].dropna().astype(str).unique()[:5]
    return " | ".join(map(str, vals)) if len(vals)>0 else "빈 컬럼"

# -- Column profiling for category-filtered options --
def _is_email_series(s):
    try:
        return (s.astype(str).str.contains("@")).mean() >= 0.4
    except Exception:
        return False

def _is_phone_series(s):
    try:
        return (s.astype(str).str.replace(r'[^0-9]','', regex=True).str.len()>=9).mean() >= 0.4
    except Exception:
        return False

def _numeric_ratio(s):
    try:
        t = pd.to_numeric(s.astype(str).str.replace(",","").str.strip(), errors="coerce")
        return t.notna().mean(), (t.min(), t.max())
    except Exception:
        return 0.0, (None, None)

def _profile_columns(df):
    info = {}
    for c in df.columns:
        sr = df[c]
        nr, rng = _numeric_ratio(sr)
        is_email = _is_email_series(sr)
        is_phone = _is_phone_series(sr)
        uniq = sr.astype(str).nunique()
        info[c] = {"num_ratio":nr, "min":rng[0], "max":rng[1], "is_email":is_email, "is_phone":is_phone, "uniq":uniq}
    return info

prof = _profile_columns(df)

def _cands_for(key):
    # Return candidate list for a given mapping key, filtered by category heuristics.
    cols = list(df.columns)
    # helper: prioritize by name hints
    def _prioritize(names, pool):
        prio = []
        rest = []
        ln_pool = {x.lower(): x for x in pool}
        for n in names:
            for k,raw in ln_pool.items():
                if n.lower() in k and raw not in prio:
                    prio.append(raw)
        for x in pool:
            if x not in prio:
                rest.append(x)
        return prio + rest

    if key in ["EMP_NAME"]:
        pool = [c for c in cols if prof[c]["num_ratio"] < 0.3 and not prof[c]["is_email"] and not prof[c]["is_phone"]]
        pool = _prioritize(["성명","이름","name"], pool)
        return pool
    if key in ["ORG","TEAM"]:
        pool = [c for c in cols if prof[c]["num_ratio"] < 0.3 and not prof[c]["is_email"] and not prof[c]["is_phone"]]
        # prefer moderate-cardinality categoricals
        pool = sorted(pool, key=lambda c: (abs(prof[c]["uniq"]), c))
        pool = _prioritize(["본부","조직","org","division","팀","부서","team","dept"], pool)
        return pool
    if key == "GRADE":
        pool = [c for c in cols if prof[c]["num_ratio"] < 0.3]
        pool = _prioritize(["성과등급","grade"], pool)
        return pool
    if key == "TALENT":
        pool = [c for c in cols if prof[c]["num_ratio"] < 0.3]
        pool = _prioritize(["인재등급","talent"], pool)
        return pool
    if key == "LEVEL":
        pool = [c for c in cols if prof[c]["num_ratio"] >= 0.7]
        pool = [c for c in pool if (prof[c]["min"] is not None and prof[c]["max"] is not None and 0 <= prof[c]["min"] <= 20 and 0 < prof[c]["max"] <= 20)]
        pool = _prioritize(["직급","레벨","level"], pool)
        return pool
    if key == "AGE":
        pool = [c for c in cols if prof[c]["num_ratio"] >= 0.7]
        pool = [c for c in pool if (prof[c]["min"] is not None and prof[c]["max"] is not None and 14 <= prof[c]["min"] <= 90 and 14 < prof[c]["max"] <= 100)]
        pool = _prioritize(["나이","age"], pool)
        return pool
    if key == "TENURE":
        pool = [c for c in cols if prof[c]["num_ratio"] >= 0.7]
        pool = [c for c in pool if (prof[c]["min"] is not None and prof[c]["max"] is not None and 0 <= prof[c]["min"] <= 50 and 0 < prof[c]["max"] <= 60)]
        pool = _prioritize(["근속","tenure","years"], pool)
        return pool
    if key == "SALARY":
        pool = [c for c in cols if prof[c]["num_ratio"] >= 0.7]
        pool = [c for c in pool if (prof[c]["min"] is not None and prof[c]["max"] is not None and max(prof[c]["min"],0) >= 10000 or max(prof[c]["max"] or 0,0) >= 1000000)]
        pool = _prioritize(["연봉","salary","보상"], pool)
        return pool
    if key == "RISK":
        pool = [c for c in cols if prof[c]["num_ratio"] >= 0.7]
        pool = [c for c in pool if (prof[c]["min"] is not None and prof[c]["max"] is not None and 0 <= prof[c]["min"] <= 100 and 0 <= prof[c]["max"] <= 100)]
        pool = _prioritize(["퇴직위험","위험도","risk"], pool)
        return pool
    if key == "RISK_PROB":
        pool = [c for c in cols if prof[c]["num_ratio"] >= 0.7]
        pool = [c for c in pool if (prof[c]["min"] is not None and prof[c]["max"] is not None and (0 <= prof[c]["min"] <= 1 and 0 <= prof[c]["max"] <= 1 or 0 <= prof[c]["min"] <= 100 and 0 <= prof[c]["max"] <= 100))]
        pool = _prioritize(["확률","prob"], pool)
        return pool
    if key == "RISK_REASON":
        pool = [c for c in cols if prof[c]["num_ratio"] < 0.3]
        pool = _prioritize(["위험요인","reason","factor"], pool)
        return pool
    if key == "PHONE":
        pool = [c for c in cols if prof[c]["is_phone"]]
        pool = _prioritize(["휴대폰","전화","phone"], pool)
        return pool
    if key == "EMAIL":
        pool = [c for c in cols if prof[c]["is_email"]]
        pool = _prioritize(["이메일","email"], pool)
        return pool
    return cols

if "colmap" not in st.session_state:
    st.session_state["colmap"] = auto_guess_map(df.columns)

# Toggle to allow full list if needed
show_all = st.toggle("전체 컬럼 보기 (권장: 해제)", value=False)

# compute candidate options per key, filtered; and avoid duplicates
keys_order = ["EMP_NAME","ORG","TEAM","TALENT","GRADE","LEVEL","AGE","TENURE","SALARY","RISK","RISK_PROB","RISK_REASON","PHONE","EMAIL"]
current = st.session_state["colmap"]
chosen = set()

def _options_for(key):
    base = ["없음"]
    cands = df.columns.tolist() if show_all else _cands_for(key)
    # remove already chosen (except itself)
    remaining = [c for c in cands if c not in chosen or c == current.get(key)]
    return base + remaining

columns_all = df.columns.tolist()

c1, c2, c3 = st.columns(3)
layout = [
    ("EMP_NAME","성명", c1), ("ORG","본부/조직", c1), ("TEAM","팀", c1),
    ("TALENT","인재등급", c2), ("GRADE","성과등급", c2), ("LEVEL","직급레벨", c2),
    ("RISK","퇴직위험도 (필수)", c3), ("SALARY","연봉(원)", c3), ("RISK_REASON","위험요인", c3),
    ("AGE","나이", c1), ("TENURE","근속연수(년)", c1),
    ("PHONE","휴대폰", c2), ("EMAIL","이메일", c2),
    ("RISK_PROB","퇴직위험예측확률", c3),
]

select_values = {}
for key,label,col in layout:
    with col:
        opts = _options_for(key)
        # pick the index for current value if present in options
        cur = current.get(key, "없음")
        if cur not in opts:
            cur = "없음"
        sel = st.selectbox(label, opts, index=opts.index(cur), help=_col_preview(cur), key=f"map_{key}")
        select_values[key] = sel
        # preview line
        st.caption("미리보기: " + _col_preview(sel))
        if sel != "없음":
            chosen.add(sel)

# apply selected values back to session
st.session_state["colmap"] = select_values

# 필수 확인
missing = [k for k in ["EMP_NAME","ORG","RISK"] if st.session_state["colmap"][k] == "없음" or st.session_state["colmap"][k] not in df.columns]
if missing:
    st.error("다음 필수 매핑이 누락되었습니다: " + ", ".join(missing))
    st.stop()

# Save mapping JSON
map_json = json.dumps(st.session_state["colmap"], ensure_ascii=False, indent=2)
st.download_button("💾 현재 매핑 JSON 다운로드", data=map_json, file_name="mapping.json", mime="application/json")


# ---------------- Build Normalized Frame ----------------
colmap = st.session_state.get("colmap", auto_guess_map(df.columns))
if not isinstance(colmap, dict):
    st.error("내부 매핑 상태가 손상되었습니다. 새로고침 후 다시 시도하세요.")
    st.stop()
N = build_normalized(df, colmap)

# Ensure demo contact exists even if not provided
if "EMP_NAME" in N and "ORG" in N:
    import numpy as _np
    if "PHONE" not in N.columns:
        N["PHONE"] = N.apply(lambda r: _synth_contact(str(r.get("EMP_NAME","")), str(r.get("ORG","")))[0], axis=1)
    else:
        N["PHONE"] = N.apply(lambda r: r["PHONE"] if isinstance(r["PHONE"], str) and r["PHONE"].strip() else _synth_contact(str(r.get("EMP_NAME","")), str(r.get("ORG","")))[0], axis=1)
    if "EMAIL" not in N.columns:
        N["EMAIL"] = N.apply(lambda r: _synth_contact(str(r.get("EMP_NAME","")), str(r.get("ORG","")))[1], axis=1)
    else:
        N["EMAIL"] = N.apply(lambda r: r["EMAIL"] if isinstance(r["EMAIL"], str) and r["EMAIL"].strip() else _synth_contact(str(r.get("EMP_NAME","")), str(r.get("ORG","")))[1], axis=1)


# Quick sanity snapshot
with st.expander("🧪 매핑 적용 데이터(상위 8행)", expanded=False):
    st.dataframe(N.head(8), use_container_width=True)

# ---------------- Filters ----------------
st.subheader("2) 필터")
c1,c2,c3,c4,c5 = st.columns([2,2,2,2,3])
with c1: search = st.text_input("🔎 검색(성명/본부/팀/인재등급/위험요인)","")
with c2: sel_org = st.multiselect("본부", sorted(N["ORG"].dropna().astype(str).unique().tolist()) if "ORG" in N else [])
with c3: sel_team = st.multiselect("팀", sorted(N["TEAM"].dropna().astype(str).unique().tolist()) if "TEAM" in N else [])
with c4: sel_grade = st.multiselect("성과등급", sorted(N["GRADE"].dropna().astype(str).unique().tolist()) if "GRADE" in N else [])
with c5: sel_reason = st.multiselect("위험요인", sorted(N["RISK_REASON"].dropna().astype(str).unique().tolist()) if "RISK_REASON" in N else [])

cc1,cc2,cc3,cc4 = st.columns(4)
with cc1: sel_level = st.multiselect("직급레벨", sorted(N["LEVEL"].dropna().astype(int).astype(str).unique().tolist()) if "LEVEL" in N else [])
with cc2: sel_talent = st.multiselect("인재등급", sorted(N["TALENT"].dropna().astype(str).unique().tolist()) if "TALENT" in N else [])
with cc3:
    risk_threshold = st.slider("리스크 임계치", 0, 100, 30, 1)
    only_key = st.checkbox("핵심인재만 (Critical/High)", value=True)
with cc4:
    if "AGE" in N:
        age_min, age_max = st.slider("나이 범위", 18, 70, (18, 70))
    else:
        age_min, age_max = None, None
    if "TENURE" in N:
        tenure_min, tenure_max = st.slider("근속연수(년) 범위", 0, 40, (0, 40))
    else:
        tenure_min, tenure_max = None, None

def passed_row(row):
    if search:
        s = search.lower()
        fields = [k for k in ["EMP_NAME","ORG","TEAM","TALENT","RISK_REASON"] if k in N.columns]
        if not any(s in str(row.get(f,"")).lower() for f in fields):
            return False
    if sel_org and row.get("ORG") not in sel_org: return False
    if sel_team and row.get("TEAM") not in sel_team: return False
    if sel_grade and row.get("GRADE") not in sel_grade: return False
    if sel_level and str(row.get("LEVEL")) not in sel_level: return False
    if sel_talent and row.get("TALENT") not in sel_talent: return False
    if sel_reason and row.get("RISK_REASON") not in sel_reason: return False
    if pd.isna(row.get("RISK")) or float(row.get("RISK")) < risk_threshold: return False
    if only_key and "TALENT" in N.columns and row.get("TALENT") not in ["Critical","High"]: return False
    if age_min is not None and "AGE" in N.columns:
        av = row.get("AGE")
        if av is None or not (age_min <= av <= age_max): return False
    if tenure_min is not None and "TENURE" in N.columns:
        tv = row.get("TENURE")
        if tv is None or not (tenure_min <= tv <= tenure_max): return False
    return True

F = N[N.apply(passed_row, axis=1)].copy()
st.caption(f"필터 결과: **{len(F):,}명** / 원본 {len(N):,}명")

# ---------------- KPI ----------------
k1,k2,k3,k4 = st.columns(4)
avg = lambda s: float(s.dropna().mean()) if s.dropna().size>0 else None
with k1: st.metric("대상 인원", f"{len(F):,}")
with k2: st.metric("평균 위험도", "-" if "RISK" not in F else f"{avg(F['RISK']):.1f}" if F['RISK'].dropna().size>0 else "-")
with k3: st.metric("평균 근속(년)", "-" if "TENURE" not in F else f"{avg(F['TENURE']):.2f}" if F['TENURE'].dropna().size>0 else "-")
with k4: st.metric("평균 연봉(원)", "-" if "SALARY" not in F else f"{int(round(avg(F['SALARY']))) if F['SALARY'].dropna().size>0 else '-':,}" if F['SALARY'].dropna().size>0 else "-")

st.divider()

# ---------------- Charts ----------------
c_g1,c_g2,c_g3 = st.columns(3)
with c_g1:
    st.subheader("성과등급 분포")
    if "GRADE" in F and not F["GRADE"].dropna().empty:
        g = F["GRADE"].value_counts().reset_index()
        g.columns = ["성과등급","인원"]
        st.plotly_chart(px.bar(g, x="성과등급", y="인원", color_discrete_sequence=[DEFAULT_THEME["secondary"]]), use_container_width=True)
    else:
        st.info("데이터 없음 또는 성과등급 미매핑")

with c_g2:
    st.subheader("직급레벨 분포")
    if "LEVEL" in F and not F["LEVEL"].dropna().empty:
        lv = F["LEVEL"].dropna().astype(int).astype(str)
        gv = lv.value_counts().sort_index().reset_index()
        gv.columns = ["직급레벨","인원"]
        st.plotly_chart(px.bar(gv, x="직급레벨", y="인원", color_discrete_sequence=[DEFAULT_THEME["accent"]]), use_container_width=True)
    else:
        st.info("데이터 없음 또는 직급레벨 미매핑")

with c_g3:
    st.subheader("퇴직위험도 구간")
    if "RISK" in F and not F["RISK"].dropna().empty:
        rb = pd.cut(F["RISK"], [0,30,50,70,100], right=False, labels=["0-29","30-49","50-69","70+"])
        bc = rb.value_counts().reindex(["0-29","30-49","50-69","70+"]).fillna(0).reset_index()
        bc.columns = ["구간","인원"]
        fig = px.pie(bc, values="인원", names="구간", color="구간",
                     color_discrete_map={"0-29":"#10b981","30-49":DEFAULT_THEME["accent"],"50-69":DEFAULT_THEME["secondary"],"70+":DEFAULT_THEME["primary"]})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("데이터 없음 또는 위험도 미매핑")

st.divider()

# ---------------- Priority list (Fixed) ----------------
st.subheader("우선순위 대응 리스트")
rank_map = {"Critical":4,"High":3,"Standard":2,"Development":1}
if "TALENT" in F:
    F["_rankTalent"] = F["TALENT"].map(rank_map).fillna(0)
else:
    F["_rankTalent"] = 0

sort_cols = ["RISK","_rankTalent"]
asc = [False, False]
if "SALARY" in F:
    sort_cols.append("SALARY"); asc.append(False)

SF = F.sort_values(sort_cols, ascending=asc)
cols_show = [c for c in ["EMP_NAME","ORG","TALENT","LEVEL","SALARY","RISK","RISK_REASON"] if c in SF.columns]
rename_kor = {"EMP_NAME":"성명","ORG":"본부","TALENT":"인재등급","LEVEL":"직급레벨","SALARY":"연봉(원)","RISK":"퇴직위험도","RISK_REASON":"위험요인"}
st.dataframe(SF[cols_show].rename(columns=rename_kor).head(200), use_container_width=True, height=360)

# ---------------- Per-employee package ----------------
st.subheader("직원별 리텐션 패키지 생성")
names = SF["EMP_NAME"].astype(str).unique().tolist() if "EMP_NAME" in SF else []
sel = st.selectbox("직원 선택", ["선택하세요"] + names)

def calc_roi(s, kr=0.3, kt=0.2, kp=0.5):
    s = float(s) if s is not None and pd.notna(s) else 0.0
    return {"total": s*(kr+kt+kp), "recruit": s*kr, "training": s*kt, "lost": s*kp}

def gen_pkg(risk, salary):
    level = "critical" if (risk or 0) >= 70 else "high" if (risk or 0) >= 50 else "medium" if (risk or 0) >= 30 else "low"
    budgets = {"critical":0.25,"high":0.15,"medium":0.08,"low":0.04}
    timelines = {"critical":"48시간 내","high":"1주일 내","medium":"2주 내","low":"1개월 내"}
    base = [
        {"action":"팀장 정기 1:1 설정","person":"팀장","deadline":"1주"},
        {"action":"근무환경 만족도 조사","person":"HR팀","deadline":"1주"},
    ]
    if level in ["critical","high"]:
        immediate = [
            {"action":"CEO/임원진 긴급 면담","person":"CEO","deadline":"24~48시간"},
            {"action":"특별 보상 인상 검토","person":"CHO","deadline":"48시간"},
            {"action":"프로젝트/팀 재배치","person":"부서장","deadline":"1주"},
        ]
    else:
        immediate = base
    follow = [{"action":"전담 멘토/성장 로드맵","person":"CHO/HR","deadline":"2주"}] if level=="critical" else [{"action":"외부 교육/세미나","person":"HR팀","deadline":"2주"}]
    return {
        "title": "🚨 긴급 리텐션 패키지" if level=="critical" else "⚠️ 집중 관리 패키지" if level=="high" else "🎯 예방적 관리 패키지" if level=="medium" else "🙂 정기 케어 패키지",
        "budget": round((salary or 0)*budgets[level]/10000),
        "timeline": timelines[level], "immediate": immediate, "follow": follow, "level": level
    }

if sel != "선택하세요" and "EMP_NAME" in SF:
    row = SF[SF["EMP_NAME"].astype(str)==sel].iloc[0]
    risk_val = float(row.get("RISK")) if pd.notna(row.get("RISK")) else 0.0
    sal_val  = float(row.get("SALARY")) if "SALARY" in SF and pd.notna(row.get("SALARY")) else 0.0
    pkg = gen_pkg(risk_val, sal_val); roi = calc_roi(sal_val)

    left2, right2 = st.columns(2)
    with left2:
        st.markdown(f"""
        <div style="padding:16px;border-radius:16px;background:#fff5f5;border:1px solid {DEFAULT_THEME['primary']}33">
          <div style="font-weight:700;color:{DEFAULT_THEME['primary']}">{pkg['title']}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:8px">
            <div><div style="color:{DEFAULT_THEME['secondary']}">예산</div><div style="font-weight:700">{pkg['budget']:,}만원</div></div>
            <div><div style="color:{DEFAULT_THEME['secondary']}">기한</div><div style="font-weight:700">{pkg['timeline']}</div></div>
            <div><div style="color:{DEFAULT_THEME['secondary']}">손실 방지</div><div style="font-weight:700">{int(roi['total']/10000):,}만원</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with right2:
        st.write("**즉시 연락** (데모 연락처)")
        cA, cB = st.columns(2)
        if "EMAIL" in SF:
            cA.link_button("📧 이메일", f"mailto:{row.get('EMAIL','')}?subject={sel}%20면담%20요청", use_container_width=True)
        if "PHONE" in SF:
            cB.link_button("📞 전화", f"tel:{row.get('PHONE','')}", use_container_width=True)

    st.write("**실행 체크리스트**")
    done_im, done_f = [], []
    for i,a in enumerate(pkg["immediate"]):
        if st.checkbox(f"{a['action']} (담당:{a['person']}, 기한:{a['deadline']})", key=f"im_{i}"):
            done_im.append(a)
    for i,a in enumerate(pkg["follow"]):
        if st.checkbox(f"{a['action']} (담당:{a['person']}, 기한:{a['deadline']})", key=f"fu_{i}"):
            done_f.append(a)

    report = {
        "employee": sel,
        "org": row.get("ORG"),
        "talent": row.get("TALENT") if "TALENT" in SF else None,
        "risk": risk_val,
        "budget(만원)": pkg["budget"],
        "expected_loss_saved(만원)": int(roi["total"]/10000),
        "completed_immediate":[a["action"] for a in done_im],
        "completed_follow":[a["action"] for a in done_f],
    }
    st.download_button("리텐션 리포트 JSON 다운로드", data=json.dumps(report, ensure_ascii=False, indent=2),
                       file_name=f"retention_report_{sel}.json", mime="application/json")

st.divider()

# ---------------- Export filtered data ----------------
if not F.empty:
    buff = io.StringIO(); F.to_csv(buff, index=False, encoding="utf-8-sig")
    st.download_button("필터된 데이터 CSV 다운로드", data=buff.getvalue(), file_name="filtered_hr_data.csv", mime="text/csv")

st.caption("© SK Retention Dashboard — Mapping-Stable Edition")

def load_logo():
    """Return a path to an sRGB, non-transparent PNG for consistent SK colors."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    # Ensure placeholder exists if official logo missing
    ensure_logo()
    src_path = LOGO_PATH
    try:
        from PIL import Image, ImageCms
        im = Image.open(src_path)
        # Convert palette/CMYK/LA/RGBA to RGB on white
        if im.mode in ("LA","RGBA","P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255,255,255,255))
            im = Image.alpha_composite(bg, im).convert("RGB")
        elif im.mode not in ("RGB","L"):
            im = im.convert("RGB")
        # Convert embedded ICC to sRGB if present
        try:
            icc = im.info.get("icc_profile")
            if icc:
                src_prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                dst_prof = ImageCms.createProfile("sRGB")
                im = ImageCms.profileToProfile(im, src_prof, dst_prof, outputMode="RGB")
        except Exception:
            pass
        fixed_path = os.path.join(ASSETS_DIR, "sk_group_logo_srgb.png")
        im.save(fixed_path, format="PNG")
        return fixed_path if os.path.exists(fixed_path) else src_path
    except Exception:
        return src_path
