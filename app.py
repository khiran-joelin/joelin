# app.py
# Edu2Job - UI wired to your dataset fields: Education, Certification, Skill1, Skill2, Skill3 + others.
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Edu2Job — Predicting Job Roles", layout="wide")
BASE = Path(__file__).parent

# ---------- search dirs & filenames ----------
SEARCH_DIRS = [BASE, Path.cwd(), Path(r"D:\joelin\infosys springboard"), Path("/mnt/data")]

MODEL_FILES = [
    "model_output/rf_model.joblib",
    "model_output/rf_tuned_best.joblib",
    "models/job_role_model.joblib",
    "models/job_role_model.pkl",
    "models/rf_model.joblib"
]

ENCODER_CANDIDATES = ["target_encoder.pkl", "target_encoder.joblib",
                      "categorical_label_encoders.pkl", "label_encoders.pkl",
                      "numerical_scaler.pkl", "numerical_scaler.joblib"]

PREPROCESS_CANDIDATES = [
    "cleaned_dataset_before_encoding.csv",
    "fully_cleaned_encoded_dataset.csv",
    "fully_preprocessed_dataset_with_target_encoded.csv",
    "X_train_preprocessed.csv"
]

META_NAME = "models/model_metadata.pkl"
USERS_DB = BASE / "users.csv"
USER_DATA = BASE / "user_data.csv"
FLAGGED = BASE / "flagged_predictions.csv"

# ---------- utilities ----------
def find_first(names):
    for d in SEARCH_DIRS:
        for n in names:
            p = d / n
            if p.exists():
                return p
    return None

def find_model_path():
    for name in MODEL_FILES:
        p = find_first([name])
        if p:
            return p
    # fallback: any rf_model.* in search dirs
    for d in SEARCH_DIRS:
        for ext in ("joblib", "pkl", "pickle"):
            p = d / "model_output" / f"rf_model.{ext}"
            if p.exists():
                return p
    return None

def load_pickle_or_joblib(p):
    try:
        return joblib.load(p)
    except Exception:
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

def load_metadata():
    p = find_first([META_NAME])
    if p:
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            try:
                return joblib.load(p)
            except Exception:
                return None
    return None

# ---------- load dataset used to populate dropdowns ----------
def load_primary_dataset():
    p = find_first(PREPROCESS_CANDIDATES)
    if p:
        try:
            df = pd.read_csv(p)
            return df, p
        except Exception:
            try:
                return pd.read_csv(p, encoding="latin1"), p
            except Exception:
                return None, None
    return None, None

df_raw, raw_path = load_primary_dataset()

# ---------- build dropdown pools from dataset ----------
def build_pools(df):
    # defaults
    education_vals = ["High School", "Diploma", "B.Com", "B.E", "B.Tech", "Bachelor's", "Master's", "MBA", "PhD", "Other"]
    certification_vals = ["None"]
    skill_pool = []  # will populate from Skill1/Skill2/Skill3 or Skills column
    degree_vals = []
    domain_vals = []
    location_vals = []
    university_tiers = []

    if df is not None:
        # Education candidates
        for c in ("Education_Level","Education","education","Education Level"):
            if c in df.columns:
                education_vals = sorted(df[c].dropna().astype(str).unique().tolist())
                break

        # Certification column
        for c in ("Certification","Certifications","Certificates","certificate"):
            if c in df.columns:
                certs = sorted(df[c].dropna().astype(str).unique().tolist())
                certification_vals = ["None"] + [v for v in certs if str(v).strip() != "" and str(v).strip().lower() != "none"]
                break

        # Degree
        for c in ("Degree","degree"):
            if c in df.columns:
                degree_vals = sorted(df[c].dropna().astype(str).unique().tolist())
                break

        # Domain
        for c in ("Domain","domain","Field"):
            if c in df.columns:
                domain_vals = sorted(df[c].dropna().astype(str).unique().tolist())
                break

        # Location
        for c in ("Location","location","City"):
            if c in df.columns:
                location_vals = sorted(df[c].dropna().astype(str).unique().tolist())
                break

        # University tier
        for c in ("University_Tier","University Tier","UniversityTier"):
            if c in df.columns:
                university_tiers = sorted(df[c].dropna().astype(str).unique().tolist())
                break

        # gather skill pool from Skill1/Skill2/Skill3 or 'Skills'
        if any(col in df.columns for col in ("Skill1","Skill2","Skill3")):
            skill_vals = []
            for col in ("Skill1","Skill2","Skill3"):
                if col in df.columns:
                    skill_vals += df[col].dropna().astype(str).tolist()
            skill_vals = [s for s in skill_vals if str(s).strip() != "" and str(s).strip().lower() != "none"]
            # dedupe preserving order
            seen = set()
            pool = []
            for s in skill_vals:
                s0 = s.strip()
                if s0 not in seen:
                    seen.add(s0)
                    pool.append(s0)
            skill_pool = sorted(pool)[:200]
        else:
            # if there is a single Skills column with csv lists
            for c in ("Skills","skills","SkillsList"):
                if c in df.columns:
                    all_sk = df[c].dropna().astype(str).str.split(r",|;").explode().str.strip()
                    all_sk = all_sk[all_sk != ""]
                    skill_pool = sorted(all_sk.unique().tolist())[:200]
                    break

    # fallbacks if something empty
    if not skill_pool:
        skill_pool = ["Python","SQL","Excel","PowerBI","Communication","Leadership","Java","C++","JavaScript","Docker","Linux","AWS","Azure","NLP","CV"]

    if not degree_vals:
        degree_vals = ["B.Com","B.E","B.Tech","B.Sc","M.Sc","MBA","Other"]

    if not domain_vals:
        domain_vals = ["Data Science","Finance","Mechanical","Civil","Electrical","Marketing","Business","HR","Security","Operations"]

    if not location_vals:
        location_vals = ["City A","City B"]  # small fallback

    if not university_tiers:
        university_tiers = ["Tier 1","Tier 2","Tier 3"]

    return {
        "education": education_vals,
        "certificates": certification_vals,
        "skills": skill_pool,
        "degree": degree_vals,
        "domain": domain_vals,
        "location": location_vals,
        "uni_tier": university_tiers
    }

POOLS = build_pools(df_raw)

# ---------- load model & encoders & metadata ----------
model_path = find_first(MODEL_FILES)
model = load_pickle_or_joblib(model_path) if model_path else None

# encoders
enc_map = {}
for n in ENCODER_CANDIDATES:
    p = find_first([n])
    if p:
        enc_map[n] = load_pickle_or_joblib(p)

target_encoder = enc_map.get("target_encoder.pkl") or enc_map.get("target_encoder.joblib")
categorical_encoders = enc_map.get("categorical_label_encoders.pkl") or enc_map.get("label_encoders.pkl")
numerical_scaler = enc_map.get("numerical_scaler.pkl") or enc_map.get("numerical_scaler.joblib")
metadata = load_metadata()

# ---------- helper to figure feature_order used in training ----------
def infer_feature_order():
    if metadata and isinstance(metadata, dict) and "feature_order" in metadata:
        return metadata["feature_order"]
    # try header of X_train_preprocessed.csv
    for d in SEARCH_DIRS:
        p = d / "X_train_preprocessed.csv"
        if p.exists():
            try:
                return list(pd.read_csv(p, nrows=0).columns)
            except Exception:
                pass
    # fallback to assume common order found in your dataset:
    # Degree, Domain, Skill1, Skill2, Skill3, Certification, Age, Location, Education_Level, University_Tier
    # (this fallback is used only if no metadata/file header found)
    return ["Degree","Domain","Skill1","Skill2","Skill3","Certification","Age","Location","Education_Level","University_Tier"]

FEATURE_ORDER = infer_feature_order()

# ---------- build feature vector aligned to FEATURE_ORDER ----------
def build_vector_from_ui(education, degree, domain, skill1, skill2, skill3, certificate, age, location, edu_level, uni_tier):
    # create map of values using the naming that appears in the dataset
    fmap = {}
    # handle Degree: if degree exists in pool, keep as string or encode via categorical_encoders
    fmap["Degree"] = degree if not isinstance(degree, (int,float)) else degree
    fmap["Domain"] = domain
    fmap["Skill1"] = skill1 if skill1 != "None" else ""
    fmap["Skill2"] = skill2 if skill2 != "None" else ""
    fmap["Skill3"] = skill3 if skill3 != "None" else ""
    fmap["Certification"] = certificate
    fmap["Age"] = float(age)
    fmap["Location"] = location
    fmap["Education_Level"] = edu_level
    fmap["University_Tier"] = uni_tier

    # Convert to numeric vector that matches training preprocessing:
    # If numeric scaler / categorical encoders exist, we will attempt to follow them.
    # Build an ordered feature list according to FEATURE_ORDER and fill missing with zeros or encoded values.
    vector = []
    for fname in FEATURE_ORDER:
        val = fmap.get(fname, 0.0)
        # naive encoding behavior:
        # - If categorical_encoders present and contains a mapping for 'Degree' or 'Education_Level', use it
        if fname in ("Degree","Education_Level","Location","Domain","Certification","University_Tier"):
            encoded_val = None
            try:
                if isinstance(categorical_encoders, dict):
                    # try keyed encoders
                    # common keys could be 'Degree', 'edu', etc. attempt case-insensitive match
                    for key, enc in categorical_encoders.items():
                        if key.lower().startswith(fname.lower().split()[0]):
                            try:
                                encoded_val = enc.transform([val])[0]
                                break
                            except Exception:
                                continue
                elif hasattr(categorical_encoders, "transform"):
                    encoded_val = categorical_encoders.transform([val])[0]
            except Exception:
                encoded_val = None
            if encoded_val is not None:
                vector.append(float(encoded_val))
            else:
                # fallback: if value looks numeric already, keep; else hash to index
                try:
                    vector.append(float(val))
                except Exception:
                    # stable hash to numeric index
                    vector.append(float(abs(hash(str(val))) % 1000) / 1000.0)
        elif fname in ("Skill1","Skill2","Skill3"):
            # simple indicator: if skill equals the label, set 1 else 0
            # but training might have had skill columns as separate Skill_Feature_0..n — handle mapping in metadata if available
            vector.append(0.0)  # placeholder — more robust mapping below
        else:
            # generic numeric
            try:
                vector.append(float(val))
            except Exception:
                vector.append(0.0)

    # If metadata has skill-based feature_order (e.g., Skill_Feature_0..14), map selected skills into those slots:
    if metadata and isinstance(metadata, dict) and "feature_order" in metadata:
        order = metadata["feature_order"]
        skill_cols = [c for c in order if "skill" in c.lower()]
        if skill_cols:
            # create initial zero map for skill cols
            # compute set of selected skills lowercased
            sel = {s.strip().lower() for s in [skill1, skill2, skill3] if s and str(s).strip().lower() != "none"}
            # If metadata contains 'skill_names', use them to map
            if "skill_names" in metadata and isinstance(metadata["skill_names"], (list,tuple)):
                skill_names = [s.lower() for s in metadata["skill_names"][:len(skill_cols)]]
                # build a mapping col -> 1 if skill name in sel
                skill_map = {col: 1.0 if skill_names[i] in sel else 0.0 for i,col in enumerate(skill_cols)}
            else:
                # otherwise map selected skills to first 3 skill_cols
                skill_map = {}
                for i,col in enumerate(skill_cols):
                    if i == 0:
                        skill_map[col] = 1.0 if skill1 and skill1.lower() != "none" else 0.0
                    elif i == 1:
                        skill_map[col] = 1.0 if skill2 and skill2.lower() != "none" else 0.0
                    elif i == 2:
                        skill_map[col] = 1.0 if skill3 and skill3.lower() != "none" else 0.0
                    else:
                        skill_map[col] = 0.0
            # now insert these skill_map values into vector according to order indexes
            final_vec = []
            for fname, val in zip(FEATURE_ORDER, vector):
                if fname in skill_map:
                    final_vec.append(float(skill_map[fname]))
                else:
                    final_vec.append(float(val))
            vector = final_vec

    arr = np.array(vector, dtype=float).reshape(1, -1)

    # apply numeric scaler if present and dims match
    try:
        if numerical_scaler and hasattr(numerical_scaler, "transform"):
            if arr.shape[1] == getattr(numerical_scaler, "n_features_in_", arr.shape[1]):
                arr = numerical_scaler.transform(arr)
    except Exception:
        pass

    return arr, list(FEATURE_ORDER)

# ---------- User auth helper ----------
def init_users():
    if not USERS_DB.exists():
        df = pd.DataFrame([{"email":"admin@example.com","name":"admin","password":"admin123","role":"Admin"}])
        df.to_csv(USERS_DB, index=False)

def register_user(email, name, password, role="User"):
    init_users()
    df = pd.read_csv(USERS_DB)
    if email in df["email"].astype(str).values:
        return False, "Email already registered."
    df = pd.concat([df, pd.DataFrame([{"email":email,"name":name,"password":password,"role":role}])], ignore_index=True)
    df.to_csv(USERS_DB, index=False)
    return True, "Registered."

def authenticate(email, password):
    init_users()
    df = pd.read_csv(USERS_DB)
    row = df[(df["email"].astype(str)==str(email)) & (df["password"].astype(str)==str(password))]
    if len(row) == 1:
        return True, row.iloc[0].to_dict()
    else:
        return False, None

# ---------- UI header ----------
st.title("Edu2Job — Predicting Job Roles from Educational Background")

init_users()
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

nav = st.sidebar.radio("Navigation", ["Home","Register","Login","Admin"] if not st.session_state["logged_in"] else ["Home","Predict","Profile","Admin","Logout"])

if nav == "Logout" or (st.sidebar.button("Logout") and st.session_state["logged_in"]):
    st.session_state.clear()
    st.rerun()

# ---------- Register ----------
if nav == "Register":
    st.header("Register")
    with st.form("reg"):
        name = st.text_input("Full name")
        email = st.text_input("Email")
        pwd = st.text_input("Password", type="password")
        pwd2 = st.text_input("Confirm password", type="password")
        submit = st.form_submit_button("Register")
    if submit:
        if not name or not email or not pwd:
            st.error("Fill all fields")
        elif pwd != pwd2:
            st.error("Passwords don't match")
        else:
            ok,msg = register_user(email, name, pwd)
            if ok:
                st.success("Registered — please login")
            else:
                st.error(msg)

# ---------- Login ----------
elif nav == "Login":
    st.header("Login")
    with st.form("login"):
        email = st.text_input("Email")
        pwd = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
    if submit:
        ok, user = authenticate(email, pwd)
        if ok:
            st.session_state["logged_in"] = True
            st.session_state["user"] = user
            st.success("Logged in")
            st.rerun()
        else:
            st.error("Invalid credentials")

# ---------- Home ----------
elif nav == "Home" and not st.session_state["logged_in"]:
    st.header("Welcome to Edu2Job")
    st.info("Register or login to predict. Default admin: admin@example.com / admin123")

# ---------- Predict ----------
elif nav == "Predict" or (nav == "Home" and st.session_state["logged_in"]):
    if not st.session_state["logged_in"]:
        st.warning("Please login")
    else:
        st.header("Candidate Profile")

        with st.form("predict_form"):
            # dropdowns from POOLS
            education = st.selectbox("Education", POOLS["education"])
            degree = st.selectbox("Degree", POOLS["degree"])
            domain = st.selectbox("Domain", POOLS["domain"])
            skill1 = st.selectbox("Skill 1", ["None"] + POOLS["skills"])
            skill2 = st.selectbox("Skill 2", ["None"] + POOLS["skills"])
            skill3 = st.selectbox("Skill 3", ["None"] + POOLS["skills"])

            # >>> NEW: Education Gap input (use float step to avoid mixed-type error)
            education_gap = st.number_input(
                "Education Gap (years)",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=0.5
            )

            certificate = st.selectbox("Certificate", POOLS["certificates"])
            age = st.number_input("Age", min_value=16, max_value=80, value=25)
            location = st.selectbox("Location", POOLS["location"])
            edu_level = st.selectbox("Education Level", POOLS["education"])
            uni_tier = st.selectbox("University Tier", POOLS["uni_tier"])

            # >>> IMPORTANT: form submit button (must be inside the form)
            submit = st.form_submit_button("Predict Job Role")


        if submit:
            X_user, used_order = build_vector_from_ui(education, degree, domain, skill1, skill2, skill3, certificate, age, location, edu_level, uni_tier)
            st.write("Feature order used (length):", len(used_order))
            st.write(used_order)
            st.write("Input vector:", X_user.tolist())

            if model is None:
                st.warning("No trained model detected. Go to Admin to upload/retrain.")
            else:
                # pad/trim for model expected features
                expected = getattr(model, "n_features_in_", None)
                if expected is None and metadata and isinstance(metadata, dict) and "feature_order" in metadata:
                    expected = len(metadata["feature_order"])
                if expected is not None and X_user.shape[1] != expected:
                    if X_user.shape[1] < expected:
                        X_try = np.hstack([X_user, np.zeros((1, expected - X_user.shape[1]))])
                    else:
                        X_try = X_user[:, :expected]
                else:
                    X_try = X_user

                try:
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_try)[0]
                        classes = list(model.classes_)
                        # decode classes
                        decoded = []
                        for c in classes:
                            try:
                                if target_encoder is not None and hasattr(target_encoder, "inverse_transform"):
                                    decoded.append(str(target_encoder.inverse_transform([c])[0]))
                                elif metadata and isinstance(metadata, dict) and "classes" in metadata:
                                    decoded.append(str(metadata["classes"][int(c)]))
                                else:
                                    decoded.append(str(c))
                            except Exception:
                                decoded.append(str(c))
                        top_idx = np.argsort(probs)[::-1][:10]
                        top = [(decoded[i], float(probs[i])) for i in top_idx]
                    else:
                        pred = model.predict(X_try)[0]
                        if target_encoder is not None and hasattr(target_encoder, "inverse_transform"):
                            top = [(str(target_encoder.inverse_transform([pred])[0]), 1.0)]
                        elif metadata and isinstance(metadata, dict) and "classes" in metadata:
                            top = [(str(metadata["classes"][int(pred)]), 1.0)]
                        else:
                            top = [(str(pred), 1.0)]

                    st.subheader("Top predictions")
                    for r,p in top[:5]:
                        st.write(f"**{r}** — {p*100:.2f}%")

                    # plot top probabilities
                    st.subheader("Prediction probabilities (top 10)")
                    labels = [x[0] for x in top]
                    vals = [x[1] for x in top]
                    fig, ax = plt.subplots(figsize=(8,4))
                    sns.barplot(x=vals, y=labels, ax=ax)
                    ax.set_xlabel("Probability")
                    st.pyplot(fig)

                    # feature importances if present
                    st.subheader("Feature importances")
                    if hasattr(model, "feature_importances_"):
                        names = metadata["feature_order"] if metadata and isinstance(metadata, dict) and "feature_order" in metadata else FEATURE_ORDER
                        if names and len(names) == len(model.feature_importances_):
                            fi = pd.DataFrame({"feature":names, "importance": model.feature_importances_}).sort_values("importance", ascending=False).head(20)
                            fig2, ax2 = plt.subplots(figsize=(8, min(6, 0.3*len(fi))))
                            sns.barplot(x="importance", y="feature", data=fi, ax=ax2)
                            st.pyplot(fig2)
                        else:
                            st.info("Feature importance length mismatch; cannot display.")
                    else:
                        st.info("Model does not expose sklearn-like feature_importances_.")

                except Exception as e:
                    st.error("Prediction failed.")
                    st.exception(e)

            # save submission row
            user = st.session_state.get("user", {"email":"guest","name":"guest"})
                       # use the education_gap value captured from the Predict form
            education_gap = float(education_gap) if 'education_gap' in locals() else 0.0
            education_gap_dummy = 1 if education_gap > 0 else 0

            row = {
                "email": user.get("email","guest"),
                "name": user.get("name","guest"),
                "education": education, "degree": degree, "domain": domain,
                "skill1": skill1, "skill2": skill2, "skill3": skill3,
                "certificate": certificate, "age": age, "location": location,
                "education_level": edu_level, "university_tier": uni_tier,
                "education_gap": float(education_gap),
                "education_gap_dummy": education_gap_dummy
            }

            # add predictions
            for i,(r,p) in enumerate(top[:3], start=1):
                row[f"prediction_{i}"] = r
                row[f"confidence_{i}"] = round(float(p*100),2)
            if USER_DATA.exists():
                dfu = pd.read_csv(USER_DATA)
                dfu = pd.concat([dfu, pd.DataFrame([row])], ignore_index=True)
            else:
                dfu = pd.DataFrame([row])
            dfu.to_csv(USER_DATA, index=False)
            st.success("Saved submission.")

# ---------- Profile ----------
elif nav == "Profile":
    if not st.session_state["logged_in"]:
        st.warning("Please login")
    else:
        st.header("Profile & submissions")
        user = st.session_state.get("user", {"email":"guest"})
        st.write(f"Logged in as: {user.get('name')} ({user.get('email')})")
        if USER_DATA.exists():
            dfu = pd.read_csv(USER_DATA)
            mine = dfu[dfu["email"].astype(str)==str(user.get("email"))]
            if not mine.empty:
                st.dataframe(mine.tail(50))
            else:
                st.info("No submissions yet.")
        else:
            st.info("No submissions yet.")

# ---------- Admin ----------
elif nav == "Admin":
    if not st.session_state["logged_in"]:
        st.warning("Please login")
    else:
        user = st.session_state.get("user", {})
        if str(user.get("role","")) != "Admin" and str(user.get("email","")).lower() != "admin@example.com":
            st.error("Admin required")
        else:
            st.header("Admin Dashboard")
            if USER_DATA.exists():
                dfu = pd.read_csv(USER_DATA)
                st.dataframe(dfu.tail(200))
                st.download_button("Download", dfu.to_csv(index=False).encode("utf-8"), "user_data.csv")
            else:
                st.info("No submissions yet")

            st.markdown("---")
            st.subheader("Upload encoded CSV to retrain model")
            uploaded = st.file_uploader("Upload encoded CSV", type=["csv"])
            if uploaded:
                try:
                    df_new = pd.read_csv(uploaded)
                    st.dataframe(df_new.head())
                    if st.button("Retrain"):
                        # choose feature columns
                        if metadata and isinstance(metadata, dict) and "feature_order" in metadata:
                            feat_cols = metadata["feature_order"]
                        else:
                            # guess: all columns except JobRole / target
                            candidates = [c for c in df_new.columns if c.lower() not in ("jobrole","job_role","target","label")]
                            feat_cols = candidates
                        if "JobRole" in df_new.columns:
                            df_new["target"] = df_new["JobRole"]
                        elif "target" not in df_new.columns:
                            st.error("Uploaded CSV must include JobRole or target column.")
                            st.stop()
                        X = df_new[feat_cols].values
                        y = df_new["target"].astype(str).values
                        m = RandomForestClassifier(random_state=42, n_jobs=-1)
                        m.fit(X,y)
                        outp = BASE / "model_output" / "rf_model.joblib"
                        outp.parent.mkdir(parents=True, exist_ok=True)
                        joblib.dump(m, outp)
                        meta = {"feature_order": feat_cols, "classes": list(m.classes_)}
                        meta_path = BASE / "models" / "model_metadata.pkl"
                        meta_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(meta_path, "wb") as f:
                            pickle.dump(meta,f)
                        st.success("Retrained and saved model.")
                except Exception as e:
                    st.error("Could not load uploaded CSV.")
                    st.exception(e)

st.markdown("---")
st.caption("Edu2Job — UI bound to dataset fields (Degree, Domain, Skill1/2/3, Certification, Education_Level, University_Tier, Location).")
