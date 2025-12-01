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
            for c in ("Skills","skills","SkillsList"):
                if c in df.columns:
                    all_sk = df[c].dropna().astype(str).str.split(r",|;").explode().str.strip()
                    all_sk = all_sk[all_sk != ""]
                    skill_pool = sorted(all_sk.unique().tolist())[:200]
                    break

    # fallbacks
    if not skill_pool:
        skill_pool = ["Python","SQL","Excel","PowerBI","Communication","Leadership","Java","C++","JavaScript","Docker","Linux","AWS","Azure","NLP","CV"]
    if not degree_vals:
        degree_vals = ["B.Com","B.E","B.Tech","B.Sc","M.Sc","MBA","Other"]
    if not domain_vals:
        domain_vals = ["Data Science","Finance","Mechanical","Civil","Electrical","Marketing","Business","HR","Security","Operations"]
    if not location_vals:
        location_vals = ["City A","City B"]
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

# ---------- determine feature order from X_train_preprocessed.csv header (definitive) ----------
def infer_feature_order():
    # prefer metadata if available
    if metadata and isinstance(metadata, dict) and "feature_order" in metadata:
        return metadata["feature_order"]
    # try header of X_train_preprocessed.csv or first found preprocessed file
    for d in SEARCH_DIRS:
        p = d / "X_train_preprocessed.csv"
        if p.exists():
            try:
                return list(pd.read_csv(p, nrows=0).columns)
            except Exception:
                pass
    # fallback: last-resort default (shouldn't be used)
    return ["Degree","Domain","Skill1","Skill2","Skill3","Certification","Age","Location","Education_Level","University_Tier"]

FEATURE_ORDER = infer_feature_order()

# show feature order so you can verify in UI
st.write("DEBUG → Inferred FEATURE_ORDER:", FEATURE_ORDER)

# ---------- build feature vector aligned to FEATURE_ORDER (NO Experience, NO Education Gap) ----------
def build_vector_from_ui(education, degree, domain, skill1, skill2, skill3, certificate, age, location, edu_level, uni_tier):
    fmap = {
        "Degree": degree,
        "Domain": domain,
        "Skill1": skill1 if skill1 != "None" else "",
        "Skill2": skill2 if skill2 != "None" else "",
        "Skill3": skill3 if skill3 != "None" else "",
        "Certification": certificate,
        "Age": float(age),
        "Location": location,
        "Education_Level": edu_level,
        "University_Tier": uni_tier
    }

    vector = []
    for fname in FEATURE_ORDER:
        val = fmap.get(fname, 0.0)
        # If categorical_encoders exist, try to transform
        encoded_val = None
        if fname in ("Degree","Education_Level","Location","Domain","Certification","University_Tier"):
            try:
                if isinstance(categorical_encoders, dict):
                    # attempt to find a matching encoder key
                    for key, enc in categorical_encoders.items():
                        if key and isinstance(key, str) and key.lower().startswith(fname.lower().split()[0]):
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
                try:
                    vector.append(float(encoded_val))
                    continue
                except Exception:
                    pass
            # fallback: if value already numeric-like, use it
            try:
                vector.append(float(val))
                continue
            except Exception:
                # last resort stable hash mapped into [0,1)
                vector.append(float(abs(hash(str(val))) % 1000) / 1000.0)
                continue
        elif "skill" in fname.lower() or fname in ("Skill1","Skill2","Skill3"):
            # placeholder numeric for skill columns - if model trained with numeric skill encodings,
            # we expect the dropdown choices to be the same numeric strings as in X_train_preprocessed.
            try:
                vector.append(float(val) if val != "" else 0.0)
            except Exception:
                # if val is text, try to hash
                vector.append(float(abs(hash(str(val))) % 1000) / 1000.0 if val else 0.0)
        else:
            # numeric fallback
            try:
                vector.append(float(val))
            except Exception:
                vector.append(0.0)

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

            # >>> NEW: Education Gap input (for presentation only, NOT used for model)
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

            submit = st.form_submit_button("Predict Job Role")

        if submit:
            # Build vector (NO Experience, NO Education Gap included)
            X_user, used_order = build_vector_from_ui(education, degree, domain, skill1, skill2, skill3, certificate, age, location, edu_level, uni_tier)

            st.write("Feature order used (length):", len(used_order))
            st.write("Used order:", used_order)
            st.write("Input vector:", X_user.tolist())

            if model is None:
                st.warning("No trained model detected. Go to Admin to upload/retrain.")
            else:
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
                    top = []
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_try)[0]
                        classes = list(model.classes_)
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
                    if len(labels) > 0:
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

            # save submission row (save education_gap for presentation only)
            user = st.session_state.get("user", {"email":"guest","name":"guest"})
            education_gap_val = float(education_gap) if 'education_gap' in locals() else 0.0
            education_gap_dummy = 1 if education_gap_val > 0 else 0

            row = {
                "email": user.get("email","guest"),
                "name": user.get("name","guest"),
                "education": education, "degree": degree, "domain": domain,
                "skill1": skill1, "skill2": skill2, "skill3": skill3,
                "certificate": certificate, "age": age, "location": location,
                "education_level": edu_level, "university_tier": uni_tier,
                "education_gap": education_gap_val,
                "education_gap_dummy": education_gap_dummy
            }

            # add predictions (if available)
            if 'top' in locals() and top:
                for i,(r,p) in enumerate(top[:3], start=1):
                    row[f"prediction_{i}"] = r
                    row[f"confidence_{i}"] = round(float(p*100),2)

            try:
                if USER_DATA.exists():
                    dfu = pd.read_csv(USER_DATA)
                    dfu = pd.concat([dfu, pd.DataFrame([row])], ignore_index=True)
                else:
                    dfu = pd.DataFrame([row])
                dfu.to_csv(USER_DATA, index=False)
                st.success("Saved submission.")
            except Exception as e:
                st.error("Failed to save submission.")
                st.exception(e)

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
        if str(user.get("role","")).lower() != "admin" and str(user.get("email","")).lower() != "admin@example.com":
            st.error("Admin required")
        else:
            st.header("Admin Dashboard — Manage Users, Data & Model")

            # --- USERS MANAGEMENT ---
            st.subheader("User Management")
            if USERS_DB.exists():
                df_users = pd.read_csv(USERS_DB)
                st.write("Registered users:")
                st.dataframe(df_users)
                del_email = st.text_input("Email to delete (exact match)", value="")
                if st.button("Delete user"):
                    if del_email.strip() == "":
                        st.error("Enter an email to delete.")
                    else:
                        if del_email in df_users["email"].astype(str).values:
                            df_users = df_users[df_users["email"].astype(str) != del_email]
                            df_users.to_csv(USERS_DB, index=False)
                            st.success(f"Deleted user: {del_email}")
                        else:
                            st.error("Email not found.")
                with st.form("add_user_form", clear_on_submit=True):
                    st.markdown("**Add new user**")
                    new_email = st.text_input("Email")
                    new_name = st.text_input("Name")
                    new_pwd = st.text_input("Password", type="password")
                    new_role = st.selectbox("Role", ["User", "Admin"])
                    add_submit = st.form_submit_button("Add user")
                if add_submit:
                    if not new_email or not new_name or not new_pwd:
                        st.error("Fill all fields to add user.")
                    else:
                        df_users = pd.read_csv(USERS_DB) if USERS_DB.exists() else pd.DataFrame(columns=["email","name","password","role"])
                        if new_email in df_users["email"].astype(str).values:
                            st.error("Email already exists.")
                        else:
                            df_users = pd.concat([df_users, pd.DataFrame([{"email":new_email,"name":new_name,"password":new_pwd,"role":new_role}])], ignore_index=True)
                            df_users.to_csv(USERS_DB, index=False)
                            st.success("User added.")
            else:
                st.info("No users database found. Create admin user by registering first.")
                if st.button("Init users db (create default admin)"):
                    init_users()
                    st.success("users.csv created with default admin.")

            st.markdown("---")

            # --- USER SUBMISSIONS MANAGEMENT & VISUALS ---
            st.subheader("User Submissions & Visualizations")
            if USER_DATA.exists():
                dfu = pd.read_csv(USER_DATA)
                st.write(f"Total submissions: {len(dfu)}")
                cols = dfu.columns.tolist()
                cols_to_show = st.multiselect("Columns to show", cols, default=["email","education","degree","skill1","skill2","skill3","prediction_1","confidence_1","education_gap"])
                st.dataframe(dfu[cols_to_show].tail(200))

                st.markdown("**Delete a submission**")
                idx_list = dfu.index.tolist()
                selected_idx = st.selectbox("Select row index to delete", idx_list)
                if st.button("Delete selected submission"):
                    flagged_row = dfu.loc[[selected_idx]]
                    dfu = dfu.drop(index=selected_idx).reset_index(drop=True)
                    dfu.to_csv(USER_DATA, index=False)
                    st.success(f"Deleted row {selected_idx}.")
                    if FLAGGED.exists():
                        dff = pd.read_csv(FLAGGED)
                        dff = pd.concat([dff, flagged_row], ignore_index=True)
                    else:
                        dff = flagged_row
                    dff.to_csv(FLAGGED, index=False)

                st.markdown("---")
                st.markdown("### Visualizations")
                if any(c.lower().startswith("prediction") for c in dfu.columns):
                    pred_col = None
                    for c in dfu.columns:
                        if c.lower().startswith("prediction"):
                            pred_col = c
                            break
                    if pred_col:
                        top_roles = dfu[pred_col].fillna("Unknown").value_counts().reset_index()
                        top_roles.columns = ["role","count"]
                        st.markdown("**Top predicted roles (by count)**")
                        fig1, ax1 = plt.subplots(figsize=(8, min(6, 0.4 * len(top_roles))))
                        sns.barplot(x="count", y="role", data=top_roles.head(20), ax=ax1)
                        ax1.set_xlabel("Count")
                        ax1.set_ylabel("Role")
                        st.pyplot(fig1)
                else:
                    st.info("No prediction columns available to plot top roles.")

                if "education" in dfu.columns or "education_level" in dfu.columns:
                    edu_col = "education" if "education" in dfu.columns else "education_level"
                    edu_counts = dfu[edu_col].fillna("Unknown").value_counts().reset_index()
                    edu_counts.columns = ["education","count"]
                    st.markdown("**Education level distribution**")
                    fig2, ax2 = plt.subplots(figsize=(8,4))
                    sns.barplot(x="count", y="education", data=edu_counts.head(20), ax=ax2)
                    ax2.set_xlabel("Count")
                    st.pyplot(fig2)

                skill_cols = [c for c in dfu.columns if c.lower().startswith("skill")]
                if skill_cols:
                    all_sk = []
                    for sc in skill_cols:
                        all_sk += dfu[sc].dropna().astype(str).tolist()
                    all_sk = [s.strip() for s in all_sk if s and s.strip().lower() not in ("none","nan","nan.0","")]
                    if all_sk:
                        sk_series = pd.Series(all_sk).value_counts().reset_index()
                        sk_series.columns = ["skill","count"]
                        st.markdown("**Top skills (from skill1/2/3)**")
                        fig3, ax3 = plt.subplots(figsize=(8, min(6, 0.25 * len(sk_series))))
                        sns.barplot(x="count", y="skill", data=sk_series.head(30), ax=ax3)
                        ax3.set_xlabel("Count")
                        st.pyplot(fig3)

                if "education_gap" in dfu.columns:
                    st.markdown("**Education Gap (years) distribution**")
                    fig4, ax4 = plt.subplots(figsize=(8,3))
                    sns.histplot(dfu["education_gap"].dropna().astype(float), bins=10, ax=ax4)
                    ax4.set_xlabel("Education Gap (years)")
                    st.pyplot(fig4)

                st.markdown("---")
                st.download_button("Download user_data.csv", dfu.to_csv(index=False).encode("utf-8"), "user_data.csv")
                if FLAGGED.exists():
                    df_flagged = pd.read_csv(FLAGGED)
                    st.download_button("Download flagged_predictions.csv", df_flagged.to_csv(index=False).encode("utf-8"), "flagged_predictions.csv")
                else:
                    st.info("No flagged rows yet.")
            else:
                st.info("No user submissions yet.")

            st.markdown("---")

            # --- MODEL MANAGEMENT ---
            st.subheader("Model & Retrain Management")
            st.markdown("Upload a new trained model (joblib / pickle) to replace the current model file used by the app.")
            uploaded_model = st.file_uploader("Upload model file (.joblib, .pkl)", type=["joblib","pkl","pickle"])
            if uploaded_model:
                try:
                    target_dir = BASE / "model_output"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    filename = uploaded_model.name
                    out_path = target_dir / filename
                    with open(out_path, "wb") as f:
                        f.write(uploaded_model.getbuffer())
                    st.success(f"Saved uploaded model to {out_path}. You may need to restart the app to pick it up.")
                    if st.button("Load uploaded model into memory now"):
                        loaded_model = load_pickle_or_joblib(out_path)
                        if loaded_model is not None:
                            std_path = target_dir / "rf_model.joblib"
                            joblib.dump(loaded_model, std_path)
                            st.success("Model loaded and saved as model_output/rf_model.joblib")
                        else:
                            st.error("Could not load the uploaded model.")
                except Exception as e:
                    st.error("Failed to save uploaded model.")
                    st.exception(e)

            st.markdown("---")

            st.subheader("Upload encoded CSV to retrain model (existing flow)")
            uploaded = st.file_uploader("Upload encoded CSV", type=["csv"], key="retrain_csv")
            if uploaded:
                try:
                    df_new = pd.read_csv(uploaded)
                    st.dataframe(df_new.head())
                    if st.button("Retrain", key="retrain_button"):
                        if metadata and isinstance(metadata, dict) and "feature_order" in metadata:
                            feat_cols = metadata["feature_order"]
                        else:
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
st.caption("Edu2Job")
