import streamlit as st
import pickle
import numpy as np

# Load model dan threshold dari file
@st.cache_resource
def load_model():
    with open("model_with_threshold.pkl", "rb") as f:
        obj = pickle.load(f)
    return obj['model'], obj['threshold']

model, threshold = load_model()

# Title
st.title("Diabetes Prediction")
st.write("Input patient data for prediction diabetes.")

# Count DPF
weights = {
    "Mother": 1.0,
    "Father": 1.0,
    "Sibling": 0.75,
    "Grandparent": 0.5,
    "Aunt/uncle": 0.25,
    "Cousin": 0.1
}

def family():
    num_members = st.number_input("How many family members you have diagnose diabetes?", min_value=0, max_value=20, value=1, step=1)

    family_data = []

    for i in range(num_members):
        st.subheader(f"Family Member #{i+1}")
        relation = st.selectbox(f"Relationship of family member #{i+1}", options=list(weights.keys()), key=f"rel{i}")
        diabetes = st.checkbox(f"Does this person have diabetes?", key=f"diab{i}")

        diagnosis_age = None
        if diabetes:
            diagnosis_age = st.number_input(
                f"Age at diabetes diagnosis (if known)",
                min_value=1, max_value=120, value=50, step=1,
                key=f"age{i}"
            )
            if diagnosis_age == 0:
                diagnosis_age = np.random.rand(10, 81)

        family_data.append({
            "relation": relation,
            "diabetes": diabetes,
            "diagnosis_age": diagnosis_age
        })
    
    return family_data

def calculate_dpf():

    family_data = family()
    total_score = 0
    total_weight = 0

    for member in family_data:
        relation = member["relation"]
        has_diabetes = member.get("diabetes", False)

        if has_diabetes:
            age = member.get("diagnosis_age", 60)
            weight = weights.get(relation, 0.1)
            score = weight * (1 / age)
            total_score += score
            total_weight += weight

    if total_weight == 0:
        return 0.0
    else:
        return 2.5 * (total_score / total_weight)

def bmi_count(weight, height):
    return weight / height * height

# Input Form

gender = st.selectbox("Gender", options=["Male", "Female"])
if gender == "Male": gender_value = 0
elif gender == "Female": gender_value = 1

if(gender_value == 1):
    pregenancy = st.number_input("Number of Pregnancies (0 - 20)", min_value=0, max_value=20, step=1)
else:
    pregenancy = 0

age = st.number_input("Age", min_value=0, max_value=120, step=1)

glucose = st.number_input("Blood Glucose Level (50 - 350)", min_value=50, max_value=350, step=1)

blood_pressure = st.number_input("Blood Pressure (30 - 200)", min_value=30, max_value=200, step=1)

skin_thickness = st.number_input("Skin Thickness (0 - 100)", min_value=0, max_value=100, step=1)

weight = st.number_input("Input Weight(in kg)", min_value=10, max_value=200, step=1)

height = st.number_input("Input Height(in cm)", min_value=10, max_value=300, step=1)

diabetes_pedigree = calculate_dpf()

bmi = bmi_count(weight, height)


# Prediksi saat tombol ditekan
if st.button("Prediksi"):
    # Buat input feature array sesuai urutan model
    input_data = np.array([[pregenancy, glucose, blood_pressure, skin_thickness, bmi, diabetes_pedigree, age]])

    # Dapatkan probabilitas prediksi dari model
    prob = model.predict_proba(input_data)[0][1]

    # Terapkan threshold khusus
    pred = int(prob >= threshold)

    st.subheader("Prediction Result")
    st.write(f"Diabetes Probability: {prob * 100:.2f}%")
    if pred == 1:
        st.error("The patient is predicted to be DIABETIC.")
    else:
        st.success("The patient is predicted to be NON-DIABETIC.")
