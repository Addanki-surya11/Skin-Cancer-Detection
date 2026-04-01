import json
from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np

from infer import SkinCancerDetector


st.set_page_config(page_title="Skin Cancer Detector", page_icon="🩺", layout="wide")
st.title("🩺 AI Skin Cancer Detection System")
st.caption("Advanced CNN + XGBoost detection with clinical risk assessment")

# Sidebar: Fitzpatrick Skin Type Selector
with st.sidebar:
    st.markdown("### 👤 User Profile")
    fitzpatrick = st.select_slider(
        "Fitzpatrick Skin Type",
        options=["I", "II", "III", "IV", "V", "VI"],
        value="III",
        help="I=Very Fair, II=Fair, III=Medium, IV=Olive, V=Brown, VI=Dark",
        key="fitzpatrick_slider"
    )
    st.session_state.fitzpatrick = fitzpatrick
    fitzpatrick_note = {
        "I": "Very fair; always burns, never tans",
        "II": "Fair; usually burns, tans minimally",
        "III": "Medium; sometimes mild burn, tans gradually",
        "IV": "Olive; rarely burns, tans easily",
        "V": "Brown; very rarely burns, tans very easily",
        "VI": "Dark; never burns, always tans well"
    }
    st.caption(f"📝 {fitzpatrick_note[fitzpatrick]}")

model_dir = st.text_input("Model artifacts folder", value="./model_artifacts")

# Disease information database with clinical notes
disease_info = {
    "AKIEC": {
        "full_name": "Actinic Keratosis / Bowen's Disease",
        "description": "Pre-cancerous lesion caused by sun exposure. Usually appears as rough, scaly patches.",
        "severity": "Medium",
        "urgency": "Monitor and treat within weeks",
        "notes": "More common in fair-skinned individuals. Risk increases with age and sun exposure.",
    },
    "BCC": {
        "full_name": "Basal Cell Carcinoma",
        "description": "Most common type of skin cancer. Grows slowly and rarely spreads.",
        "severity": "Low-Medium",
        "urgency": "Treat within months",
        "notes": "Usually treatable with high cure rates when caught early.",
    },
    "BKL": {
        "full_name": "Benign Keratosis",
        "description": "Common, harmless skin growth. Usually dark and waxy-looking.",
        "severity": "Low",
        "urgency": "No urgent treatment needed",
        "notes": "Often appears as people age. Benign but can be removed for cosmetic reasons.",
    },
    "DF": {
        "full_name": "Dermatofibroma",
        "description": "Benign skin bump, often itchy. Usually doesn't require treatment.",
        "severity": "Low",
        "urgency": "No urgent treatment needed",
        "notes": "May itch or irritate. Can be removed if bothersome.",
    },
    "MEL": {
        "full_name": "Melanoma",
        "description": "Most dangerous type of skin cancer. Can spread rapidly if not treated.",
        "severity": "High",
        "urgency": "URGENT - Biopsy and treatment needed",
        "notes": "Early detection significantly improves survival rates. Watch for ABCDE signs.",
    },
    "NV": {
        "full_name": "Nevus (Mole)",
        "description": "Common, usually harmless skin growth. Most people have multiple moles.",
        "severity": "Low",
        "urgency": "Monitor regularly for changes",
        "notes": "Monitor for ABCDE signs: Asymmetry, Border irregularity, Color, Diameter >6mm, Evolving.",
    },
    "VASC": {
        "full_name": "Vascular Lesion",
        "description": "Abnormal blood vessels in skin, such as hemangiomas or angiomas.",
        "severity": "Low",
        "urgency": "Monitor for cosmetic concerns",
        "notes": "Usually harmless but may require removal for cosmetic reasons.",
    },
}


def assess_image_quality(image_obj, skin_ratio):
    """Assess image quality and provide feedback."""
    quality_score = 0
    feedback = []
    
    # Check image resolution
    width, height = image_obj.size
    if width >= 500 and height >= 500:
        quality_score += 25
    else:
        feedback.append("⚠️ Image resolution is low (recommend 500x500 or higher)")
    
    # Check skin coverage
    if skin_ratio >= 0.7:
        quality_score += 25
        feedback.append("✅ Good skin coverage detected")
    elif skin_ratio >= 0.4:
        quality_score += 15
        feedback.append("ℹ️ Moderate skin coverage - try to fill frame more")
    else:
        feedback.append("⚠️ Low skin coverage - center the lesion")
    
    # Check for brightness (rough estimate)
    arr = np.asarray(image_obj)
    brightness = np.mean(arr) / 255
    if 0.3 <= brightness <= 0.8:
        quality_score += 25
        feedback.append("✅ Good lighting conditions")
    else:
        feedback.append("⚠️ Optimize lighting - avoid too bright or too dark")
    
    quality_score += 25  # Base score
    return min(100, quality_score), feedback


def generate_risk_score(confidence, severity):
    """Generate comprehensive risk score."""
    # Base confidence contribution (50%)
    conf_score = confidence * 50
    
    # Severity mapping (50%)
    severity_map = {"Low": 10, "Low-Medium": 20, "Medium": 30, "High": 40}
    sev_score = severity_map.get(severity, 0)
    
    total_score = conf_score + sev_score
    return min(100, total_score)


def generate_doctor_report(uploaded_filename, predicted_class, confidence, 
                          disease_data, all_probs, skin_ratio, fitzpatrick):
    """Generate a professional doctor-friendly report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""================================================================================
                    AI SKIN CANCER SCREENING REPORT
================================================================================
Report Generated: {timestamp}
Patient Information: File: {uploaded_filename} | Fitzpatrick Type: {fitzpatrick}

PRIMARY DIAGNOSIS
─────────────────────────────────────────────────────────────────────────────
Predicted Condition: {predicted_class}
Full Name: {disease_data.get('full_name', 'Unknown')}
Model Confidence: {confidence:.1%}
Description: {disease_data.get('description', 'N/A')}

CLINICAL ASSESSMENT
─────────────────────────────────────────────────────────────────────────────
Severity Level: {disease_data.get('severity', 'N/A')}
Clinical Urgency: {disease_data.get('urgency', 'N/A')}
Skin Coverage Detected: {skin_ratio:.1%}

Clinical Notes:
{disease_data.get('notes', 'N/A')}

DIFFERENTIAL DIAGNOSIS (All Predictions)
─────────────────────────────────────────────────────────────────────────────"""
    
    for i, (condition, prob) in enumerate(sorted(all_probs.items(), 
                                                   key=lambda x: x[1], reverse=True), 1):
        report += f"\n{i}. {condition.upper():12} : {prob:.1%}"
    
    report += f"""

TECHNICAL METRICS
─────────────────────────────────────────────────────────────────────────────
Model Architecture: ResNet18 (CNN) + XGBoost
Training Dataset: HAM10000 (2,039 images, 7 conditions)
Model Accuracy: 78.8%

RECOMMENDATIONS FOR CLINICIAN
─────────────────────────────────────────────────────────────────────────────
1. This is an AI-assisted screening tool, NOT a definitive diagnosis
2. Consider clinical examination and dermoscopy for confirmation
3. If confident diagnosis is >85%, urgent dermatology referral recommended
4. If diagnosis is <70%, additional assessment (biopsy) may be needed
5. Baseline photography recommended for follow-up comparison

DISCLAIMER
─────────────────────────────────────────────────────────────────────────────
This report is generated by artificial intelligence for informational purposes only.
It is NOT a substitute for professional medical diagnosis or treatment.
Always consult with a qualified dermatologist for final diagnosis and management.

================================================================================"""
    return report


@st.cache_resource
def load_detector(path: str):
    return SkinCancerDetector(path)


uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if st.button("🔍 Run Detection", use_container_width=True):
    if not uploaded:
        st.warning("Please upload an image first.")
    else:
        try:
            detector = load_detector(model_dir)
            image_bytes = uploaded.read()
            result = detector.predict(image_bytes)

            if result.get("accepted"):
                predicted_class = result['predicted_class'].upper()
                confidence = result['confidence']
                
                # Determine risk level
                if confidence >= 0.85:
                    risk_level = "🔴 HIGH CONFIDENCE"
                    risk_color = "red"
                elif confidence >= 0.70:
                    risk_level = "🟡 MODERATE CONFIDENCE"
                    risk_color = "orange"
                else:
                    risk_level = "🟢 LOW CONFIDENCE"
                    risk_color = "green"
                
                # Display image + results
                col_img, col_results = st.columns([1, 1.5])
                
                with col_img:
                    st.markdown("### 📷 Uploaded Image")
                    image_obj = Image.open(uploaded)
                    st.image(image_obj, use_column_width=True)
                    st.caption(f"File: {uploaded.name}")
                
                with col_results:
                    st.markdown("### 📋 Detection Results")
                    st.markdown(f"<h1 style='color: {risk_color}; text-align: center;'>{predicted_class}</h1>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence Score:**")
                    st.progress(confidence, text=f"{confidence:.1%}")
                    st.markdown(f"**Risk Level:** {risk_level}")
                    
                    disease_data = disease_info.get(predicted_class, {})
                    if disease_data:
                        st.info(f"**{disease_data['full_name']}**\n\n{disease_data['description']}")
                
                # Detailed analysis with tabs
                st.markdown("---")
                st.markdown("### 📊 Detailed Analysis")
                
                # Image quality assessment
                image_obj = Image.open(uploaded)
                quality_score, quality_feedback = assess_image_quality(image_obj, result['skin_ratio'])
                
                # Calculate comprehensive risk score
                disease_data = disease_info.get(predicted_class, {})
                risk_score = generate_risk_score(confidence, disease_data.get('severity', 'Low'))
                
                # Create tabs
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📖 Clinical Info", 
                    "📈 All Predictions", 
                    "💊 Recommendations", 
                    "⚠️ Risk Assessment",
                    "🔀 Condition Comparison",
                    "📋 Doctor Report"
                ])
                
                with tab1:
                    disease_data = disease_info.get(predicted_class, {})
                    if disease_data:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Disease Severity", disease_data.get('severity', 'N/A'))
                        with col_b:
                            st.metric("Clinical Urgency", disease_data.get('urgency', 'N/A'))
                        
                        st.markdown(f"### {disease_data['full_name']}")
                        st.markdown(disease_data.get('notes', ''))
                
                with tab2:
                    st.markdown("### Confidence for All 7 Conditions")
                    probs_dict = result.get("all_probabilities", {})
                    if probs_dict:
                        probs_list = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
                        
                        probs_df = pd.DataFrame(
                            [
                                {
                                    "#": i+1,
                                    "Condition": cls.upper(),
                                    "Confidence": f"{conf*100:.1f}%",
                                }
                                for i, (cls, conf) in enumerate(probs_list)
                            ]
                        )
                        st.dataframe(probs_df, use_container_width=True, hide_index=True)
                        
                        st.markdown("### Confidence Chart")
                        probs_chart_df = pd.DataFrame(probs_list, columns=["Condition", "Confidence"])
                        probs_chart_df["Condition"] = probs_chart_df["Condition"].str.upper()
                        st.bar_chart(probs_chart_df.set_index("Condition"))
                
                with tab3:
                    disease_data = disease_info.get(predicted_class, {})
                    st.markdown(f"### Recommended Actions")
                    
                    if confidence >= 0.85:
                        st.error("🚨 **HIGH CONFIDENCE DIAGNOSIS**")
                        st.markdown(f"""
                        #### Immediate Actions:
                        - **Schedule dermatologist appointment URGENTLY** (1-2 weeks)
                        - Bring this AI report to your appointment
                        - Consider getting a second opinion
                        - Take photos to track changes
                        - Avoid sun exposure to the area
                        
                        **Condition:** {disease_data.get('full_name', 'Unknown')}  
                        **Action Required:** {disease_data.get('urgency', 'Consult specialist')}
                        """)
                    elif confidence >= 0.70:
                        st.warning("⚠️ **MODERATE CONFIDENCE DIAGNOSIS**")
                        st.markdown(f"""
                        #### Recommended Actions:
                        - **Schedule dermatologist appointment** (2-4 weeks)
                        - This is suggestive, not definitive
                        - Professional examination/biopsy may be needed
                        - Monitor for changes (size, color, texture)
                        - Bring this report to your dermatologist
                        
                        **Condition:** {disease_data.get('full_name', 'Unknown')}  
                        **Action Required:** {disease_data.get('urgency', 'Consult specialist')}
                        """)
                    else:
                        st.info("ℹ️ **LOW CONFIDENCE DIAGNOSIS**")
                        st.markdown(f"""
                        #### Recommended Actions:
                        - **Consult dermatologist for definitive diagnosis**
                        - Result is inconclusive - don't rely solely on this
                        - Additional professional evaluation needed
                        - May require dermatoscopy or other tools
                        - Document changes over time
                        
                        **Possible Condition:** {disease_data.get('full_name', 'Multiple possible')}  
                        **Action Required:** {disease_data.get('urgency', 'Consult specialist')}
                        """)
                    
                    st.metric("Skin Content Detected", f"{result['skin_ratio']:.1%}")
                
                with tab4:
                    st.markdown("### ⚠️ Risk Assessment & Image Quality")
                    
                    # Risk Score
                    col_risk1, col_risk2 = st.columns(2)
                    with col_risk1:
                        st.markdown("**Overall Risk Score**")
                        st.metric("", f"{risk_score:.0f}/100", delta="Higher = More Concerning")
                        risk_desc = "🔴 HIGH RISK" if risk_score >= 70 else "🟡 MODERATE RISK" if risk_score >= 40 else "🟢 LOW RISK"
                        st.info(f"Classification: {risk_desc}")
                    
                    with col_risk2:
                        st.markdown("**Image Quality Score**")
                        st.metric("", f"{quality_score:.0f}/100")
                        st.caption("Assessment of upload quality")
                    
                    # Quality feedback
                    st.markdown("#### 📸 Image Quality Feedback")
                    for feedback in quality_feedback:
                        st.write(feedback)
                
                with tab5:
                    st.markdown("### 🔀 Condition Comparison")
                    st.markdown("Compare top detected conditions:")
                    
                    probs_dict = result.get("all_probabilities", {})
                    if probs_dict:
                        probs_list = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
                        
                        # Top 3 comparison
                        for rank, (condition, prob) in enumerate(probs_list[:3], 1):
                            cond_upper = condition.upper()
                            cond_data = disease_info.get(cond_upper, {})
                            
                            col_rank, col_info = st.columns([1, 3])
                            with col_rank:
                                if rank == 1:
                                    st.markdown(f"🥇 **#{rank}**")
                                elif rank == 2:
                                    st.markdown(f"🥈 **#{rank}**")
                                else:
                                    st.markdown(f"🥉 **#{rank}**")
                            
                            with col_info:
                                st.markdown(f"**{cond_upper}** ({prob:.1%})")
                                st.caption(cond_data.get('description', 'N/A'))
                            
                            st.progress(prob, text=f"{prob:.1%} confidence")
                            st.markdown("---")
                
                with tab6:
                    st.markdown("### 📋 Doctor-Friendly Report")
                    
                    # Generate report
                    report_text = generate_doctor_report(
                        uploaded.name,
                        predicted_class,
                        confidence,
                        disease_data,
                        result.get('all_probabilities', {}),
                        result['skin_ratio'],
                        st.session_state.get('fitzpatrick', 'III')
                    )
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        st.download_button(
                            label="📥 Download Report (TXT)",
                            data=report_text,
                            file_name=f"skin_cancer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with col_btn2:
                        # Copy to clipboard alternative
                        st.text_area(
                            "Report Preview (select and copy):",
                            value=report_text,
                            height=300,
                            disabled=True
                        )
                
                # Technical tab (moved to separate section for clarity)
                with st.expander("🔬 Technical Details"):
                    st.markdown("### Model Performance Metrics")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Model Confidence", f"{confidence:.1%}")
                    with col_m2:
                        st.metric("Skin Ratio", f"{result['skin_ratio']:.1%}")
                    with col_m3:
                        st.metric("OOD Score", f"{result.get('ood_score', 0):.3f}")
                    
                    st.markdown("### Raw JSON Data")
                    st.json(result)
                
            else:
                st.error(f"❌ Image Rejected")
                st.warning(f"**Reason:** {result.get('reason', 'Unknown').replace('_', ' ').title()}")
                
                if result.get('reason') == 'non_skin_rejected_by_skin_ratio':
                    st.markdown(f"""
                    **The image does not appear to be human skin.**
                    
                    - Detected: **{result.get('skin_ratio', 0):.1%}** skin
                    - Required: **{result.get('threshold', 0):.1%}** minimum
                    
                    ✅ **Fix:**
                    - Clear photo of affected area
                    - Natural lighting
                    - Skin fills most of frame
                    - Remove background/equipment
                    """)
                elif result.get('reason') == 'non_skin_or_ood_rejected':
                    st.markdown("""
                    **Image is unusual or unfamiliar to the model.**
                    
                    Possible causes:
                    - Unusual lighting/extreme angle
                    - Medical devices visible
                    - Low quality image
                    
                    ✅ **Fix:**
                    - Try natural lighting
                    - Photograph straight-on
                    - Ensure focus and clarity
                    - Remove equipment from frame
                    """)
                elif result.get('reason') == 'low_confidence_rejected':
                    st.markdown(f"""
                    **Model cannot make a confident classification.**
                    
                    - Detected confidence: **{result.get('confidence', 0):.1%}**
                    - Minimum required: **{result.get('threshold', 0):.1%}**
                    
                    ✅ **Fix:**
                    - Take clearer, closer photo
                    - Improve lighting
                    - Ensure lesion is in focus
                    - Try different angle
                    """)
                
                with st.expander("Technical details"):
                    st.json(result)
        except Exception as exc:
            st.exception(exc)


# Footer
st.markdown("---")
st.markdown("### ℹ️ Important Information")
st.markdown("""
**DISCLAIMER:** This AI tool is for informational purposes only. 
**NOT** a substitute for professional medical diagnosis. 
Always consult a qualified dermatologist for accurate diagnosis and treatment.

**7 Skin Conditions Detected:**
- **AKIEC**: Actinic Keratosis / Bowen's Disease (pre-cancerous)
- **BCC**: Basal Cell Carcinoma (common skin cancer)
- **BKL**: Benign Keratosis (harmless growth)
- **DF**: Dermatofibroma (harmless bump)
- **MEL**: Melanoma (dangerous skin cancer)
- **NV**: Nevus / Mole (common, usually harmless)
- **VASC**: Vascular Lesion (abnormal blood vessels)
""")

with st.expander("📚 How to use this tool"):
    st.markdown("""
    1. **Upload**: Click "Upload an image" and select a clear skin photo
    2. **Run Detection**: Click "🔍 Run Detection"
    3. **Review Results**:
       - See detected condition
       - Check confidence level
       - Read clinical recommendations
    4. **Consult Doctor**: Always follow up with dermatologist
    
    **Best Practices:**
    - Good natural lighting
    - Clear focus on lesion
    - Skin fills most of frame
    - No equipment visible
    - Photograph straight-on (not angled)
    """)

with st.expander("🔧 Technical Details"):
    st.markdown("""
    **Model Architecture:**
    - CNN: ResNet18 fine-tuned on HAM10000
    - Classifier: XGBoost on CNN embeddings
    - Safety: Skin detection + OOD detection
    
    **Training:**
    ```bash
    python train.py --data-dir ../sorted_by_dx --output-dir ./model_artifacts --epochs 20 --batch-size 32
    ```
    
    **CLI Inference:**
    ```bash
    python infer.py --model-dir ./model_artifacts --image image.jpg
    ```
    """)

