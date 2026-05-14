import streamlit as st
import requests

st.set_page_config(page_title="LabTest AI Analyzer", page_icon="🧬")

st.title("🧬 LabTest AI Analyzer")
st.markdown("Upload your lab report (PDF or Image) for agentic analysis.")

# File uploader - supports types defined in your workflow [cite: 1]
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display a preview
    if uploaded_file.type.startswith("image"):
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    else:
        st.info("PDF uploaded successfully.")

    if st.button("Analyze Report"):
        with st.spinner("Agentic pipeline running..."):
            # Prepare the file for the multipart request
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                # Routes to your FastAPI endpoint
                response = requests.post("http://127.0.0.1:8000/analyze", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("status") == "failed":
                        st.error(f"Analysis Failed: {result.get('reason')}")
                    else:
                        st.success("Analysis Complete!")
                        
                        # Layout for Positives/Negatives as per flowchart 
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("✅ Positives")
                            st.write(result.get("structured_data", {}).get("positives", "None found"))
                        
                        with col2:
                            st.subheader("⚠️ Negatives")
                            st.write(result.get("structured_data", {}).get("negatives", "None found"))
                        
                        st.divider()
                        st.subheader("📖 Medical Insight")
                        st.write(result.get("analysis", "No detailed analysis available yet."))
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI server. Make sure main.py is running on port 8000.")