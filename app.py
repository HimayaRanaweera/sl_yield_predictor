import gradio as gr
import joblib
import pandas as pd

# Load model
model = joblib.load("best_model_tuned.joblib")

def predict_yield(year, season, province, district, crop, soil_type, irrigation,
                  area_sown, area_harvested, rainfall, temperature, fertilizer):

    input_data = {
        "Year": year,
        "Season": season,
        "Province": province,
        "District": district,
        "Crop": crop,
        "Soil_Type": soil_type,
        "Irrigation": irrigation,
        "Area_Sown_ha": area_sown,
        "Area_Harvested_ha": area_harvested,
        "Rainfall_mm": rainfall,
        "Temperature_C": temperature,
        "Fertilizer_kg_per_ha": fertilizer
    }

    input_df = pd.DataFrame([input_data])

    try:
        pred_yield = model.predict(input_df)[0]
        production = pred_yield * area_harvested

        # Beautiful result format with Sri Lankan agriculture colors
        result = f"""
        <div style="background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
                    padding: 25px; border-radius: 15px; color: white; text-align: center;
                    border: 3px solid #FFD700; box-shadow: 0 8px 25px rgba(0,0,0,0.2);">
            <h2 style="margin: 0; font-size: 28px; color: #FFD700;">🌾 Prediction Results</h2>
            <div style="margin-top: 20px;">
                <p style="font-size: 22px; margin: 15px 0; background: rgba(255,255,255,0.2);
                         padding: 10px; border-radius: 8px;">
                    <strong>Yield:</strong> {pred_yield:.2f} mt/ha
                </p>
                <p style="font-size: 22px; margin: 15px 0; background: rgba(255,255,255,0.2);
                         padding: 10px; border-radius: 8px;">
                    <strong>Production:</strong> {production:.1f} metric tons
                </p>
            </div>
        </div>
        """
        return result
    except Exception as e:
        return f"<div style='color: red; padding: 20px; text-align: center; background: #FFE4E1; border-radius: 10px;'>Error: {str(e)}</div>"

# Custom CSS with Sri Lankan agriculture theme
custom_css = """
.gradio-container {
    background: url('https://images.unsplash.com/photo-1595974482597-4b8da8879bc5?w=1200&auto=format&fit=crop&q=80&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8cGFkZHklMjBmaWVsZHxlbnwwfHwwfHx8MA%3D%3D') no-repeat center center fixed;
    background-size: cover;
}
.main-container {
    background: rgba(255, 255, 255, 0.92) !important;
    backdrop-filter: blur(8px);
    border-radius: 20px;
    padding: 30px;
    margin: 20px;
    border: 2px solid #2E8B57;
}
"""

# Create interface with beautiful styling
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald")) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #2E8B57, #3CB371);
                    border-radius: 15px; color: white; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 36px;">🌾 Sri Lanka Crop Yield Predictor</h1>
            <p style="margin: 10px 0 0 0; font-size: 18px; opacity: 0.9;">
                Predict expected yield before the season ends and estimate production
            </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📋 Enter Farm Details")
            year = gr.Number(label="📅 Year", minimum=2010, maximum=2030)
            season = gr.Dropdown(["Maha", "Yala"], label="🌤️ Season")
            province = gr.Dropdown(["Western","Central","Southern","Northern","Eastern","North Western","North Central","Uva","Sabaragamuwa"],
                                 label="🗺️ Province")
            district = gr.Textbox(label="🏘️ District")
            crop = gr.Dropdown(["Paddy","Maize","Vegetables","Onion","Chili","Coconut","Rubber","Tea"],
                             label="🌱 Crop")
            soil_type = gr.Dropdown(["Clay","Sandy","Loamy","Laterite","Alluvial"],
                                  label="🌍 Soil Type")
            irrigation = gr.Dropdown(["Irrigated","Rainfed"], label="💧 Irrigation")

        with gr.Column():
            gr.Markdown("### 🌿 Growing Conditions")
            area_sown = gr.Number(label="📏 Area Sown (ha)")
            area_harvested = gr.Number(label="📏 Area Harvested (ha)")
            rainfall = gr.Number(label="🌧️ Rainfall (mm)")
            temperature = gr.Number(label="🌡️ Temperature (°C)")
            fertilizer = gr.Number(label="🧪 Fertilizer (kg/ha)")

            predict_btn = gr.Button("🚀 Predict Yield", variant="primary", size="lg",
                                  elem_id="predict-btn")

    # Result section that pops up when clicked
    with gr.Row(visible=False) as result_row:
        with gr.Column():
            gr.Markdown("### 📊 Prediction Results")
            output = gr.HTML()

    # Function to show result section
    def show_results(year, season, province, district, crop, soil_type, irrigation,
                    area_sown, area_harvested, rainfall, temperature, fertilizer):
        result = predict_yield(year, season, province, district, crop, soil_type, irrigation,
                              area_sown, area_harvested, rainfall, temperature, fertilizer)
        return {result_row: gr.update(visible=True), output: result}

    predict_btn.click(
        fn=show_results,
        inputs=[year, season, province, district, crop, soil_type, irrigation,
                area_sown, area_harvested, rainfall, temperature, fertilizer],
        outputs=[result_row, output]
    )

if __name__ == "__main__":
    demo.launch()
