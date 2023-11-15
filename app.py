from flask import Flask, render_template, request
from src.pipelines.prediction_pipeline import CustomData, PredictionPipeline


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict", methods = ['GET','POST'])
def predict_datapoint():
    if request.method=='POST':

        data = CustomData(
                mean_radius=float(request.form.get("mean_radius")),
                mean_texture=float(request.form.get("mean_texture")),
                mean_smoothness=float(request.form.get("mean_smoothness")),
                mean_compactness=float(request.form.get("mean_compactness")),
                mean_concavity=float(request.form.get("mean_concavity")),
                mean_symmetry=float(request.form.get("mean_symmetry")),
                mean_fractal_dimension=float(request.form.get("mean_fractal_dimension")),
                radius_error=float(request.form.get("radius_error")),
                texture_error=float(request.form.get("texture_error")),
                smoothness_error=float(request.form.get("smoothness_error")),
                compactness_error=float(request.form.get("compactness_error")),
                concavity_error=float(request.form.get("concavity_error")),
                concave_points_error=float(request.form.get("concave_points_error")),
                symmetry_error=float(request.form.get("symmetry_error")),
                fractal_dimension_error=float(request.form.get("fractal_dimension_error")),
                worst_smoothness=float(request.form.get("worst_smoothness")),
                worst_compactness=float(request.form.get("worst_compactness")),
                worst_concavity=float(request.form.get("worst_concavity")),
                worst_symmetry=float(request.form.get("worst_symmetry")),
                worst_fractal_dimension=float(request.form.get("worst_fractal_dimension"))
                )
        new_Data = data.get_data_as_dataframe() 

        prediction_pipeline = PredictionPipeline()
        pred = prediction_pipeline.initiate_prediction(new_Data)
        result = pred[0]

        if result == 1:
            output = "Malignant Tumor, Need Treatment"
        else:
            output = "Benign Tumor, No Need of Treatment"
        
        return render_template("form.html", final_result = output)
    else:
        return render_template("form.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)