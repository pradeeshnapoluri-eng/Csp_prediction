from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open("csp_models/csp_best_model.pkl", "rb") as f:
    model = pickle.load(f)

ticket_type_map = {
    "Billing inquiry": 0,
    "Cancellation request": 1,
    "Product inquiry": 2,
    "Refund request": 3,
    "Technical issue": 4
}

ticket_subject_map = {
    "Battery life": 0,
    "Cancellation request": 1,
    "Data loss": 2,
    "Delivery problem": 3,
    "Hardware issue": 4,
    "Installation support": 5,
    "Network problem": 6,
    "Payment issue": 7,
    "Product compatibility": 8,
    "Product setup": 9,
    "Refund request": 10,
    "Software bug": 11
}

ticket_status_map = {
    "Closed": 0,
    "Open": 1,
    "Pending Customer Response": 2
}

ticket_priority_map = {
    "Critical": 0,
    "High": 1,
    "Low": 2,
    "Medium": 3
}

ticket_channel_map = {
    "Chat": 0,
    "Email": 1,
    "Phone": 2,
    "Social media": 3
}

@app.route("/")
def home():
    return render_template("csp_index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        customer_age      = int(data["customer_age"])
        ticket_type       = ticket_type_map[data["ticket_type"]]
        ticket_subject    = ticket_subject_map[data["ticket_subject"]]
        ticket_status     = ticket_status_map[data["ticket_status"]]
        ticket_priority   = ticket_priority_map[data["ticket_priority"]]
        ticket_channel    = ticket_channel_map[data["ticket_channel"]]
        response_duration = float(data["response_duration"])
        purchase_year     = int(data["purchase_year"])
        purchase_month    = int(data["purchase_month"])

        age_group       = 1 if customer_age <= 25 else 2 if customer_age <= 35 else 3 if customer_age <= 50 else 4 if customer_age <= 65 else 5
        duration_bucket = min(int(response_duration / 5) + 1, 5)
        purchase_recency = 2024 - purchase_year

        # Match EXACT feature names from training
        df = pd.DataFrame([{
            'Ticket Type':            ticket_type,
            'Ticket Subject':         ticket_subject,
            'Ticket Channel':         ticket_channel,
            'Customer Age':           customer_age,
            'Purchase Month':         purchase_month,
            'Customer Gender':        1,
            'Duration Bucket':        duration_bucket,
            'Purchase Recency':       purchase_recency,
            'Purchase Year':          purchase_year,
            'Ticket Priority':        ticket_priority,
            'Ticket Status':          ticket_status,
            'Age Group':              age_group,
            'Response Duration (hrs)':response_duration,
        }])

        # Load feature list from training
        import os
        if os.path.exists("csp_features_final.csv"):
            train_cols = pd.read_csv("csp_features_final.csv").drop(
    columns=['Satisfaction Rating', 'Customer Satisfaction Rating'], errors='ignore').columns.tolist()
            df = df.reindex(columns=train_cols, fill_value=0)

        prediction = model.predict(df)[0]

        messages = {1:"Very Dissatisfied",2:"Dissatisfied",3:"Neutral",4:"Satisfied",5:"Very Satisfied"}

        return jsonify({"prediction":int(prediction),"message":messages.get(int(prediction),"Unknown"),"status":"success"})

    except Exception as e:
        return jsonify({"status":"error","message":str(e)})
if __name__ == "__main__":
    app.run(debug=True)