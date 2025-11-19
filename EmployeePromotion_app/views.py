from django.shortcuts import render
import joblib
import pandas as pd

def homepage(request):
    return render(request, 'EmployeePromotion_app/Homepage.html')



def predict(request):
    if request.method == "POST":
        import joblib
        import pandas as pd

        # --- Correct categorical mappings ---
        department_mapping = {
            'Sales & Marketing': 0, 'Operations': 1, 'Technology': 2,
            'Analytics': 3, 'R&D': 4, 'Procurement': 5,
            'Finance': 6, 'HR': 7, 'Legal': 8
        }

        region_mapping = {
            'region_7': 0, 'region_22': 1, 'region_19': 2, 'region_23': 3, 'region_26': 4,
            'region_2': 5, 'region_20': 6, 'region_34': 7, 'region_1': 8, 'region_4': 9,
            'region_29': 10, 'region_31': 11, 'region_15': 12, 'region_14': 13,
            'region_11': 14, 'region_5': 15, 'region_28': 16, 'region_17': 17,
            'region_13': 18, 'region_16': 19, 'region_25': 20, 'region_10': 21,
            'region_27': 22, 'region_30': 23, 'region_12': 24, 'region_21': 25,
            'region_8': 26, 'region_32': 27, 'region_6': 28, 'region_33': 29,
            'region_24': 30, 'region_3': 31, 'region_9': 32, 'region_18': 33
        }

        education_mapping = {
            "Master's & above": 0, "Bachelor's": 1, 'Below Secondary': 2
        }

        gender_mapping = {'f': 0, 'm': 1}

        recruitment_channel_mapping = {
            'sourcing': 0, 'other': 1, 'referred': 2
        }

        # --- Get values from POST ---
        department = department_mapping.get(request.POST['department'], 0)
        region = region_mapping.get(request.POST['region'], 0)
        education = education_mapping.get(request.POST['education'], 0)
        gender = gender_mapping.get(request.POST['gender'], 0)
        recruitment_channel = recruitment_channel_mapping.get(request.POST['recruitment_channel'], 0)

        no_of_trainings = int(request.POST['no_of_trainings'])
        age = int(request.POST['age'])
        previous_year_rating = int(request.POST['previous_year_rating'])
        length_of_service = int(request.POST['length_of_service'])
        awards_won = int(request.POST['awards_won'])
        avg_training_score = float(request.POST['avg_training_score'])

        # --- Load ONLY XGBoost model ---
        model = joblib.load("ml_models/xgboost_model.pkl")

        # --- Prepare dataframe ---
        columns = [
            'department', 'region', 'education', 'gender', 'recruitment_channel',
            'no_of_trainings', 'age', 'previous_year_rating',
            'length_of_service', 'awards_won', 'avg_training_score'
        ]

        features = pd.DataFrame([[department, region, education, gender, recruitment_channel,
                                  no_of_trainings, age, previous_year_rating, length_of_service,
                                  awards_won, avg_training_score]], columns=columns)

        # --- Predict ---
        prediction = model.predict(features)[0]

        # Decide result
        result = "Eligible for Promotion" if prediction == 1 else "Not Eligible for Promotion"

        return render(request, 'EmployeePromotion_app/prediction.html', {
            'prediction': result
        })

    return render(request, 'EmployeePromotion_app/prediction.html')
