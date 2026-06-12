import joblib

model = joblib.load("cardiopredict_model.pkl")

print(type(model))
print("Model loaded successfully!")
