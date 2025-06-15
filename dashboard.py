print("📢 Starting dashboard.py...")

try:
    import joblib
    import pandas as pd
    from explainerdashboard import ClassifierExplainer, ExplainerDashboard
    from preprocess import load_data, preprocess_data

    print("🔄 Loading dataset...")
    df = load_data('loan_dataset.csv')

    print("✅ Preprocessing...")
    X, y = preprocess_data(df)

    print("📦 Loading model...")
    model = joblib.load('loan_model.pkl')

    print("🧠 Creating Explainer...")
    explainer = ClassifierExplainer(model, X, y, labels=["Rejected", "Approved"])

    print("🚀 Launching dashboard on http://localhost:8050")
    ExplainerDashboard(explainer).run()

except Exception as e:
    print("❌ CRASHED with error:", e)
