# Car Damage Detection App

This app lets you upload a car image and predicts the likely damage class.

## Model Details

1. Backbone: ResNet50 (transfer learning)
2. Target classes:
   1. Front Normal
   2. Front Crushed
   3. Front Breakage
   4. Rear Normal
   5. Rear Crushed
   6. Rear Breakage
3. Validation accuracy: ~80%

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

Note:

- Run `app.py` as the entrypoint (not `model_helper.py`).
- Ensure model checkpoint exists at `model/saved_model.pth`.
