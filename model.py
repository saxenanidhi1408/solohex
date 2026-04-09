import pickle
import re

model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

fake_words = ["click","free","win","shocking","miracle","earn","offer"]
real_words = ["government","policy","india","court","election","official"]

def clean_text(text):
    return re.sub(r'[^a-zA-Z ]', '', text.lower())

def predict_news(text):
    text_clean = clean_text(text)
    vec = vectorizer.transform([text_clean])

    decision = model.decision_function(vec)[0]
    confidence = int(min(abs(decision)*20,100))
    prediction = model.predict(vec)[0]

    text_lower = text.lower()
    wc = len(text.split())

    if wc < 3:
        return {"label":"Uncertain","confidence":0,"class":"fake",
                "fake_score":50,"real_score":50,
                "reasons":["Too short"],"message":"Enter proper text"}

    fake_c = sum(w in text_lower for w in fake_words)
    real_c = sum(w in text_lower for w in real_words)

    ml = "Real" if prediction==1 else "Fake"

    if fake_c > real_c:
        final = "Fake"
    elif real_c > fake_c:
        final = "Real"
    else:
        final = ml

    if confidence < 40:
        final = "Uncertain"

    if final == "Fake":
        fs = max(confidence,60); rs = 100-fs
    elif final == "Real":
        rs = max(confidence,60); fs = 100-rs
    else:
        fs = rs = 50

    reasons=[]
    if fake_c>0: reasons.append("Suspicious words found")
    if real_c>0: reasons.append("News-related words found")
    if not reasons: reasons.append("ML prediction")

    return {
        "label":final,
        "confidence":confidence,
        "class":"real" if final=="Real" else "fake",
        "fake_score":fs,
        "real_score":rs,
        "reasons":reasons,
        "message":f"Result: {final}"
    }