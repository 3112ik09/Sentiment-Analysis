from flask import Flask , render_template ,request 
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask('__name__')
model = load_model('model3.h5')
tokenizer = pickle.load(open('tokenizer.pkl','rb'))

def predict():
   
    text = list(request.form.values())
    a_sequence = tokenizer.texts_to_sequences(text)
    a_padding = pad_sequences(a_sequence,maxlen=120,padding='post',truncating='post')
    output = model.predict(a_padding)



    return output


@app.route('/')
def start():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def my_form_post():
    text1 = list(request.form.values())
    output = predict()
    result = ""
    if output>0.5:
        result="Positive"
    else :
        result ='Negative'

    return render_template('index.html',final=output,text1=text1,result=result)


if __name__ == '__main__':
    app.run(debug=True)