import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
model = load_model('model3.h5')

   
    
tokenizer = pickle.load(open('tokenizer.pkl','rb'))
b = ["this movie was too bad i hate it "]

a_sequence = tokenizer.texts_to_sequences(b)
    # print(a_sequence)
a_padding = pad_sequences(a_sequence,maxlen=120,padding='post',truncating='post')
x = model.predict(a_padding)
print(x)
if(x>0.5):
    print("positive")
else :
    print("neagtive")
