def predict_message(message):
    # Preprocess the message
    sequence = tokenizer.texts_to_sequences([message])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    
    # Predict the class probability
    prob = model.predict(padded_sequence)[0][0]
    
    # Determine the class label
    label = "ham" if prob < 0.5 else "spam"
    
    return [prob, label]
