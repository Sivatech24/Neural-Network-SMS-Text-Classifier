# Load the dataset
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Preprocessing
# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['message'])
X_train = tokenizer.texts_to_sequences(train_data['message'])
X_test = tokenizer.texts_to_sequences(test_data['message'])

# Padding
max_len = max(len(seq) for seq in X_train)
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

# Encoding labels
y_train = train_data['label'].map({'ham': 0, 'spam': 1}).values
y_test = test_data['label'].map({'ham': 0, 'spam': 1}).values
