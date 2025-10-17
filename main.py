from load_data import load_and_show_samples
from preprocess import normalize_data
from model_simple import make_simple_cnn
from train_model import train_model
from utils_plot import plot_history
from evaluate import evaluate_and_predict

(x_train, y_train), (x_test, y_test) = load_and_show_samples()
x_train, x_test = normalize_data(x_train, x_test)
model = make_simple_cnn()
history = train_model(model, x_train, y_train)
plot_history(history)
evaluate_and_predict(model, x_test, y_test)0