import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, input_data, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, input_data.shape) / (1 - self.dropout_rate)
            return input_data * self.mask
        return input_data

    def backward(self, grad_output):
        return grad_output * self.mask


class Layer:
    def __init__(self, input_size, output_size, activation='relu', l2_lambda=0.005, dropout_rate=0.2):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        self.activation = activation
        self.l2_lambda = l2_lambda
        self.dropout = Dropout(dropout_rate)
        self.input = None
        self.output = None
        self.z = None

    def get_weights(self):
        return {
            'weights': self.weights.copy(),
            'bias': self.bias.copy()
        }

    def set_weights(self, weights):
        self.weights = weights['weights'].copy()
        self.bias = weights['bias'].copy()

    def forward(self, input_data, training=True):
        self.input = input_data
        self.z = np.dot(input_data, self.weights) + self.bias

        if self.activation == 'relu':
            self.output = np.maximum(0, self.z)
        elif self.activation == 'softmax':
            exp_values = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
            self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        if self.activation != 'softmax':
            self.output = self.dropout.forward(self.output, training)

        return self.output

    def backward(self, grad_output, learning_rate):
        if self.activation != 'softmax':
            grad_output = self.dropout.backward(grad_output)

        if self.activation == 'relu':
            grad_z = grad_output * (self.z > 0)
        elif self.activation == 'softmax':
            grad_z = grad_output

        grad_weights = np.dot(self.input.T, grad_z) + self.l2_lambda * self.weights
        grad_bias = np.sum(grad_z, axis=0, keepdims=True)
        grad_input = np.dot(grad_z, self.weights.T)

        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input


class NeuralNetwork:
    def __init__(self, input_dim, output_dim, l2_lambda=0.005):
        self.layers = [
            Layer(input_dim, 128, 'relu', l2_lambda, dropout_rate=0.1),
            Layer(128, 64, 'relu', l2_lambda, dropout_rate=0.1),
            Layer(64, output_dim, 'softmax', l2_lambda, dropout_rate=0.0)
        ]

    def get_weights(self):
        return [layer.get_weights() for layer in self.layers]

    def set_weights(self, weights):
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)

    def forward(self, X, training=True):
        current_output = X
        for layer in self.layers:
            current_output = layer.forward(current_output, training)
        return current_output

    def backward(self, X, y, learning_rate):
        grad_output = self.layers[-1].output - y
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)

    def train_step(self, X_batch, y_batch, learning_rate):
        predictions = self.forward(X_batch, training=True)
        self.backward(X_batch, y_batch, learning_rate)
        loss = self.compute_loss(predictions, y_batch)
        return loss, predictions

    def predict(self, X):
        return self.forward(X, training=False)

    def compute_loss(self, predictions, y_true):
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(predictions), axis=1))

    def compute_accuracy(self, predictions, y_true):
        return np.mean(np.argmax(predictions, axis=1) == np.argmax(y_true, axis=1))


def plot_training_history(histories, n_splits):
    plt.figure(figsize=(15, 10))

    # Plot training loss
    plt.subplot(2, 1, 1)
    for fold in range(n_splits):
        plt.plot(histories[fold]['train_loss'], label=f'Fold {fold + 1}')
    plt.title('Training Loss per Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot validation accuracy
    plt.subplot(2, 1, 2)
    for fold in range(n_splits):
        plt.plot(histories[fold]['val_acc'], label=f'Fold {fold + 1}')
    plt.title('Validation Accuracy per Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def train_and_evaluate_model(X, y, n_splits=6, epochs=1000, batch_size=32, learning_rate=0.0001, patience=100):
    y_encoded = y.values
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []
    histories = []

    n_classes = y_encoded.shape[1]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, np.argmax(y_encoded, axis=1)), 1):
        print(f"\nFold {fold}/{n_splits}")

        X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = NeuralNetwork(X_train.shape[1], n_classes, 0.001)

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        best_val_accuracy = 0
        best_weights = None
        best_epoch = 0
        patience_counter = 0

        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train_scaled))
            X_train_shuffled = X_train_scaled[indices]
            y_train_shuffled = y_train[indices]

            epoch_train_loss = 0
            epoch_train_acc = 0
            n_batches = 0

            for i in range(0, len(X_train_scaled), batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]

                loss, predictions = model.train_step(X_batch, y_batch, learning_rate)
                accuracy = model.compute_accuracy(predictions, y_batch)

                epoch_train_loss += loss
                epoch_train_acc += accuracy
                n_batches += 1

            avg_train_loss = epoch_train_loss / n_batches
            avg_train_acc = epoch_train_acc / n_batches

            val_predictions = model.predict(X_val_scaled)
            val_loss = model.compute_loss(val_predictions, y_val)
            val_accuracy = model.compute_accuracy(val_predictions, y_val)

            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(avg_train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_accuracy)

            # Early stopping logic
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_weights = model.get_weights()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"No improvement in validation accuracy for {patience} epochs")
                break

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Restore best weights for this fold
        model.set_weights(best_weights)
        histories.append(history)

        fold_results.append({
            'fold': fold,
            'best_val_accuracy': best_val_accuracy,
            'best_epoch': best_epoch,
            'model': model
        })

        print(f"\nFold {fold} Best Results:")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

    # Find the best overall model
    best_fold = max(fold_results, key=lambda x: x['best_val_accuracy'])

    print("\nOverall Cross-validation results:")
    print(f"Mean validation accuracy: {np.mean([r['best_val_accuracy'] for r in fold_results]):.4f}")
    print(f"Standard deviation: {np.std([r['best_val_accuracy'] for r in fold_results]):.4f}")
    print(f"\nBest model from fold {best_fold['fold']} at epoch {best_fold['best_epoch']}")
    print(f"Best validation accuracy: {best_fold['best_val_accuracy']:.4f}")

    # Plot training history
    plot_training_history(histories, n_splits)

    return best_fold['model'], fold_results, histories


def prepare_data(data):
    race_columns = [col for col in data.columns if col.startswith('Race_')]
    X = data.drop(columns=race_columns)
    y = data[race_columns]
    return X, y


def train(data):
    X, y = prepare_data(data)
    best_model, fold_results, histories = train_and_evaluate_model(X, y)
    return best_model, fold_results, histories
