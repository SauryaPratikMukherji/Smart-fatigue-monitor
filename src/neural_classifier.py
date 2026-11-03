import os
import json
import numpy as np
from typing import Tuple, Dict

MODEL_CLASSES = ["Focused", "Distracted", "Drowsy"]


# softmax function
def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


# relu activation
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


# 3-layer neural network classifier
class AttentionNeuralClassifier:
    def __init__(self, weights_path: str = "models/attention_weights.npz"):
        self.weights_path = weights_path
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.w3 = None
        self.b3 = None
        self._initialize_network()

    # load weights from file or train if missing
    def _initialize_network(self):
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)

        if os.path.exists(self.weights_path):
            try:
                data = np.load(self.weights_path)
                self.w1 = data['w1']
                self.b1 = data['b1']
                self.w2 = data['w2']
                self.b2 = data['b2']
                self.w3 = data['w3']
                self.b3 = data['b3']
                return
            except Exception:
                pass

        self._train_and_save_weights()

    # train model weights
    def _train_and_save_weights(self):
        np.random.seed(42)
        n_per_class = 2000

        X_list = []
        y_list = []

        # 1. focused samples
        for _ in range(n_per_class):
            ear_l = np.random.uniform(0.28, 0.38)
            ear_r = ear_l + np.random.uniform(-0.02, 0.02)
            ear_avg = (ear_l + ear_r) / 2.0
            mar = np.random.uniform(0.15, 0.38)
            yaw = np.random.uniform(-10.0, 10.0) / 90.0
            pitch = np.random.uniform(-8.0, 10.0) / 90.0
            roll = np.random.uniform(-8.0, 8.0) / 90.0
            bpm = np.random.uniform(10.0, 22.0) / 60.0
            closure = np.random.uniform(0.0, 0.15)
            yawn = 0.0
            X_list.append([ear_l, ear_r, ear_avg, mar, yaw, pitch, roll, bpm, closure, yawn])
            y_list.append([1.0, 0.0, 0.0])

        # 2. distracted samples
        for _ in range(n_per_class):
            ear_l = np.random.uniform(0.26, 0.38)
            ear_r = ear_l + np.random.uniform(-0.02, 0.02)
            ear_avg = (ear_l + ear_r) / 2.0
            mar = np.random.uniform(0.15, 0.40)
            yaw = np.random.choice([np.random.uniform(25.0, 60.0), np.random.uniform(-60.0, -25.0)]) / 90.0
            pitch = np.random.uniform(-30.0, 30.0) / 90.0
            roll = np.random.uniform(-15.0, 15.0) / 90.0
            bpm = np.random.uniform(5.0, 35.0) / 60.0
            closure = np.random.uniform(0.0, 0.20)
            yawn = 0.0
            X_list.append([ear_l, ear_r, ear_avg, mar, yaw, pitch, roll, bpm, closure, yawn])
            y_list.append([0.0, 1.0, 0.0])

        # 3. drowsy samples
        for _ in range(n_per_class):
            is_yawn = np.random.choice([True, False])
            if is_yawn:
                ear_l = np.random.uniform(0.20, 0.30)
                mar = np.random.uniform(0.58, 0.90)
                yawn = np.random.uniform(0.7, 1.0)
                closure = np.random.uniform(0.2, 0.5)
            else:
                ear_l = np.random.uniform(0.08, 0.21)
                mar = np.random.uniform(0.20, 0.45)
                yawn = 0.0
                closure = np.random.uniform(0.75, 1.0)

            ear_r = ear_l + np.random.uniform(-0.02, 0.02)
            ear_avg = (ear_l + ear_r) / 2.0
            yaw = np.random.uniform(-15.0, 15.0) / 90.0
            pitch = np.random.uniform(-40.0, -10.0) / 90.0
            roll = np.random.uniform(-10.0, 10.0) / 90.0
            bpm = np.random.uniform(2.0, 12.0) / 60.0
            X_list.append([ear_l, ear_r, ear_avg, mar, yaw, pitch, roll, bpm, closure, yawn])
            y_list.append([0.0, 0.0, 1.0])

        X = np.array(X_list, dtype=np.float32)
        Y = np.array(y_list, dtype=np.float32)

        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

        # layers: 10 -> 32 -> 16 -> 3
        d_in = 10
        d_h1 = 32
        d_h2 = 16
        d_out = 3

        # initialize weights
        w1 = np.random.randn(d_in, d_h1).astype(np.float32) * np.sqrt(2.0 / d_in)
        b1 = np.zeros(d_h1, dtype=np.float32)
        w2 = np.random.randn(d_h1, d_h2).astype(np.float32) * np.sqrt(2.0 / d_h1)
        b2 = np.zeros(d_h2, dtype=np.float32)
        w3 = np.random.randn(d_h2, d_out).astype(np.float32) * np.sqrt(2.0 / d_h2)
        b3 = np.zeros(d_out, dtype=np.float32)

        lr = 0.01
        batch_size = 64
        epochs = 25
        n_batches = len(X) // batch_size

        v_w1, v_b1 = np.zeros_like(w1), np.zeros_like(b1)
        v_w2, v_b2 = np.zeros_like(w2), np.zeros_like(b2)
        v_w3, v_b3 = np.zeros_like(w3), np.zeros_like(b3)
        gamma = 0.9

        # training loop
        for _ in range(epochs):
            for b in range(n_batches):
                start = b * batch_size
                end = start + batch_size
                xb = X[start:end]
                yb = Y[start:end]

                # forward pass
                z1 = np.dot(xb, w1) + b1
                a1 = relu(z1)
                z2 = np.dot(a1, w2) + b2
                a2 = relu(z2)
                z3 = np.dot(a2, w3) + b3
                a3 = softmax(z3)

                # backward pass
                dz3 = (a3 - yb) / batch_size
                dw3 = np.dot(a2.T, dz3)
                db3 = np.sum(dz3, axis=0)

                da2 = np.dot(dz3, w3.T)
                dz2 = da2 * (z2 > 0)
                dw2 = np.dot(a1.T, dz2)
                db2 = np.sum(dz2, axis=0)

                da1 = np.dot(dz2, w2.T)
                dz1 = da1 * (z1 > 0)
                dw1 = np.dot(xb.T, dz1)
                db1 = np.sum(dz1, axis=0)

                # update weights with momentum
                v_w3 = gamma * v_w3 + lr * dw3
                v_b3 = gamma * v_b3 + lr * db3
                v_w2 = gamma * v_w2 + lr * dw2
                v_b2 = gamma * v_b2 + lr * db2
                v_w1 = gamma * v_w1 + lr * dw1
                v_b1 = gamma * v_b1 + lr * db1

                w3 -= v_w3
                b3 -= v_b3
                w2 -= v_w2
                b2 -= v_b2
                w1 -= v_w1
                b1 -= v_b1

        # save trained weights
        np.savez_compressed(
            self.weights_path,
            w1=w1, b1=b1,
            w2=w2, b2=b2,
            w3=w3, b3=b3
        )

        self.w1, self.b1 = w1, b1
        self.w2, self.b2 = w2, b2
        self.w3, self.b3 = w3, b3

    # predict state from features
    def predict(
        self,
        ear_l: float,
        ear_r: float,
        ear_avg: float,
        mar: float,
        yaw: float,
        pitch: float,
        roll: float,
        bpm: float,
        closure_ratio: float,
        yawn_ratio: float
    ) -> Tuple[str, float, Dict[str, float]]:
        x = np.array([
            ear_l, ear_r, ear_avg, mar,
            yaw / 90.0, pitch / 90.0, roll / 90.0,
            bpm / 60.0,
            min(1.0, max(0.0, closure_ratio)),
            min(1.0, max(0.0, yawn_ratio))
        ], dtype=np.float32)

        if self.w1 is not None:
            # forward pass
            z1 = np.dot(x, self.w1) + self.b1
            a1 = relu(z1)
            z2 = np.dot(a1, self.w2) + self.b2
            a2 = relu(z2)
            z3 = np.dot(a2, self.w3) + self.b3
            probs = softmax(z3)

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            prob_dict = {
                MODEL_CLASSES[i]: float(probs[i])
                for i in range(len(MODEL_CLASSES))
            }
            return MODEL_CLASSES[pred_idx], confidence, prob_dict

        # fallback check
        if closure_ratio >= 0.7 or yawn_ratio >= 0.7 or ear_avg < 0.20:
            pred = "Drowsy"
            conf = 0.90
        elif abs(yaw) > 25.0 or pitch < -18.0 or pitch > 20.0:
            pred = "Distracted"
            conf = 0.85
        else:
            pred = "Focused"
            conf = 0.95

        return pred, conf, {
            "Focused": 0.9 if pred == "Focused" else 0.05,
            "Distracted": 0.9 if pred == "Distracted" else 0.05,
            "Drowsy": 0.9 if pred == "Drowsy" else 0.05
        }
