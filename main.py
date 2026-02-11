import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import numpy as np
import os, cv2


# =============================
# Helper Function
# =============================
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):

    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

    for i in range(min(len(images), n_row * n_col)):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=10)
        plt.xticks(())
        plt.yticks(())


# =============================
# Dataset Loading (Training Faces)
# =============================
dir_name = "dataset/faces/"

y = []
x = []
person_id = 0
h = w = 300
class_names = []
n_samples = 0

for person_name in os.listdir(dir_name):

    dir_path = dir_name + person_name + "/"
    class_names.append(person_name)

    for image_name in os.listdir(dir_path):

        image_path = dir_path + image_name

        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (h, w))

        x.append(resized.flatten())
        y.append(person_id)
        n_samples += 1

    person_id += 1

x = np.array(x)
y = np.array(y)
x = x / 255.0


print("Dataset Loaded")
print("Samples:", n_samples)
print("Classes:", len(class_names))


# =============================
# Load Imposters
# =============================
imposter_dir = "dataset/imposters/"
imposter_x = []

if os.path.exists(imposter_dir):

    for img_name in os.listdir(imposter_dir):

        img_path = imposter_dir + img_name
        img = cv2.imread(img_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (h, w))

        imposter_x.append(resized.flatten())

imposter_x = np.array(imposter_x)
imposter_x = imposter_x / 255.0

print("Imposter Samples:", imposter_x.shape)

# =============================
# Show Imposter Images
# =============================

print("\nShowing Imposter Images...")

if imposter_x is None or len(imposter_x) == 0:
    print("No imposter images to display")
else:

    n_show = min(12, len(imposter_x))  # show max 12 images

    plt.figure(figsize=(8, 6))

    for i in range(n_show):
        plt.subplot(3, 4, i+1)
        plt.imshow(imposter_x[i].reshape(h, w), cmap='gray')
        plt.title(f"Imposter {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# =============================
# Train Test Split
# =============================
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.40, random_state=42
)


# =============================
# Accuracy vs K Experiment
# =============================
k_values = [20, 50, 80, 100, 120, 150]
accuracies = []

best_model = None
best_pca = None
best_lda = None
best_acc = 0

for k in k_values:

    print("\nTesting k =", k)

    # PCA
    pca = PCA(n_components=k, svd_solver='randomized', whiten=True)
    pca.fit(x_train)

    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    # LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train_pca, y_train)

    x_train_lda = lda.transform(x_train_pca)
    x_test_lda = lda.transform(x_test_pca)

    # ANN
    clf = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000)
    clf.fit(x_train_lda, y_train)

    acc = clf.score(x_test_lda, y_test) * 100
    accuracies.append(acc)

    print("Accuracy:", acc)

    if acc > best_acc:
        best_acc = acc
        best_model = clf
        best_pca = pca
        best_lda = lda


# =============================
# Plot Accuracy vs K Graph
# =============================
plt.figure()
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("PCA Components (k)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs PCA Components")
plt.show()


# =============================
# Show Eigenfaces (Best PCA)
# =============================
eigenfaces = best_pca.components_.reshape((-1, h, w))
titles = ["Eigenface %d" % i for i in range(len(eigenfaces))]
plot_gallery(eigenfaces, titles, h, w)
plt.show()


# =============================
# Final Prediction (Best Model)
# =============================
x_test_pca = best_pca.transform(x_test)
x_test_lda = best_lda.transform(x_test_pca)

y_pred = best_model.predict(x_test_lda)
y_prob = np.max(best_model.predict_proba(x_test_lda), axis=1)

prediction_titles = []
true_positive = 0

for i in range(len(y_pred)):

    true_name = class_names[y_test[i]]
    pred_name = class_names[y_pred[i]]

    title = f"Pred: {pred_name} ({str(y_prob[i])[:4]})\nTrue: {true_name}"
    prediction_titles.append(title)

    if true_name == pred_name:
        true_positive += 1

final_acc = true_positive * 100 / len(y_pred)
print("\nFinal Accuracy:", final_acc)

plot_gallery(x_test, prediction_titles, h, w)
plt.show()


# Imposter Testing

print("\n===== Imposter Testing =====")

threshold = 0.90

if len(imposter_x) > 0:

    imposter_pca = best_pca.transform(imposter_x)
    imposter_lda = best_lda.transform(imposter_pca)

    probs = best_model.predict_proba(imposter_lda)

    for i in range(len(probs)):

        max_prob = np.max(probs[i])

        if max_prob < threshold:
            print("Imposter", i+1, "→ Not Enrolled Person")
        else:
            pred = np.argmax(probs[i])
            print("Imposter", i+1, "→ Misclassified as:", class_names[pred])
