import matplotlib.pyplot as plt

# Only your MAIN accuracy
labels = ["Test Accuracy"]
values = [78.76]

plt.figure()
plt.bar(labels, values)
plt.title("Model Test Accuracy")
plt.ylabel("Accuracy (%)")

# Show value on top
for i, v in enumerate(values):
    plt.text(i, v + 1, f"{v}%", ha='center')

plt.ylim(0, 100)

plt.savefig("main_accuracy.png")  # 👈 file will appear in explorer
plt.show()