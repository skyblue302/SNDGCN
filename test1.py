import matplotlib.pyplot as plt

# Data for Kl, Valid Acc, and F1-Score
Kl_values = [1.5, 1.8, 2.0, 2.2, 2.4]
valid_acc = [98.77, 98.68, 98.82, 98.63, 98.63]
f1_score = [98.79, 98.70, 98.84, 98.65, 98.66]

# Plotting Kl vs Valid Acc and F1-Score
plt.figure(figsize=(8, 6))
plt.plot(Kl_values, valid_acc, marker='o', label='Valid Acc (%)', color='b')
plt.plot(Kl_values, f1_score, marker='o', label='F1-Score (%)', color='g')

# Adding labels and title
plt.xlabel('Kl Values')
plt.ylabel('Percentage (%)')
plt.title('Kl vs Valid Acc and F1-Score')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()
