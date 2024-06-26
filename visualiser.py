import matplotlib.pyplot as plt

values = [85,70,80,65,50,35,20,5,15,0]

x = range(len(values))

# Plot the line graph
plt.plot(x, values, marker='o', linestyle='-', color='b')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Health/ Fuel reward visualised')

# Show grid
plt.grid(True)

# Display the plot
plt.show()