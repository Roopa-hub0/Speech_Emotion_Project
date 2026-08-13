import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))
plt.pie([4,3,2], labels=["calm","angry","sad"], autopct='%1.1f%%')
plt.title("Test Chart")
plt.savefig("static/test_chart.png")
plt.close()

print("Chart Created")